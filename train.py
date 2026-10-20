import argparse
import os
import sys
import logging
from pathlib import Path
import numpy as np
from tqdm import tqdm

import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))    
MODELS_DIR = os.path.join(BASE_DIR, "models")                
if MODELS_DIR not in sys.path:
    sys.path.insert(0, MODELS_DIR)

from datautils_best import PCSOD
from utils import inplace_relu, bn_momentum_adjust
from text_guided_runner import TextGuidedPSODRunner

def parse_args():
    parser = argparse.ArgumentParser('Training')
    parser.add_argument('--data_root', type=str, default='./data')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epoch', type=int, default=400)
    parser.add_argument('--learning_rate', type=float, default=0.0005) 
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--log_dir', type=str, default='MLLM-PCSOD')
    parser.add_argument('--npoint', type=int, default=4096)
    parser.add_argument('--optimizer', type=str, default='AdamW')
    parser.add_argument('--decay_rate', type=float, default=0.05)
    parser.add_argument('--step_size', type=int, default=200)
    parser.add_argument('--txt_train_dir', type=str, default='./data/text/train')
    parser.add_argument('--pcsod_ckpt', type=str, default='')
    parser.add_argument('--resume', action='store_true', default=False, help='whether to resume from best_model.pth')
    return parser.parse_args()

def _stem(name: str) -> str: return os.path.splitext(name)[0]

def cmtnet_data_augmentation(pts):
    B, N, C = pts.shape
    device = pts.device
    aug_pts = pts.clone()

    flip_x = (torch.rand(B, device=device) > 0.5).float() * 2 - 1.0
    flip_y = (torch.rand(B, device=device) > 0.5).float() * 2 - 1.0
    
    aug_pts[:, :, 0] *= flip_x.unsqueeze(1)
    aug_pts[:, :, 1] *= flip_y.unsqueeze(1)
    if C >= 9:
        aug_pts[:, :, 6] *= flip_x.unsqueeze(1)
        aug_pts[:, :, 7] *= flip_y.unsqueeze(1)

    if C >= 6:
        rgb = aug_pts[:, :, 3:6]
        
        if rgb.max() > 2.0:
            rgb = rgb / 255.0
        
        auto_contrast_mask = torch.rand(B, device=device) > 0.5
        for b in range(B):
            if auto_contrast_mask[b]:
                min_c = rgb[b].min(dim=0, keepdim=True)[0]
                max_c = rgb[b].max(dim=0, keepdim=True)[0]
                diff = torch.clamp(max_c - min_c, min=1e-5)
                rgb[b] = (rgb[b] - min_c) / diff
        
        noise = torch.randn(B, N, 1, device=device) * 0.05
        rgb = torch.clamp(rgb + noise, 0.0, 1.0)
        
        aug_pts[:, :, 3:6] = rgb

    return aug_pts

def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(1)

    experiment_dir = Path('./log').joinpath(args.log_dir)
    checkpoints_dir = experiment_dir.joinpath('checkpoints')
    checkpoints_dir.mkdir(exist_ok=True, parents=True)
    log_dir = experiment_dir.joinpath('logs')
    log_dir.mkdir(exist_ok=True)

    logger = logging.getLogger("Train")
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(log_dir / 'train.txt')
    fh.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    logger.addHandler(fh)
    def log_string(s):
        print(s)
        logger.info(s)

    TRAIN_DATASET = PCSOD(split='train', data_root=args.data_root, num_point=args.npoint, transform=None)
    train_loader = torch.utils.data.DataLoader(
        TRAIN_DATASET, batch_size=args.batch_size, shuffle=True,
        num_workers=8, pin_memory=True, drop_last=True,
        persistent_workers=True
    )
    weights = torch.Tensor(TRAIN_DATASET.labelweights).to(device)

    classifier = TextGuidedPSODRunner(num_classes=2, device="cuda").to(device)

    base_params = []
    gamma_params = []

    for name, p in classifier.named_parameters():
        if not p.requires_grad: 
            continue
            
        if 'gamma' in name:
            gamma_params.append(p)
        else:
            base_params.append(p)
    
    optimizer = torch.optim.AdamW([
        {'params': base_params, 'lr': args.learning_rate, 'weight_decay': args.decay_rate},
        {'params': gamma_params, 'lr': args.learning_rate, 'weight_decay': 0.0},
    ])
            
    classifier.apply(inplace_relu)
    MODEL = __import__('models.ptsod_textguided', fromlist=['get_loss'])
    criterion = MODEL.get_loss().to(device)

    WARMUP_EPOCHS = 10
    best_train_mae = float('inf')
    start_epoch = 0

    best_model_path = checkpoints_dir / 'best_model.pth'

    def restore_checkpoint(path, preserve_lr=False):
        checkpoint = torch.load(path, map_location='cpu')
        current_lrs = [group['lr'] for group in optimizer.param_groups]
        for name, tensor in checkpoint['model_state_dict'].items():
            if not torch.isfinite(tensor).all().item():
                raise RuntimeError(f'Checkpoint has non-finite model tensor: {name}')
        classifier.load_state_dict(checkpoint['model_state_dict'])
        optimizer.zero_grad(set_to_none=True)
        optimizer.state.clear()
        if 'optimizer_state_dict' in checkpoint:
            for state in checkpoint['optimizer_state_dict']['state'].values():
                for name, value in state.items():
                    if torch.is_tensor(value) and not torch.isfinite(value).all().item():
                        raise RuntimeError(f'Checkpoint has non-finite optimizer tensor: {name}')
            optimizer_state = checkpoint['optimizer_state_dict']
            saved_groups = optimizer_state['param_groups']
            if len(saved_groups) == 4 and not saved_groups[2]['params'] and not saved_groups[3]['params']:
                optimizer_state['param_groups'] = saved_groups[:2]
            optimizer.load_state_dict(optimizer_state)
        if preserve_lr:
            for group, current_lr in zip(optimizer.param_groups, current_lrs):
                group['lr'] = current_lr
        return (
            int(checkpoint.get('epoch', 0)),
            float(checkpoint.get('best_train_mae', float('inf'))),
        )
    if args.resume:
        if not best_model_path.exists():
            raise FileNotFoundError(f'指定了 --resume，但存档不存在：{best_model_path}')
        start_epoch, best_train_mae = restore_checkpoint(best_model_path)

    for epoch in range(start_epoch, args.epoch):
        
        if epoch < WARMUP_EPOCHS:
            lr = args.learning_rate * (epoch + 1) / WARMUP_EPOCHS
        else:
            progress = (epoch - WARMUP_EPOCHS) / (args.epoch - WARMUP_EPOCHS)
            lr = 0.5 * args.learning_rate * (1.0 + np.cos(np.pi * progress))
        
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        momentum = max(0.01, 0.9 * (0.5 ** (epoch // args.step_size)))
        classifier.apply(lambda x: bn_momentum_adjust(x, momentum))
        
        classifier.train()
        loss_sum, mae_sum = 0.0, 0.0
        valid_batches = 0

        for batch in tqdm(train_loader, smoothing=0.9):
            optimizer.zero_grad(set_to_none=True)
            local_pts, local_lbl, full_pts, full_lbl, coord_min, coord_max, scene_name = batch
            
            local_pts = local_pts.to(device, non_blocking=True)
            local_lbl = local_lbl.to(device, non_blocking=True)
            txt_paths = [os.path.join(args.data_root, args.txt_train_dir, f"{_stem(n)}.txt") for n in scene_name]
            
            local_pts = cmtnet_data_augmentation(local_pts)
            
            x9_4096 = local_pts.transpose(2, 1).contiguous()
            
            seg_pred_local, seg_o_local, aux_preds_list, pre_cls_features, bound_pred, xyz_pts, x_dec, txt_bkd = classifier(
                x9_4096, 
                full_pts, 
                txt_paths
            )
            
            total_loss = criterion(
                pred_log_softmax=seg_pred_local, 
                target=local_lbl, 
                weights=weights,
                bound_pred=bound_pred,
                xyz=xyz_pts,
                dense_features_64=x_dec,
                text_features=txt_bkd
            )
            

            total_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                classifier.parameters(), max_norm=0.5
            )
            optimizer.step()

            valid_batches += 1
            loss_sum += total_loss.item()
            with torch.no_grad():
                prob_fg_local = torch.exp(seg_pred_local)[:, :, 1].reshape(-1)
                mae_sum += torch.mean(
                    torch.abs(local_lbl.reshape(-1).float() - prob_fg_local)
                ).item()


        if valid_batches == 0:
            continue

        avg_loss, avg_mae = loss_sum / valid_batches, mae_sum / valid_batches
        
        poisoned = False
        for param in classifier.state_dict().values():
            if torch.isnan(param).any() or torch.isinf(param).any():
                poisoned = True; break

        log_string(
            f"Epoch {epoch+1} | Train Loss = {avg_loss:.6f} | "
            f"Train MAE = {avg_mae:.6f}"
        )
        if avg_mae < best_train_mae and not poisoned:
            best_train_mae = avg_mae
            savepath = checkpoints_dir / 'best_model.pth' 
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_train_mae': best_train_mae,
            }, savepath)
        if (epoch + 1) % 200 == 0:
            savepath = checkpoints_dir / f'model_epoch_{epoch+1}.pth'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_train_mae': best_train_mae,
            }, savepath)

if __name__ == '__main__':
    main(parse_args())
