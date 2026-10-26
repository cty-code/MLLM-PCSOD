import argparse
import os
import sys
import logging
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import open3d as o3d
from datautils import PCSOD
from text_guided_runner import TextGuidedPSODRunner
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

def parse_args():
    parser = argparse.ArgumentParser('Model Testing')
    parser.add_argument('--data_root', type=str, default='./data')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--num_point', type=int, default=4096)
    parser.add_argument('--log_dir', type=str, default='MLLM-PCSOD')
    parser.add_argument('--num_votes', type=int, default=4)
    parser.add_argument('--test_batch_size', type=int, default=16)
    parser.add_argument('--txt_test_dir', type=str, default='./data/text/test')
    parser.add_argument('--pcsod_ckpt', type=str, default='')
    return parser.parse_args()

def _stem(name: str) -> str: return os.path.splitext(name)[0]

@torch.no_grad()
def compute_metric_curves(pred, gt, num_thresholds=255):
    pred = pred.reshape(-1)
    gt = gt.reshape(-1).float()
    thresholds = torch.linspace(0, 1 - 1e-10, num_thresholds, device=pred.device)
    f_scores = torch.zeros(num_thresholds, device=pred.device)
    e_scores = torch.zeros_like(f_scores)
    iou_scores = torch.zeros_like(f_scores)
    beta2 = 0.3
    centered_gt = gt - gt.mean()
    for i, threshold in enumerate(thresholds):
        binary_pred = (pred >= threshold).float()
        tp = (binary_pred * gt).sum()
        precision = tp / (binary_pred.sum() + 1e-20)
        recall = tp / (gt.sum() + 1e-20)
        f_score = (1 + beta2) * precision * recall / (beta2 * precision + recall)
        f_scores[i] = torch.where(torch.isnan(f_score), torch.zeros_like(f_score), f_score)
        centered_pred = binary_pred - binary_pred.mean()
        alignment = 2 * centered_gt * centered_pred / (
            centered_gt * centered_gt + centered_pred * centered_pred + 1e-20
        )
        enhanced = ((alignment + 1) * (alignment + 1)) / 4
        e_scores[i] = enhanced.sum() / (gt.numel() - 1 + 1e-20)
        intersection = ((binary_pred == 1) & (gt == 1)).sum()
        union = ((binary_pred == 1) | (gt == 1)).sum()
        iou_scores[i] = intersection / union
    return f_scores, e_scores, iou_scores


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(6)
    experiment_dir = Path('log') / args.log_dir
    logger = logging.getLogger("Test")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    file_handler = logging.FileHandler(experiment_dir / 'test.txt')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(message)s'))
    logger.addHandler(file_handler)
    
    def log_string(str): logger.info(str); print(str)
    NUM_POINT, NUM_CLASSES, NUM_VOTE = args.num_point, 2, args.num_votes
    TEST_DATASET = PCSOD(split='test', data_root=args.data_root, num_point=NUM_POINT, transform=None)
    test_loader = torch.utils.data.DataLoader(
        TEST_DATASET, batch_size=1, shuffle=False,
        num_workers=10, pin_memory=True, drop_last=False,
        persistent_workers=True
    )
    classifier = TextGuidedPSODRunner(num_classes=NUM_CLASSES, device="cuda").to(device)
    model_path = experiment_dir / 'checkpoints' / '.pth'
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint.get('model_state_dict', checkpoint)

    if any(key.startswith('seg_model.') for key in state_dict):
        classifier.load_state_dict(state_dict)
    else:
        classifier.seg_model.load_state_dict(state_dict)
    classifier.eval()
    with torch.no_grad():
        mae_list = []
        f_curve_sum = torch.zeros(255, device=device)
        e_curve_sum = torch.zeros(255, device=device)
        iou_curve_sum = torch.zeros(255, device=device)
        metric_scene_count = 0
        
        for batch in tqdm(test_loader, total=len(test_loader), smoothing=0.9):
            local_pts, local_lbl, full_pts, full_lbl, coord_min, coord_max, scene_names = batch
            scene_name = scene_names[0]
            full_lbl_1 = full_lbl.squeeze(0)  
            pc_240k_1 = full_pts.squeeze(0).to(device)  

            N_all = full_lbl_1.shape[0]
            target_1 = full_lbl_1.to(device).float()

            mod = N_all % NUM_POINT
            NUM_ADD = (NUM_POINT - mod) if (mod != 0) else 0
            NUM_BATCH = (N_all + NUM_ADD) // NUM_POINT
            final_pred = torch.zeros(N_all, device=device)

            txt_path_1 = os.path.join(args.data_root, args.txt_test_dir, f"{_stem(scene_name)}.txt")
            top1_text_1 = classifier._get_prompt_from_txt(txt_path_1)

            for _ in range(NUM_VOTE):
                idx = torch.randperm(N_all, device=device)
                if NUM_ADD > 0: idx = torch.cat([idx, torch.arange(NUM_ADD, device=device)], dim=0)

                pts = pc_240k_1[idx]
                chunks = torch.chunk(pts, NUM_BATCH, dim=0)
                pts_b = torch.stack(chunks, dim=0) 
                
                xyz = pts_b[:, :, :3].clone()   
                rgb = pts_b[:, :, 3:6].clone()  
                
                if torch.max(rgb) > 2.0: 
                    rgb = rgb / 255.0
                
                c_min = coord_min.to(device).unsqueeze(1) 
                c_max = coord_max.to(device).unsqueeze(1) 
                
                norm_xyz = (xyz - c_min) / (c_max - c_min + 1e-6)
                
                center = (c_min + c_max) / 2.0
                xyz = xyz - center
                
                if _ > 0:
                    flip_x = (torch.rand(xyz.shape[0], device=device) > 0.5).float() * 2 - 1.0
                    flip_y = (torch.rand(xyz.shape[0], device=device) > 0.5).float() * 2 - 1.0
                    
                    xyz[:, :, 0] *= flip_x.unsqueeze(1)
                    xyz[:, :, 1] *= flip_y.unsqueeze(1)
                    norm_xyz[:, :, 0] *= flip_x.unsqueeze(1)
                    norm_xyz[:, :, 1] *= flip_y.unsqueeze(1)
                

                x9 = torch.cat([xyz, rgb, norm_xyz], dim=-1).permute(0, 2, 1).contiguous()

                top1_texts_b = [top1_text_1] * x9.shape[0]
                text_feats_b = classifier.seg_model.text_encoder(top1_texts_b, x9.device).unsqueeze(1)
                
                NUM_TEST = NUM_BATCH // args.test_batch_size + (1 if NUM_BATCH % args.test_batch_size != 0 else 0)
                seg_pred_shuffle = None
                
                for i in range(NUM_TEST):
                    start_idx, end_idx = i * args.test_batch_size, min((i + 1) * args.test_batch_size, NUM_BATCH)
                    x9_batch = x9[start_idx:end_idx]
                    text_feats_batch = text_feats_b[start_idx:end_idx]
                    
                    if x9_batch.shape[0] != text_feats_batch.shape[0]:
                        text_feats_batch = text_feats_b[:x9_batch.shape[0]]

                    outputs = classifier.seg_model(x9_batch, text_list=None, text_feats=text_feats_batch)
                    seg_o = outputs[1]
                    seg_pred_shuffle = seg_o if seg_pred_shuffle is None else torch.cat([seg_pred_shuffle, seg_o], dim=0)
                        
                prob_fg = F.softmax(seg_pred_shuffle, dim=-1)[:, :, 1] 
                prob_fg_combined = torch.cat(torch.chunk(prob_fg, prob_fg.shape[0], dim=0), dim=1).squeeze()
                final_pred[idx[:N_all]] += prob_fg_combined.view(-1, 1)[:N_all, 0]

            final_pred /= float(NUM_VOTE)
            
            pc_xyz = full_pts.squeeze(0)[:, :3].cpu().numpy()
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pc_xyz)
            
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=16))
            
            pts_gpu = full_pts.squeeze(0)[:, :3].to(device)
            norms_gpu = torch.tensor(np.asarray(pcd.normals), device=device, dtype=torch.float32) 
            preds_gpu = final_pred.clone() 
            
            K = 512
            normal_threshold = 0.90
            
            uncertain_mask = (preds_gpu > 0.2) & (preds_gpu < 0.8)
            uncertain_indices = torch.nonzero(uncertain_mask).squeeze(1) 
            
            smoothed_preds = preds_gpu.clone()
            
            chunk_size = 2048 
            
            for start in range(0, len(uncertain_indices), chunk_size):
                end = min(start + chunk_size, len(uncertain_indices))
                idx_chunk = uncertain_indices[start:end]
                
                query_pts = pts_gpu[idx_chunk]     
                query_norms = norms_gpu[idx_chunk] 
                
                dists = torch.cdist(query_pts, pts_gpu) 
                
                _, knn_idx = torch.topk(dists, k=K, dim=1, largest=False) 
                
                neighbor_norms = norms_gpu[knn_idx] 
                neighbor_preds = preds_gpu[knn_idx] 
                
                dot_prods = torch.abs(torch.sum(query_norms.unsqueeze(1) * neighbor_norms, dim=2)) 
                
                valid_mask = dot_prods > normal_threshold 
                
                valid_preds = neighbor_preds * valid_mask.float()
                valid_counts = valid_mask.float().sum(dim=1) 
                
                avg_preds = torch.where(
                    valid_counts > 0, 
                    valid_preds.sum(dim=1) / valid_counts, 
                    preds_gpu[idx_chunk]
                )
                
                smoothed_preds[idx_chunk] = avg_preds
                
            final_pred = smoothed_preds
            
            mae = torch.mean(torch.abs(final_pred - target_1)).item()
            
            pc_vector_points = full_pts.squeeze(0)[:, :3].cpu().numpy() 
            pc_vector = o3d.geometry.PointCloud()
            pc_vector.points = o3d.utility.Vector3dVector(pc_vector_points)
            
            preds_np = final_pred.cpu().numpy()
            pc_vector_colors = np.stack((preds_np, np.zeros_like(preds_np), np.zeros_like(preds_np)), axis=-1)
            pc_vector.colors = o3d.utility.Vector3dVector(pc_vector_colors)
            
            visual_dir = Path(experiment_dir) / "visual"
            visual_dir.mkdir(exist_ok=True, parents=True)
            visual_path = str(visual_dir / f"{scene_name}")
            o3d.io.write_point_cloud(visual_path, pc_vector)
            
            if not np.isnan(mae):
                mae_list.append(mae)
                f_curve, e_curve, iou_curve = compute_metric_curves(final_pred, target_1)
                f_curve_sum += f_curve
                e_curve_sum += e_curve
                iou_curve_sum += iou_curve
                metric_scene_count += 1

                logger.info(f"Scene: {scene_name} | MAE: {mae:.4f}")

        avg_mae = float(np.mean(mae_list)) if mae_list else float("nan")

        log_string("\n" + "=" * 50)
        log_string(f"MAE: {avg_mae:.6f}")
        log_string("=" * 50)

if __name__ == '__main__':
    main(parse_args())
