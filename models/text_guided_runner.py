import os
import torch
import torch.nn as nn
from typing import List
from models.ptsod_textguided import get_model as get_seg_model
class TextGuidedPSODRunner(nn.Module):
    def __init__(
        self,
        num_classes: int = 2,
        device: str = "cuda",
        pcsod_ckpt_path: str = None, 
        temperature: float = 0.07,
        topk_points: int = 16384  
    ):
        super().__init__()
        self.device = torch.device(device)
        self.seg_model = get_seg_model(num_classes=num_classes).to(self.device)

    def _get_prompt_from_txt(self, txt_path: str) -> str:
        if not os.path.exists(txt_path):
            raise FileNotFoundError(f"找不到对应的文本文件 {txt_path}！请检查你的数据集。")
        with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read().strip().rstrip('.')
            if not text:
                raise ValueError(f"文本文件内容为空 {txt_path}！请补充文本。")
            return text

    def forward(self, x9_4096: torch.Tensor, pc_240k: torch.Tensor, 
                txt_paths: List[str], full_labels: torch.Tensor = None):

        B = x9_4096.shape[0]

        prompts = []
        for b in range(B):
            text_prompt = self._get_prompt_from_txt(txt_paths[b])
            prompts.append(text_prompt)

        text_feats = self.seg_model.text_encoder(prompts, x9_4096.device)
        
        outputs = self.seg_model(x9_4096, text_list=None, text_feats=text_feats)
        
        if len(outputs) == 8:
            pred, pred_o, aux_log_softmax, pre_cls_features, bound_pred, xyz_pts, x_dec, txt_bkd = outputs
            return pred, pred_o, aux_log_softmax, pre_cls_features, bound_pred, xyz_pts, x_dec, txt_bkd
            
        elif len(outputs) == 6:
            pred, pred_o, aux_log_softmax, pre_cls_features, bound_pred, xyz_pts = outputs
            return pred, pred_o, aux_log_softmax, pre_cls_features, bound_pred, xyz_pts
            
        elif len(outputs) == 4:
            pred, pred_o, aux_log_softmax, pre_cls_features = outputs
            return pred, pred_o, aux_log_softmax, pre_cls_features
            
        elif len(outputs) == 3:
            pred, pred_o, aux_log_softmax = outputs
            return pred, pred_o, aux_log_softmax
            
        else:
            pred, pred_o = outputs[:2]
            return pred, pred_o
