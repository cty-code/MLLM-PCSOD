<div align="center">

# MLLM-PCSOD: A Semantic Prior-Driven Framework for Point Cloud Salient Object Detection via Multimodal Large Language Model

**Tianyu Cheng¹　Ziwei Luo¹、Zhaojing Wang¹、Xiaoyun Yan¹　Li Li¹**  
¹ Wuhan Textile University

论文：*Coming Soon*　|　代码：*Coming Soon*　|　数据：*Coming Soon*

</div>

---

## Abstract

Existing point cloud salient object detection methods all follow a purely data-driven paradigm, relying solely on local geometric and color features for bottom-up saliency inference. Lacking high-level semantic understanding, they commonly suffer  from  background  false  activation  and  object  omission  in complex  backgrounds,  for  small  objects,  and  in  low-contrast scenarios.  To  fundamentally  break  this  limitation,  we  propose a novel semantic prior-driven paradigm and design the MLLMPCSOD model based on it, adopting a top-down strategy of“semantic understanding first, then geometric segmentation”. Specifically,  a  multimodal  large  language  model,  LLaVA-OneVision, is  efficiently  fine-tuned  with  saliency-aware  prompts  and  2D point  cloud  projection  to  parse  scene  semantics  and  output salient object category text; the corresponding text features then guide a PointNeXt-based structure to detect saliency through a lightweight text-guided point-wise  attention module  and threestage cross-modal alignment mechanism. Comprehensive experiments  show  that  MLLM-PCSOD  outperforms  all  representative  methods  on  F-measure,  E-measure,  and  intersection  over union, validating the effectiveness of the semantic prior-driven paradigm and showing that saliency detection performance can be markedly improved at a negligible cross-modal alignment cost.

## Highlights

- We propose for the first time a novel semantic prior-driven paradigm for PCSOD task, which breaks through the fundamental limitation of existing purely data-driven paradigms through a top-down strategy of “semantic understanding first, then geometric segmentation.” Based on this paradigm, we design the MLLM-PCSOD model, offering a promising research direction for this field.
- We design a dedicated saliency-aware prompt and instruction-tune a MLLM to output the categories of salient objects as text, providing MLLM-derived high-level semantic priors for PCSOD task.
- We propose highly lightweight TGPA modules (only 0.097M trainable parameters in total) and a three-stage cross-modal alignment mechanism to inject textual semantic priors into saliency prediction.
- Extensive experiments demonstrate the clear advantages of the proposed paradigm and model and reveal that PCSOD performance can be markedly improved at negligible cross-modal alignment cost.

## Method

![Overview of MLLM-PCSOD](fig1.png)

MLLM-PCSOD consists of two branches:
1. **Semantic prior extraction branch：** The semantic prior extraction branch (top) employs a LoRA-fine-tuned LLaVA-OneVision and a frozen CLIP text encoder to produce saliency-related text features from the 2D point cloud projection.
2. **Semantic prior-driven segmentation branch：** The semantic prior-driven segmentation branch (bottom) builds a PointNeXt-based network and injects the saliency-related text features at three key positions via the TGPA modules to predict saliency.

## Results

定量实验、消融实验和可视化结果将在完成后补充。

| Dataset | Metric 1 | Metric 2 | Metric 3 |
|:-------:|:--------:|:--------:|:--------:|
| PCSOD | — | — | — |

## Dataset

You can download the PCSOD dataset from [OpenI](https://openi.pcl.ac.cn/OpenPointCloud/PCSOD) and place it in the following directory:

## Code

训练、测试代码和预训练模型将在整理完成后发布。

```text
VLM3DSOD\
├── data\
│   ├── train\
│   └── test\
├── log\
|   ├── pcsod_textguided\
|       ├── logs\
|       └── checkpoints\
|           └── best_model.pth
├── models/
├── train.py
├── test.py
├── requirements.txt
└── ...

```

## Running

### Training

```bash
python train.py
```

### Testing

```bash
python test.py
```


## Citation

如果本工作对您的研究有所帮助，请引用：

```bibtex
@article{vlm3dsod2026,
  title  = {3D Point Cloud Salient Object Detection with Semantic Priors from Vision-Language Models},
  author = {Author Name and Coauthor Name},
  year   = {2026}
}
```

## Acknowledgement

感谢所有为本研究提供帮助的项目、数据集及研究工作。详细致谢信息将在后续补充。

