<div align="center">

# MLLM-PCSOD: A Semantic Prior-Driven Framework for Point Cloud Salient Object Detection via Multimodal Large Language Model

**作者姓名¹　合作者姓名¹　指导教师¹**  
¹ 单位名称 · 学院或实验室

论文：*Coming Soon*　|　代码：*Coming Soon*　|　数据：*Coming Soon*

</div>

---

## Abstract

本研究面向三维点云场景中的显著物体检测任务，探索如何利用视觉语言模型提供的高层语义先验，增强模型对复杂场景中目标语义与几何结构的联合理解。此处为摘要占位文本，待论文内容确定后可随时替换。

## Highlights

- 利用视觉语言模型提取开放词汇语义先验。
- 融合二维语义信息与三维点云几何特征。
- 在复杂三维场景中实现点级显著性预测。

## Method

> **方法框架图即将补充。**  
> 后续上传框架图后，可将本段替换为：`![Method](fig1.jpg)`

本方法主要包括以下三个阶段：

1. **语义先验提取：** 利用视觉语言模型获取高层语义表征。
2. **跨模态特征融合：** 对齐并融合语义信息与三维几何特征。
3. **显著性预测：** 输出点级显著性检测结果。

## Results

定量实验、消融实验和可视化结果将在完成后补充。

| Dataset | Metric 1 | Metric 2 | Metric 3 |
|:-------:|:--------:|:--------:|:--------:|
| Dataset A | — | — | — |
| Dataset B | — | — | — |

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

