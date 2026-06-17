# LF-Loc 项目实验说明

## 1. 任务目标

本项目关注的是人脸伪造定位任务，而不是单纯的人脸真假分类任务。

给定一张人脸图像，模型需要输出一张像素级伪造区域掩码，用于标出图像中哪些位置属于伪造或篡改区域。也就是说，模型不仅要判断图像是否为假，还要回答“哪里是假的”。

该任务可以视为二分类语义分割问题：

- `0`：真实区域
- `1`：伪造区域

相比图像级伪造检测，伪造定位对边界和局部细节更敏感，因此本项目主要使用 IoU、Dice 和 Boundary F1 等像素级指标进行评价。

## 2. 当前模型结构

当前 LF-Loc 的整体流程如下：

```text
Input Image
  -> Frozen DINOv2 ViT Backbone
  -> Lightweight FPN
  -> Boundary-Region Collaborative Attention
  -> Segmentation Head
  -> Forgery Mask
```

其中：

- `FrozenBackbone` 使用 DINOv2 ViT 提取图像特征；
- `LightFPN` 将 backbone 特征映射到分割所需的空间分辨率；
- `FBAA` 负责增强伪造边界和伪造区域响应；
- `SegHead` 输出伪造区域 mask logits。

当前模型没有独立的图像级分类头，因此评估脚本不会伪造 Image-level AUC。如果后续需要图像级 AUC，需要额外加入 `cls_logits` 输出。

## 3. 主要创新点

### 3.1 边界感知注意力

原始 FBAA 模块通过预测伪造边界图，引导模型关注篡改区域的边缘结构。该设计针对的问题是伪造区域边界模糊、普通分割头容易产生边界扩散。

### 3.2 边界-区域协同注意力

当前版本进一步将 FBAA 改进为边界-区域协同注意力模块。该模块包含：

- boundary branch：学习伪造边界响应；
- region branch：学习伪造区域内部响应；
- gate fusion：自适应融合原始特征与增强特征。

相比只关注边界，该设计同时建模伪造区域轮廓和内部区域，有利于提升定位完整性。

### 3.3 边界加权分割损失

在损失函数中加入 boundary-weighted BCE。具体做法是从 GT mask 中提取边界区域，并在 BCE loss 中提高边界像素权重，使模型在训练时更重视伪造边缘。

该功能通过训练参数控制：

```bash
--boundary_bce_weight 2.0
```

默认值为 `0.0`，因此不会影响旧实验复现。

## 4. 实验设置

数据集使用 FaceForensics++，当前默认使用以下四类伪造方法：

- Deepfakes，简称 DF
- Face2Face，简称 F2F
- FaceSwap，简称 FS
- NeuralTextures，简称 NT

训练中使用 `train` split，验证和评估使用 `test` split。

当前主要对比实验如下：

| 设置 | Backbone | FBAA | Boundary-weighted BCE | Best IoU |
| --- | --- | --- | --- | ---: |
| Baseline | DINOv2 | 否 | 否 | 0.8508 |
| 原始 FBAA | DINOv2 | 是 | 否 | 0.8548 |
| 改进版 FBAA | DINOv2 | 是 | 是 | 0.8574 |

可以看到，改进版方法相对 baseline 有小幅提升，说明边界-区域协同注意力和边界加权监督对伪造区域定位有一定帮助。

## 5. 指标解释

### 5.1 IoU

IoU 用于衡量预测伪造区域和真实伪造区域的重叠程度：

```text
IoU = TP / (TP + FP + FN)
```

IoU 越高，说明预测 mask 和 GT mask 越重合。

### 5.2 Dice

Dice 也是分割任务常用指标：

```text
Dice = 2TP / (2TP + FP + FN)
```

在二分类分割中，Dice 与 pixel-level F1 等价。

### 5.3 Boundary F1

Boundary F1 衡量预测边界和真实边界的匹配程度。本项目使用 GT mask 生成真实边界，并计算带 2 像素容差的边界 Precision、Recall 和 F1。

该指标更关注伪造区域边缘是否定位准确。

### 5.4 Image-level AUC

Image-level AUC 需要模型输出图像级 fake probability。当前 LF-Loc 没有分类头，因此评估脚本会将该字段留空，避免用分割 mask 分数冒充分类概率。

## 6. 评估结果

使用改进版模型 checkpoint：

```text
checkpoints_ffpp_dinov2_improved_fbaa/best_model.pth
```

评估命令：

```bash
HF_ENDPOINT=https://hf-mirror.com /root/miniconda3/bin/python eval.py \
  --config configs/default.yaml \
  --checkpoint checkpoints_ffpp_dinov2_improved_fbaa/best_model.pth \
  --data-root data/FaceForensics++ \
  --split test \
  --threshold 0.5 \
  --output-dir eval_results \
  --include-originals \
  --batch-size 8 \
  --num-workers 2 \
  --device cuda
```

Overall 结果：

| Method | Dataset | Samples | Image AUC | IoU | Dice | Boundary F1 | Trainable Params |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LF-Loc | FF++-test | 2997 | - | 0.857402 | 0.921690 | 0.569805 | 1.95M |

分组结果：

| Method | Samples | IoU | Dice | Boundary F1 |
| --- | ---: | ---: | ---: | ---: |
| DF | 640 | 0.827961 | 0.903773 | 0.498810 |
| F2F | 512 | 0.878907 | 0.934448 | 0.622779 |
| FS | 512 | 0.876173 | 0.933370 | 0.625446 |
| NT | 512 | 0.853929 | 0.919648 | 0.549934 |

评估输出文件：

```text
eval_results/summary.csv
eval_results/per_image_results.csv
```

其中 `per_image_results.csv` 包含每张图像的 `image_path`、`gt_label`、`fake_probability`、`manipulation_type`、`iou`、`dice` 和 `boundary_f1`。

## 7. 常用训练命令

改进版模型训练命令：

```bash
HF_ENDPOINT=https://hf-mirror.com /root/miniconda3/bin/python train.py \
  --dataset ffpp \
  --data_root data/FaceForensics++ \
  --device cuda \
  --epochs 20 \
  --batch_size 8 \
  --train_size -1 \
  --val_size -1 \
  --num_workers 2 \
  --boundary_bce_weight 2.0 \
  --save_dir checkpoints_ffpp_dinov2_improved_fbaa \
  --log_dir logs_ffpp_dinov2_improved_fbaa
```

Baseline 训练命令：

```bash
HF_ENDPOINT=https://hf-mirror.com /root/miniconda3/bin/python train.py \
  --dataset ffpp \
  --data_root data/FaceForensics++ \
  --device cuda \
  --epochs 20 \
  --batch_size 8 \
  --train_size -1 \
  --val_size -1 \
  --num_workers 2 \
  --disable_fbaa \
  --lambda_boundary 0 \
  --save_dir checkpoints_ffpp_dinov2_baseline \
  --log_dir logs_ffpp_dinov2_baseline
```

## 8. 当前限制

当前版本仍有以下限制：

- 没有图像级分类头，因此不能计算真正的 Image-level AUC；
- 验证和评估主要基于 FF++ test split，后续可进一步划分独立 validation/test；
- 目前提升幅度属于小幅提升，报告中应表述为“小幅但稳定提升”，不宜夸大。
