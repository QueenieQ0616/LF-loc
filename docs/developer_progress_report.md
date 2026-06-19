# LF-Loc 开发者进展文档

## 1. 项目当前阶段

LF-Loc 当前已经从最初的轻量分割 baseline，推进到一个可以在 FaceForensics++ 上完成训练、评估、消融和横向对比的伪造区域定位项目。

当前项目主任务是 **Face Forgery Localization**，也就是给定一张人脸图像，输出像素级伪造区域 mask。它不是单纯判断图片真假，而是回答“哪里是假的”。因此项目的核心指标是 IoU、Dice、PBCA、Boundary F1 等定位/分割指标；Image-level AUC 作为补充指标，用于和 Face X-ray、FakeLocator 这类图像级检测方法做横向参考。

当前已经完成：

- FF++ 数据集接入；
- DINOv2 backbone + 轻量 FPN + FBAA + segmentation head 主模型；
- 改进版 FBAA；
- boundary-weighted BCE；
- 图像级分类头与 Image-level AUC 评估；
- baseline / FBAA / improved FBAA 消融；
- FF++ c23 横向对比评估脚本；
- 训练、评估、结果 CSV 和实验文档。

当前最稳的定位结果来自 improved FBAA checkpoint：

```text
IoU  = 0.858376
Dice = 0.922194
PBCA = 0.897788
```

补充 balanced MLP 分类头后，Image-level AUC 可以计算，当前结果为：

```text
Overall AUC = 0.674727
```

## 2. 项目目录与核心模块

### 2.1 数据模块

相关文件：

```text
data/dataset.py
```

主要组件：

| 组件 | 作用 |
| --- | --- |
| `FakeDataset` | 用于 smoke test 的随机数据集，方便在没有真实数据时检查训练流程 |
| `FFPPDataset` | 读取 FaceForensics++ 的 frame/mask 数据 |
| `build_dataloader` | 根据参数构建 fake 或 FF++ DataLoader |
| `balanced_by_label` | 分类头训练时使用 label-balanced sampler，缓解 real/fake 样本不均衡 |

当前 FF++ 数据组织假设：

```text
data_root/
  train/
    manipulated_sequences/<method>/<compression>/frames/<video>/<frame>.png
    manipulated_sequences/<method>/<compression>/masks/<video>/<frame>.png
    original_sequences/<source>/<compression>/frames/<video>/<frame>.png
  test/
    ...
```

当前默认使用四种伪造类型：

| 简写 | 方法 |
| --- | --- |
| DF | Deepfakes |
| F2F | Face2Face |
| FS | FaceSwap |
| NT | NeuralTextures |

注意：当前代码中 `split="val"` 会映射到 FF++ 的 `test` 目录，因此目前的验证和最终测试没有严格拆分。报告中建议说明这是课程实验设置，后续严谨实验应单独划分 validation/test。

### 2.2 Backbone 模块

相关文件：

```text
models/backbone.py
```

主要组件：

| 组件 | 作用 |
| --- | --- |
| `FrozenBackbone` | 使用 timm 加载 DINOv2 或 CLIP ViT backbone |

当前默认 backbone：

```text
vit_base_patch14_dinov2
```

默认冻结 backbone，只训练轻量适配模块。这是 LF-Loc 轻量化设计的重要前提。

### 2.3 FPN 模块

相关文件：

```text
models/fpn.py
```

主要组件：

| 组件 | 作用 |
| --- | --- |
| `LightFPN` | 将 backbone 输出的低分辨率特征映射到分割头需要的空间尺度和通道数 |
| `FPNWithBackbone` | 测试用组合模块 |

当前 FPN 输出通道：

```text
fpn_out = 256
```

### 2.4 FBAA 模块

相关文件：

```text
models/fbaa.py
```

FBAA 是当前项目的主要创新模块，全称可以写作：

```text
Forgery Boundary-Aware Attention
```

当前实现已经从简单边界注意力扩展为 **Boundary-Region Collaborative Attention**，包含：

| 分支 | 作用 |
| --- | --- |
| boundary branch | 学习伪造边界响应，并输出 `boundary_logits` |
| region branch | 学习伪造区域内部响应 |
| gate fusion | 自适应融合原始特征、边界增强特征和区域增强特征 |

该模块的作用是让模型不仅关注伪造区域内部，也关注伪造区域边缘的融合痕迹。

### 2.5 分割头与分类头

相关文件：

```text
models/head.py
models/lfloc.py
```

主要组件：

| 组件 | 作用 |
| --- | --- |
| `SegHead` | 输出像素级 `mask_logits` |
| `cls_head` | 输出图像级 `cls_logits`，用于 Image-level AUC |
| `LFLOC` | 组装 backbone、FPN、FBAA、segmentation head 和 classification head |

当前 `LFLOC.forward(..., return_dict=True)` 输出：

```python
{
    "mask_logits": ...,
    "boundary_logits": ...,
    "cls_logits": ...
}
```

其中：

- `mask_logits` 用于伪造区域定位；
- `boundary_logits` 用于边界监督；
- `cls_logits` 用于图像级 Real/Fake AUC。

分类头当前使用 backbone feature 的 mean pooling + max pooling，再接轻量 MLP。训练时可以通过 `--train_cls_only` 冻结定位分支，只训练分类头。

### 2.6 损失函数模块

相关文件：

```text
losses/loss.py
```

主要组件：

| 组件 | 作用 |
| --- | --- |
| `BCELoss` | 像素级 BCE loss |
| `DiceLoss` | 缓解前景区域较小导致的不平衡 |
| `BoundaryTarget` | 从 GT mask 自动生成边界监督 |
| `BoundaryLoss` | 监督 `boundary_logits` 或 mask-derived boundary |
| `MultiTaskLoss` | 组合 segmentation、boundary 和 classification loss |

总损失形式：

```text
L_total =
  lambda_seg * (L_bce + L_dice)
  + lambda_boundary * L_boundary
  + lambda_cls * L_cls
```

另外支持 boundary-weighted BCE：

```bash
--boundary_bce_weight 2.0
```

该设置会提高边界区域像素在 BCE 中的权重，让模型更重视伪造边缘。

### 2.7 评价指标模块

相关文件：

```text
metrics/metrics.py
eval_ffpp_comparison.py
```

训练过程中的基础指标由 `MetricTracker` 统计，主要包括：

- IoU
- Dice
- F1@pixel
- PixelAcc
- ImageAUC

最终横向对比使用 `eval_ffpp_comparison.py`，输出：

- Image-level AUC；
- fake-only mean IoU；
- fake-only mean Dice；
- PBCA；
- DF/F2F/FS/NT 分组 AUC；
- `ffpp_comparison.csv`；
- `per_image_results.csv`；
- `ffpp_auc_by_type.csv`。

## 3. 训练与评估流程

### 3.1 improved FBAA 定位模型训练

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

### 3.2 baseline 训练

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

### 3.3 分类头 balanced 微调

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
  --include_originals \
  --resume checkpoints_ffpp_dinov2_improved_fbaa/best_model.pth \
  --train_cls_only \
  --balanced_cls_sampler \
  --lambda_cls 1.0 \
  --lr 1e-4 \
  --save_dir checkpoints_ffpp_cls_mlp_balanced \
  --log_dir logs_ffpp_cls_mlp_balanced
```

### 3.4 FF++ 横向对比评估

定位 checkpoint：

```bash
HF_ENDPOINT=https://hf-mirror.com /root/miniconda3/bin/python eval_ffpp_comparison.py \
  --config configs/default.yaml \
  --checkpoint checkpoints_ffpp_dinov2_improved_fbaa/best_model.pth \
  --data-root data/FaceForensics++ \
  --split test \
  --output-dir eval_results \
  --include-originals \
  --batch-size 8 \
  --num-workers 2 \
  --device cuda
```

分类头 checkpoint：

```bash
HF_ENDPOINT=https://hf-mirror.com /root/miniconda3/bin/python eval_ffpp_comparison.py \
  --config configs/default.yaml \
  --checkpoint checkpoints_ffpp_cls_mlp_balanced/best_model.pth \
  --data-root data/FaceForensics++ \
  --split test \
  --output-dir eval_results_cls_mlp_balanced \
  --include-originals \
  --batch-size 8 \
  --num-workers 2 \
  --device cuda
```

## 4. 消融实验进展

当前已经完成的主要消融如下：

| 实验 | Backbone | FBAA | Boundary-weighted BCE | 分类头 | Best Val IoU |
| --- | --- | --- | --- | --- | ---: |
| DINOv2 baseline | DINOv2 | 否 | 否 | 否 | 0.8508 |
| 原始 FBAA | DINOv2 | 是 | 否 | 否 | 0.8548 |
| improved FBAA | DINOv2 | 是 | 是 | 否 | 0.8574 |

补充实验：

| 实验 | 说明 | 结果 |
| --- | --- | ---: |
| no-pretrain baseline | 不加载 backbone 预训练权重的 baseline | Best IoU 0.8282 |
| no-pretrain full | 不加载 backbone 预训练权重的完整模型 | Best IoU 0.8255 |
| 简单分类头联合微调 | 加分类头后联合训练 | AUC 0.6428，定位指标下降 |
| 只训线性分类头 | 冻结定位分支，只训练 257 参数线性头 | AUC 0.6118，定位指标保留 |
| balanced MLP 分类头 | mean/max pooling MLP + balanced sampler，只训分类头 | AUC 0.6747，定位指标保留 |

消融结论：

1. DINOv2 预训练特征对定位任务很重要，不加载预训练权重时 IoU 会下降到约 0.83。
2. FBAA 相比 baseline 有小幅提升，说明边界/区域注意力对伪造定位有帮助。
3. improved FBAA + boundary-weighted BCE 是当前定位性能最好的配置。
4. 分类头可以补齐 Image-level AUC，但当前 AUC 仍不高，说明 LF-Loc 目前更偏定位模型，而不是强图像级真假分类器。
5. 只训练分类头可以保留定位性能；联合训练分类头会影响定位分支，导致 IoU/Dice 下降。

## 5. 评价指标说明

### 5.1 Image-level AUC

使用分类头输出的连续 fake probability 计算 ROC-AUC。Fake 为正类 1，Real 为负类 0。AUC 不能用二值化后的预测标签计算。

当前用途：

- 用于和 Face X-ray、FakeLocator 等图像级检测方法做参考对比；
- 不作为当前项目的主定位指标。

### 5.2 IoU

IoU 衡量预测伪造区域与 GT 伪造区域的重叠程度：

```text
IoU = TP / (TP + FP + FN)
```

最终评估中，IoU 只在 GT mask 非空的 fake 样本上逐张计算后取平均，避免真实图片的全黑 mask 抬高定位结果。

### 5.3 Dice

Dice 衡量预测 mask 与 GT mask 的重叠程度：

```text
Dice = 2TP / (2TP + FP + FN)
```

Dice 与 pixel-level F1 等价。

### 5.4 PBCA

PBCA 表示 Pixel-wise Binary Classification Accuracy：

```text
PBCA = (TP + TN) / (TP + TN + FP + FN)
```

PBCA 在全部有效像素上统计，会受到背景区域预测结果影响。

### 5.5 Boundary F1

Boundary F1 衡量预测边界与真实边界的匹配程度。真实边界由 GT mask 生成，评估时采用 2 像素容差。

该指标用于衡量模型是否能准确定位伪造边缘。

## 6. 当前主要结果

### 6.1 FF++ c23 测试集规模

| 类型 | 数量 |
| --- | ---: |
| Real | 821 |
| Fake | 2176 |
| Total | 2997 |

Fake 样本分布：

| 类型 | 数量 |
| --- | ---: |
| DF | 640 |
| F2F | 512 |
| FS | 512 |
| NT | 512 |

### 6.2 定位 checkpoint 结果

checkpoint：

```text
checkpoints_ffpp_dinov2_improved_fbaa/best_model.pth
```

| Method | Dataset | Threshold | Image AUC | IoU | Dice | PBCA |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| LF-Loc | FF++ c23 | 0.1 | - | 0.845594 | 0.914603 | 0.887663 |
| LF-Loc | FF++ c23 | 0.5 | - | 0.858376 | 0.922194 | 0.897788 |

### 6.3 balanced 分类头 checkpoint 结果

checkpoint：

```text
checkpoints_ffpp_cls_mlp_balanced/best_model.pth
```

| Method | Dataset | Threshold | Image AUC | IoU | Dice | PBCA |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| LF-Loc | FF++ c23 | 0.1 | 0.674727 | 0.845594 | 0.914603 | 0.887663 |
| LF-Loc | FF++ c23 | 0.5 | 0.674727 | 0.858376 | 0.922194 | 0.897788 |

AUC 分组结果：

| Method | DF AUC | F2F AUC | FS AUC | NT AUC | Overall AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| LF-Loc | 0.855523 | 0.642345 | 0.650700 | 0.505141 | 0.674727 |

### 6.4 Boundary F1 结果

早期完整评估脚本得到的 improved FBAA 结果：

| Method | Dataset | Samples | IoU | Dice | Boundary F1 | Trainable Params |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| LF-Loc | FF++ test | 2997 | 0.857402 | 0.921690 | 0.569805 | 1.95M |

分组结果：

| Type | Samples | IoU | Dice | Boundary F1 |
| --- | ---: | ---: | ---: | ---: |
| DF | 640 | 0.827961 | 0.903773 | 0.498810 |
| F2F | 512 | 0.878907 | 0.934448 | 0.622779 |
| FS | 512 | 0.876173 | 0.933370 | 0.625446 |
| NT | 512 | 0.853929 | 0.919648 | 0.549934 |

## 7. 横向对比

当前可用于报告的横向参考表：

| Method | Dataset | Image AUC | IoU | Dice | PBCA |
| --- | --- | ---: | ---: | ---: | ---: |
| Face X-ray | FF++ | 0.9852 | - | - | - |
| FakeLocator | FF++ | 0.9846 | 0.3097 | 0.0811 | 0.8687 |
| LF-Loc 定位 checkpoint | FF++ c23 | - | 0.8584 | 0.9222 | 0.8978 |
| LF-Loc balanced 分类头 checkpoint | FF++ c23 | 0.6747 | 0.8584 | 0.9222 | 0.8978 |

解读建议：

1. LF-Loc 的定位指标 IoU、Dice、PBCA 很强，尤其 IoU/Dice 高于 FakeLocator 论文中给出的参考结果。
2. Image-level AUC 明显低于 Face X-ray 和 FakeLocator，说明当前 LF-Loc 的图像级分类能力还不是优势。
3. 由于数据预处理、压缩等级、划分方式和评估协议可能不完全一致，横向对比应写作“参考对比”，不宜声称严格复现 SOTA。
4. 报告重点应放在伪造区域定位能力、轻量训练参数和 FBAA 消融贡献上。

## 8. 当前结论

当前 LF-Loc 可以总结为：

> 一个基于冻结 DINOv2 backbone 的轻量伪造区域定位模型，通过 FPN、Boundary-Region Collaborative Attention 和 boundary-weighted BCE，在 FF++ c23 上取得较好的像素级定位结果。模型在 IoU、Dice、PBCA 上表现稳定，但图像级 AUC 仍有提升空间。

可以写进报告的贡献点：

1. 构建了一个冻结预训练视觉 backbone 的轻量伪造定位框架，主要训练轻量 FPN、FBAA 和 head。
2. 设计并实现 Boundary-Region Collaborative Attention，同时建模伪造边界和区域内部响应。
3. 引入 boundary-weighted BCE，加强模型对伪造边缘区域的监督。
4. 完成 FF++ 数据接入、消融实验、横向评估和可导出的 CSV 结果。
5. 补充图像级分类头，使项目可以计算 Image-level AUC，但当前 AUC 不是主要优势。

## 9. 当前限制与后续方向

### 9.1 当前限制

- 验证集与测试集目前都来自 FF++ test 目录，严格实验需要重新划分 validation/test。
- Image-level AUC 只有 0.674727，明显低于 Face X-ray/FakeLocator 的 0.98 级别。
- 当前创新主要集中在定位模块，分类分支仍比较简单。
- 尚未完成高频伪影分支、跨压缩等级鲁棒性实验、可视化案例分析。

### 9.2 后续优先级

建议后续按以下顺序推进：

1. 固定当前 improved FBAA 定位模型，整理报告和可视化结果。
2. 单独划分 validation/test，避免测试集参与 checkpoint 选择。
3. 补充 qualitative visualization，展示 mask 预测效果。
4. 做更完整的消融：无 FBAA、原 FBAA、改进 FBAA、boundary-weighted BCE、分类头。
5. 若时间允许，加入轻量高频伪影分支。
6. 若必须提高 AUC，再考虑解冻 backbone 后几层或引入更强分类分支。

## 10. 结果文件位置

远程服务器上的主要结果文件：

```text
/root/autodl-tmp/LF-Loc/eval_results/ffpp_comparison.csv
/root/autodl-tmp/LF-Loc/eval_results/per_image_results.csv
/root/autodl-tmp/LF-Loc/eval_results/ffpp_auc_by_type.csv

/root/autodl-tmp/LF-Loc/eval_results_cls_mlp_balanced/ffpp_comparison.csv
/root/autodl-tmp/LF-Loc/eval_results_cls_mlp_balanced/per_image_results.csv
/root/autodl-tmp/LF-Loc/eval_results_cls_mlp_balanced/ffpp_auc_by_type.csv
```

本地文档：

```text
docs/ffpp_comparison_results.md
docs/experiment_summary.md
docs/innovation_design.md
docs/developer_progress_report.md
```
