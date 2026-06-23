# LF-Loc

LF-Loc 是一个面向人脸伪造区域定位的轻量化工程项目。它的目标不是只判断一张图像是真还是假，而是输出像素级伪造区域 mask，回答“哪里被伪造了”。

当前最终主线为：

```text
Frozen DINOv2 ViT Backbone
  -> Lightweight FPN
  -> FBAA / Boundary-Region Collaborative Attention
  -> Segmentation Head
  -> Forgery Mask
```

项目已经完成 FaceForensics++ 数据接入、训练脚本、损失函数、指标统计、FF++ 横向评估脚本、消融实验和结果文档。

## 当前结论

最终推荐使用的定位 checkpoint：

```text
outputs/checkpoints/loc_improved_bce2_e30/best_model.pth
```

FF++ c23 test 上的主结果：

| Threshold | IoU | Dice | PBCA |
| ---: | ---: | ---: | ---: |
| 0.1 | 0.846686 | 0.915088 | 0.887694 |
| 0.5 | 0.858838 | 0.922383 | 0.897616 |

早期完整评估还记录了 Boundary F1：

| Method | Dataset | Samples | IoU | Dice | Boundary F1 | Trainable Params |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| LF-Loc | FF++ test | 2997 | 0.857402 | 0.921690 | 0.569805 | 1.95M |

Image-level AUC 是补充指标，不是当前主优势。额外 AUC 微调 checkpoint 可达到：

```text
Overall AUC = 0.799709
```

但该 checkpoint 会明显牺牲定位性能，因此最终主模型仍以 IoU、Dice、PBCA 为主。

## 主要创新点

### 1. 冻结预训练主干的轻量定位框架

项目使用 DINOv2 ViT 作为视觉 backbone，并默认冻结主干，只训练轻量 FPN、FBAA 和预测头。这样可以利用大规模预训练特征，同时控制训练成本和可训练参数量。

### 2. FBAA 边界-区域协同注意力

FBAA 是当前最终方案的核心模块，全称可写作：

```text
Forgery Boundary-Aware Attention
```

当前实现包含 boundary branch、region branch 和 gate fusion。它让模型同时关注伪造区域内部和伪造区域边界，从而增强对融合痕迹、纹理断裂和边缘异常的定位能力。

### 3. 显式边界监督

项目从 GT mask 自动生成边界 target，并监督 FBAA 输出的 `boundary_logits`。这样模型不只学习伪造区域本身，还学习伪造区域边界。

### 4. Boundary-weighted BCE

训练中支持对边界像素加权：

```bash
--boundary_bce_weight 2.0
```

当前实验中 `boundary_bce_weight=2.0` 是最优定位配置。更大的权重如 4.0 反而会降低效果。

### 5. 完整 FF++ 评估脚本

`eval_ffpp_comparison.py` 支持输出：

- Image-level AUC；
- fake-only mean IoU；
- fake-only mean Dice；
- PBCA；
- DF/F2F/FS/NT 分组 AUC；
- per-image CSV；
- comparison CSV。

## 工程模块

| 文件 | 作用 |
| --- | --- |
| `data/dataset.py` | FF++ 数据读取、mask 加载、real/fake 标签、balanced sampler |
| `models/backbone.py` | DINOv2/CLIP ViT backbone，默认冻结 |
| `models/fpn.py` | 轻量 FPN，将 backbone 特征转换为分割特征 |
| `models/fbaa.py` | 边界-区域协同注意力模块 |
| `models/head.py` | segmentation head |
| `models/lfloc.py` | 总模型组装，输出 mask/boundary/classification logits |
| `losses/loss.py` | BCE、Dice、Boundary loss、classification loss |
| `metrics/metrics.py` | 训练过程中的 IoU、Dice、PixelAcc、AUC 统计 |
| `train.py` | 训练入口 |
| `eval_ffpp_comparison.py` | FF++ 横向对比评估 |

## 数据集

当前使用 FaceForensics++ c23。

| Split | Total | Fake | Real |
| --- | ---: | ---: | ---: |
| train | 12044 | 8700 | 3344 |
| test | 2997 | 2176 | 821 |

测试集 fake 类型分布：

| Type | Method | Samples |
| --- | --- | ---: |
| DF | Deepfakes | 640 |
| F2F | Face2Face | 512 |
| FS | FaceSwap | 512 |
| NT | NeuralTextures | 512 |

注意：当前代码中 `split="val"` 会映射到 FF++ 的 `test` 目录。课程实验可以接受，但严格论文实验应重新划分 validation/test，避免使用 test 选择 checkpoint。

## 训练

### 定位主模型训练

```bash
cd /root/autodl-tmp/LF-Loc

HF_ENDPOINT=https://hf-mirror.com /root/miniconda3/bin/python train.py \
  --dataset ffpp \
  --data_root data/FaceForensics++ \
  --device cuda \
  --epochs 30 \
  --batch_size 8 \
  --train_size -1 \
  --val_size -1 \
  --num_workers 2 \
  --lambda_cls 0 \
  --boundary_bce_weight 2.0 \
  --save_dir outputs/checkpoints/loc_improved_bce2_e30 \
  --log_dir outputs/logs/loc_improved_bce2_e30
```

### AUC 补充微调

```bash
cd /root/autodl-tmp/LF-Loc

HF_ENDPOINT=https://hf-mirror.com /root/miniconda3/bin/python train.py \
  --dataset ffpp \
  --data_root data/FaceForensics++ \
  --device cuda \
  --epochs 10 \
  --batch_size 4 \
  --train_size -1 \
  --val_size -1 \
  --num_workers 2 \
  --include_originals \
  --resume outputs/checkpoints/loc_improved_bce2_e30/best_model.pth \
  --train_cls_only \
  --balanced_cls_sampler \
  --cls_unfreeze_blocks 2 \
  --lambda_cls 1.0 \
  --lr 1e-5 \
  --save_dir outputs/checkpoints/auc_unfreeze2_lr1e5_e10 \
  --log_dir outputs/logs/auc_unfreeze2_lr1e5_e10
```

## 评估

```bash
cd /root/autodl-tmp/LF-Loc

HF_ENDPOINT=https://hf-mirror.com /root/miniconda3/bin/python eval_ffpp_comparison.py \
  --config configs/default.yaml \
  --checkpoint outputs/checkpoints/loc_improved_bce2_e30/best_model.pth \
  --data-root data/FaceForensics++ \
  --split test \
  --output-dir outputs/eval_results/loc_improved_bce2_e30 \
  --include-originals \
  --batch-size 8 \
  --num-workers 2 \
  --device cuda
```

输出文件：

```text
outputs/eval_results/loc_improved_bce2_e30/ffpp_comparison.csv
outputs/eval_results/loc_improved_bce2_e30/per_image_results.csv
outputs/eval_results/loc_improved_bce2_e30/ffpp_auc_by_type.csv
```

## 消融实验

| 实验 | 说明 | 结果 |
| --- | --- | ---: |
| no-pretrain baseline | 不加载 backbone 预训练权重 | Best IoU 0.8282 |
| DINOv2 baseline | DINOv2 + FPN + head，无 FBAA | Best IoU 0.8508 |
| 原始 FBAA | 加入早期 FBAA | Best IoU 0.8548 |
| improved FBAA + boundary BCE 2.0 | 当前最终定位方案 | Eval IoU 0.858838 |
| boundary BCE 4.0 | 更强边界权重 | Best IoU 0.8552 |
| balanced MLP 分类头 | 冻结定位分支，只训练分类头 | AUC 0.674727 |
| 解冻最后 2 个 block | AUC 微调 | AUC 0.799709 |
| HFAB 高频分支 | 高频伪影分支探索 | IoU 0.854300，未纳入最终方案 |

## 横向参考对比

| Method | Dataset | Image AUC | IoU | Dice | PBCA |
| --- | --- | ---: | ---: | ---: | ---: |
| MSCCNet | FF++ c23 | 0.9894 | 0.8329 | - | - |
| FakeLocator | FF++ | 0.9846 | 0.3097 | 0.0811 | 0.8687 |
| LF-Loc 定位 checkpoint | FF++ c23 | - | 0.8588 | 0.9224 | 0.8976 |
| LF-Loc AUC checkpoint | FF++ c23 | 0.7997 | 0.3912 | 0.4793 | 0.8324 |

该表应作为参考对比，不应声称严格复现或超越 SOTA。不同方法的数据处理、压缩等级、划分方式和评估协议可能不同。

## 当前限制

- `val` 和 `test` 当前没有严格拆分；
- Image-level AUC 不是当前系统优势；
- 定位最佳 checkpoint 和 AUC 最佳 checkpoint 不是同一个；
- HFAB 高频分支实验没有提升，因此不作为最终方案；
- 还缺少系统性的 mask 可视化、成功案例和失败案例分析；
- 当前主要结果集中在 FF++ c23，跨压缩等级和跨数据集泛化仍需补充。

## 文档

更完整的最终总结见：

```text
docs/final_project_summary.md
```

