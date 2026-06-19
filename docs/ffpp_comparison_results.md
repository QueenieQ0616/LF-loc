# FF++ 横向对比评估结果

## 1. 评估目的

本次评估用于把 LF-Loc 在 FF++ c23 测试集上的结果整理成可以和 Face X-ray、FakeLocator 论文进行横向参考的格式。

参考论文结果如下：

| Method | Dataset | Image AUC | IoU | Dice | PBCA |
| --- | --- | ---: | ---: | ---: | ---: |
| Face X-ray | FF++ | 0.9852 | - | - | - |
| FakeLocator | FF++ | 0.9846 | 0.3097 | 0.0811 | 0.8687 |
| LF-Loc（定位 checkpoint） | FF++ c23 | - | 0.8584 | 0.9222 | 0.8978 |
| LF-Loc（分类头 balanced 微调 checkpoint） | FF++ c23 | 0.6747 | 0.8584 | 0.9222 | 0.8978 |

说明：Image-level AUC 必须来自图像级分类头输出的连续 fake probability。原始定位 checkpoint 没有分类头，因此不能严格计算 Image-level AUC；后续补充了分类头，并采用 balanced sampler 只微调分类头，得到可计算 AUC 的 checkpoint。

## 2. 当前模型输出

补充分类头后，LF-Loc 的输出包括：

```python
{
    "mask_logits": ...,
    "boundary_logits": ...,
    "cls_logits": ...
}
```

其中：

- `mask_logits` 用于像素级伪造区域定位；
- `boundary_logits` 用于边界辅助监督；
- `cls_logits` 用于图像级 Real/Fake 分类，并计算 Image-level AUC。

## 3. 数据集与样本数量

评估数据来自 FF++ c23 test split，实际样本数量如下：

| 类型 | 数量 |
| --- | ---: |
| Real | 821 |
| Fake | 2176 |
| Total | 2997 |

Fake 样本按伪造方法划分如下：

| Manipulation Type | 数量 |
| --- | ---: |
| DF | 640 |
| F2F | 512 |
| FS | 512 |
| NT | 512 |

## 4. 指标定义

### 4.1 Image-level AUC

Image-level AUC 使用分类头输出的连续 fake probability 计算 ROC-AUC。Fake 为正类 1，Real 为负类 0。AUC 不使用二值化后的预测标签。

### 4.2 IoU

IoU 衡量预测伪造区域和真实伪造区域的重合程度：

```text
IoU = TP / (TP + FP + FN)
```

IoU 只在 GT mask 非空的 fake 样本上逐张计算后取平均，避免真实图片的全黑 mask 抬高定位结果。

### 4.3 Dice

Dice 衡量预测 mask 与 GT mask 的重叠程度：

```text
Dice = 2TP / (2TP + FP + FN)
```

在二值分割任务中，Dice 与 pixel-level F1 等价。

### 4.4 PBCA

PBCA 表示 Pixel-wise Binary Classification Accuracy：

```text
PBCA = (TP + TN) / (TP + TN + FP + FN)
```

PBCA 在全部有效像素上统计，因此会同时受到真实背景区域和伪造区域预测结果影响。

## 5. 阈值设置

本次评估固定使用两个阈值：

| Threshold | 用途 |
| ---: | --- |
| 0.1 | 用于和 FakeLocator 论文设置进行参考比较 |
| 0.5 | LF-Loc 默认二值化阈值 |

没有在测试集上搜索最优阈值。

## 6. 原定位 checkpoint 结果

该 checkpoint 路径为：

```text
checkpoints_ffpp_dinov2_improved_fbaa/best_model.pth
```

它没有图像级分类头，因此 Image AUC 留空。

| Method | Dataset | Threshold | Image AUC | IoU | Dice | PBCA |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| LF-Loc | FF++ c23 | 0.1 | - | 0.845594 | 0.914603 | 0.887663 |
| LF-Loc | FF++ c23 | 0.5 | - | 0.858376 | 0.922194 | 0.897788 |

## 7. 分类头 balanced 微调 checkpoint 结果

该 checkpoint 路径为：

```text
checkpoints_ffpp_cls_mlp_balanced/best_model.pth
```

训练方式是在原定位 checkpoint 基础上补充 backbone-level mean/max pooling MLP 分类头，冻结定位分支，只训练分类头，并使用 `balanced_cls_sampler` 缓解 real/fake 样本不均衡。该版本可以计算 Image-level AUC，同时保留原定位 checkpoint 的 IoU、Dice 和 PBCA。

| Method | Dataset | Threshold | Image AUC | IoU | Dice | PBCA |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| LF-Loc | FF++ c23 | 0.1 | 0.674727 | 0.845594 | 0.914603 | 0.887663 |
| LF-Loc | FF++ c23 | 0.5 | 0.674727 | 0.858376 | 0.922194 | 0.897788 |

### AUC 分组结果

| Method | DF AUC | F2F AUC | FS AUC | NT AUC | Overall AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| LF-Loc | 0.855523 | 0.642345 | 0.650700 | 0.505141 | 0.674727 |

## 8. 输出文件

原定位 checkpoint 的评估文件：

```text
/root/autodl-tmp/LF-Loc/eval_results/ffpp_comparison.csv
/root/autodl-tmp/LF-Loc/eval_results/per_image_results.csv
/root/autodl-tmp/LF-Loc/eval_results/ffpp_auc_by_type.csv
```

分类头 balanced 微调 checkpoint 的评估文件：

```text
/root/autodl-tmp/LF-Loc/eval_results_cls_mlp_balanced/ffpp_comparison.csv
/root/autodl-tmp/LF-Loc/eval_results_cls_mlp_balanced/per_image_results.csv
/root/autodl-tmp/LF-Loc/eval_results_cls_mlp_balanced/ffpp_auc_by_type.csv
```

## 9. 实际评估命令

原定位 checkpoint：

```bash
cd /root/autodl-tmp/LF-Loc

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

分类头 balanced 微调 checkpoint：

```bash
cd /root/autodl-tmp/LF-Loc

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

## 10. 结论

当前最稳的定位结果仍然来自原定位 checkpoint，并且 balanced 分类头版本保留了该定位能力：IoU 0.858376、Dice 0.922194、PBCA 0.897788。

补分类头后可以计算 Image-level AUC。当前 balanced MLP 分类头的 Overall AUC 为 0.674727，相比最初简单分类头有所提升，同时没有牺牲定位指标。不过该 AUC 仍明显低于 Face X-ray 和 FakeLocator 论文中的 0.98 级别结果，说明当前 LF-Loc 的主要优势仍然在伪造区域定位，而不是图像级真假分类。后续如果要继续提高 AUC，需要更强的分类分支或更系统的联合训练策略。
