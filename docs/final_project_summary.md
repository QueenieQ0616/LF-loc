# LF-Loc 最终项目汇总文档

## 1. 项目定位

LF-Loc 是一个面向人脸伪造区域定位的轻量化工程项目。它解决的问题不是简单判断一张图是真还是假，而是给定一张人脸图像，输出像素级伪造区域 mask，回答“哪里是被伪造的”。

因此，本项目的主任务是 **Face Forgery Localization**，核心评价指标应当以定位和分割指标为主，包括 IoU、Dice、PBCA 和 Boundary F1。Image-level AUC 可以作为补充指标，用于和 Face X-ray、FakeLocator 等图像级检测方法做参考对比，但它不是本项目最主要的优势指标。

当前最终方案不纳入 HFAB 高频分支。该分支已经做过实验，但测试集 IoU 为 0.854300，低于当前最佳定位模型 0.858838，因此只作为探索性消融记录，不作为最终创新点。

## 2. 工程现状

项目已经从最初的轻量分割 baseline 推进到一个完整可运行的 FF++ 伪造定位系统，当前已经具备以下能力：

- 支持 FaceForensics++ 数据集读取；
- 支持真实图片和伪造图片的 frame/mask 加载；
- 支持 DINOv2 预训练 backbone；
- 支持轻量 FPN 特征适配；
- 支持 FBAA 边界-区域协同注意力模块；
- 支持 mask、boundary、classification 多输出；
- 支持 BCE、Dice、Boundary loss、classification loss；
- 支持 boundary-weighted BCE；
- 支持训练、resume、分类头微调和 balanced sampler；
- 支持 FF++ 横向对比评估脚本；
- 输出 per-image 结果和汇总 CSV；
- 完成 baseline、FBAA、boundary BCE、AUC 微调等多组消融。

当前主要工程入口如下：

| 文件 | 作用 |
| --- | --- |
| `data/dataset.py` | FF++ 数据读取、mask 加载、real/fake 标签、balanced sampler |
| `models/backbone.py` | 加载 DINOv2/CLIP ViT backbone，默认冻结主干 |
| `models/fpn.py` | 轻量 FPN，将 backbone 特征转换为分割特征 |
| `models/fbaa.py` | 边界-区域协同注意力模块，是最终主要创新模块 |
| `models/head.py` | segmentation head，输出伪造区域 mask logits |
| `models/lfloc.py` | 总模型组装，输出 mask、boundary、classification logits |
| `losses/loss.py` | BCE、Dice、Boundary loss、classification loss |
| `metrics/metrics.py` | 训练过程中的 IoU、Dice、PixelAcc、AUC 统计 |
| `train.py` | 训练入口，支持定位训练和分类头微调 |
| `eval_ffpp_comparison.py` | FF++ 横向对比评估，输出 AUC、IoU、Dice、PBCA |

## 3. 最终模型结构

最终推荐使用的 LF-Loc 定位模型结构如下：

```text
Input Image
  -> Frozen DINOv2 ViT Backbone
  -> Lightweight FPN
  -> FBAA: Boundary-Region Collaborative Attention
  -> Segmentation Head
  -> Forgery Mask
```

当需要计算 Image-level AUC 时，模型还包含一个图像级分类头：

```text
Backbone Feature
  -> Global Average Pooling + Global Max Pooling
  -> MLP Classification Head
  -> cls_logits
```

`LFLOC.forward(..., return_dict=True)` 的输出形式为：

```python
{
    "mask_logits": ...,      # 像素级伪造区域预测
    "boundary_logits": ...,  # 伪造边界辅助预测
    "cls_logits": ...,       # 图像级真伪分类预测
}
```

其中最终定位结果主要依赖 `mask_logits`，边界监督和 FBAA 模块使用 `boundary_logits` 辅助训练，`cls_logits` 只用于补充计算 Image-level AUC。

## 4. 当前创新点

### 4.1 冻结预训练视觉主干的轻量伪造定位框架

项目使用 DINOv2 ViT 作为视觉 backbone，并默认冻结 backbone，只训练轻量化的 FPN、FBAA 和预测头。这样做的意义是：

- 利用 DINOv2 的强语义表征能力；
- 降低训练成本；
- 控制可训练参数量；
- 更适合课程实验和资源受限 GPU 环境；
- 将工程重点放在“伪造区域定位适配模块”上，而不是从零训练大模型。

当前最佳定位模型的可训练参数约为 1.95M，整体参数主要来自冻结 backbone。这个设置支撑了项目的轻量化叙事。

### 4.2 FBAA：边界-区域协同注意力模块

FBAA 是当前最终方案里最核心的结构创新。它的全称可以写作：

```text
Forgery Boundary-Aware Attention
```

当前实现进一步扩展为边界-区域协同注意力，即不仅关注伪造区域内部，还显式关注伪造区域边缘。

在 `models/fbaa.py` 中，FBAA 包含：

| 分支 | 作用 |
| --- | --- |
| shared projection | 将输入特征压缩到轻量隐藏通道 |
| boundary branch | 预测伪造边界响应 `boundary_logits` |
| region branch | 预测伪造区域响应 |
| feature projection | 对输入特征进行局部增强 |
| gate fusion | 自适应融合原始特征和增强特征 |

核心思想是：伪造区域的明显线索往往集中在融合边界附近，例如纹理断裂、颜色不连续、局部边缘异常等。FBAA 通过边界分支学习这些边界线索，再用 gate 将边界和区域响应融合进主分割特征中，从而提升定位能力。

### 4.3 显式边界监督

项目不仅让模型输出 mask，还让 FBAA 输出 `boundary_logits`，并通过 GT mask 自动生成边界监督。

在 `losses/loss.py` 中，`BoundaryTarget` 使用 Sobel 算子从 GT mask 中生成边界图：

```text
GT mask -> Sobel gradient -> normalized boundary target
```

然后 `BoundaryLoss` 对 `boundary_logits` 和 GT boundary 进行监督。这样模型不仅学习“哪里是伪造区域”，还学习“伪造区域边界在哪里”。

这个设计比单纯加一个 Dice/BCE 分割损失更贴近人脸伪造定位任务，因为伪造边界通常是定位任务的重要线索。

### 4.4 Boundary-weighted BCE

最终最佳结果来自 `boundary_bce_weight=2.0`。它的作用是在 BCE 分割损失中提高边界区域像素的权重：

```text
BCE weight = 1 + boundary_bce_weight * boundary_target
```

也就是说，边界像素在训练时会被模型更重视。实验表明，适度增强边界区域监督可以带来小幅提升；但权重过大，例如 `boundary_bce_weight=4.0`，会导致效果下降，说明边界约束需要适中。

### 4.5 完整 FF++ 评估与横向对比脚本

项目补充了 `eval_ffpp_comparison.py`，用于输出和 Face X-ray、FakeLocator 论文可横向参考的指标：

- Image-level AUC；
- fake-only mean IoU；
- fake-only mean Dice；
- PBCA；
- DF/F2F/FS/NT 分组 AUC；
- per-image CSV；
- summary CSV。

评估脚本中 IoU 和 Dice 只在 GT mask 非空的 fake 样本上计算，避免真实图片的全黑 mask 抬高定位指标。PBCA 在全部有效像素上统计。该脚本让项目从“能训练”推进到了“能规范评估和产出结果表”。

## 5. 数据集设置

当前使用 FaceForensics++ c23。实际评估样本如下：

| Split | Total | Fake | Real |
| --- | ---: | ---: | ---: |
| train | 12044 | 8700 | 3344 |
| test | 2997 | 2176 | 821 |

测试集中 fake 样本分布如下：

| 类型 | 方法 | 数量 |
| --- | --- | ---: |
| DF | Deepfakes | 640 |
| F2F | Face2Face | 512 |
| FS | FaceSwap | 512 |
| NT | NeuralTextures | 512 |

需要注意：当前代码中 `split="val"` 会映射到 FF++ 的 `test` 目录，因此当前实验更适合作为课程项目实验结果。若要写成严格论文实验，应重新划分 validation/test，避免使用测试集选择 checkpoint。

## 6. 训练与评估指标

### 6.1 IoU

IoU 衡量预测伪造区域和 GT 伪造区域的重合程度：

```text
IoU = TP / (TP + FP + FN)
```

它是本项目最重要的主指标之一。IoU 越高，说明模型预测的伪造区域和真实伪造区域越接近。

### 6.2 Dice

Dice 衡量预测 mask 和 GT mask 的重叠程度：

```text
Dice = 2TP / (2TP + FP + FN)
```

Dice 与 pixel-level F1 等价，通常比 IoU 数值更高，但趋势相似。

### 6.3 PBCA

PBCA 是 Pixel-wise Binary Classification Accuracy：

```text
PBCA = (TP + TN) / (TP + TN + FP + FN)
```

它表示像素级二分类准确率。因为背景像素通常较多，PBCA 容易受到背景预测影响，所以它适合作为辅助指标，不能单独代表定位质量。

### 6.4 Boundary F1

Boundary F1 衡量预测边界和 GT 边界是否匹配，用于评价模型对伪造区域边缘的定位能力。当前项目中该指标用于补充说明 FBAA 的边界建模效果。

### 6.5 Image-level AUC

Image-level AUC 使用分类头输出的 fake probability 计算 ROC-AUC。它回答的是“这张图是不是 fake”，而不是“哪里 fake”。因此它可以用于横向参考，但不应替代 IoU/Dice 成为本项目主指标。

## 7. 最终主要结果

### 7.1 最佳定位 checkpoint

最终推荐作为主结果的 checkpoint：

```text
outputs/checkpoints/loc_improved_bce2_e30/best_model.pth
```

测试集评估结果：

| Threshold | Image AUC | IoU | Dice | PBCA |
| ---: | ---: | ---: | ---: | ---: |
| 0.1 | 0.546159* | 0.846686 | 0.915088 | 0.887694 |
| 0.5 | 0.546159* | 0.858838 | 0.922383 | 0.897616 |

`*` 该 checkpoint 没有专门训练分类头，因此 AUC 不作为有效结论。最终主结果应看 IoU、Dice 和 PBCA。

### 7.2 Boundary F1 结果

早期完整评估脚本得到的 improved FBAA 结果如下：

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

从分组看，模型在 F2F 和 FS 上表现最好，在 DF 和 NT 上相对弱一些。

### 7.3 AUC 补充结果

为了补充 Image-level AUC，项目额外进行了分类头微调。当前 AUC 最好的 checkpoint 为：

```text
outputs/checkpoints/auc_unfreeze2_lr1e5_e10/best_model.pth
```

训练方式：

```text
从定位 checkpoint 出发；
使用 balanced sampler；
训练分类目标；
解冻 DINOv2 最后 2 个 transformer block；
学习率 1e-5。
```

评估结果：

| Threshold | Image AUC | IoU | Dice | PBCA |
| ---: | ---: | ---: | ---: | ---: |
| 0.1 | 0.799709 | 0.476469 | 0.573708 | 0.835724 |
| 0.5 | 0.799709 | 0.391209 | 0.479275 | 0.832440 |

分组 AUC：

| DF AUC | F2F AUC | FS AUC | NT AUC | Overall AUC |
| ---: | ---: | ---: | ---: | ---: |
| 0.932618 | 0.826752 | 0.846590 | 0.559648 | 0.799709 |

该 checkpoint 的 AUC 明显高于早期 balanced MLP 分类头的 0.674727，但会严重破坏定位性能。因此当前结论是：AUC 和定位指标暂时不能由同一个 checkpoint 同时达到最佳。

## 8. 消融实验总结

| 实验 | 说明 | 结果 |
| --- | --- | ---: |
| no-pretrain baseline | 不加载 backbone 预训练权重 | Best IoU 0.8282 |
| no-pretrain full | 不加载预训练权重的完整模型 | Best IoU 0.8255 |
| DINOv2 baseline | DINOv2 + FPN + head，无 FBAA | Best IoU 0.8508 |
| 原始 FBAA | 加入早期 FBAA | Best IoU 0.8548 |
| improved FBAA + boundary BCE 2.0 | 当前最终定位方案 | Eval IoU 0.858838 |
| boundary BCE 4.0 | 更强边界权重 | Best IoU 0.8552 |
| 低学习率继续微调 | 从定位 best 继续训练 | 未继续提升 |
| balanced MLP 分类头 | 冻结定位分支，只训分类头 | AUC 0.674727 |
| 解冻最后 1 个 block | AUC 微调 | Best AUC 0.7952 |
| 解冻最后 2 个 block | AUC 微调 | Best AUC 0.799709 |
| HFAB 高频分支 | 高频伪影分支探索 | IoU 0.854300，不纳入最终方案 |

消融结论：

1. DINOv2 预训练特征对定位任务非常重要，不加载预训练权重时 IoU 明显下降。
2. FBAA 相比 baseline 有稳定小幅提升，说明边界和区域注意力对伪造定位有帮助。
3. boundary-weighted BCE 的最佳权重目前是 2.0，权重过大反而会降低效果。
4. HFAB 高频分支没有带来超过最终主模型的收益，因此不作为最终创新点。
5. AUC 可以通过解冻 backbone 后几层提高，但会牺牲定位能力。

## 9. 横向对比

当前可以整理成如下横向参考表：

| Method | Dataset | Image AUC | IoU | Dice | PBCA |
| --- | --- | ---: | ---: | ---: | ---: |
| Face X-ray | FF++ | 0.9852 | - | - | - |
| FakeLocator | FF++ | 0.9846 | 0.3097 | 0.0811 | 0.8687 |
| LF-Loc 定位 checkpoint | FF++ c23 | - | 0.8588 | 0.9224 | 0.8976 |
| LF-Loc AUC checkpoint | FF++ c23 | 0.7997 | 0.3912 | 0.4793 | 0.8324 |

这张表需要谨慎解释：

- LF-Loc 的优势在像素级伪造区域定位，尤其是 IoU 和 Dice；
- Face X-ray 和 FakeLocator 的 AUC 更强，说明它们在图像级真伪检测上更成熟；
- LF-Loc 的 AUC checkpoint 虽然能达到 0.7997，但仍低于 0.98 级别的图像级检测方法；
- 由于数据处理、压缩等级、划分方式和评估协议可能不同，横向表应写作“参考对比”，不应声称严格复现或超越 SOTA；
- 报告重点应放在 FF++ c23 上的定位质量、边界建模和轻量训练成本。

## 10. 系统效果评价

从定位角度看，系统效果是比较好的。最终定位 checkpoint 在 FF++ c23 test 上达到：

```text
IoU  = 0.858838
Dice = 0.922383
PBCA = 0.897616
```

这说明模型预测的伪造区域和 GT mask 有较高重叠，区域定位效果稳定。Dice 超过 0.92，说明预测区域整体轮廓和真实 mask 的一致性较好。PBCA 接近 0.90，说明像素级真伪分类整体准确率也较高。

从工程角度看，项目是扎实的。它不仅有模型结构，还完成了数据接入、训练脚本、loss、metrics、评估脚本、消融实验和 CSV 输出。对于课程项目来说，已经具备完整实验闭环。

从创新角度看，项目的创新点不在于发明一个全新的大模型，而是在轻量定位框架中加入了面向伪造边界的结构设计和监督策略：

- 冻结预训练视觉主干，强调参数高效；
- FBAA 建模边界和区域协同注意力；
- 显式边界监督让模型学习伪造边缘；
- boundary-weighted BCE 强化边界像素训练；
- 评估脚本将定位指标和横向对比指标规范化输出。

从分类角度看，系统还有明显不足。Image-level AUC 最好达到 0.799709，低于 Face X-ray 和 FakeLocator 的 0.98 级别。这说明 LF-Loc 当前更适合作为“伪造区域定位模型”，而不是强图像级 deepfake detector。

## 11. 当前限制

当前项目仍有以下限制，需要在报告中客观说明：

1. validation/test 目前没有严格拆分，`val` 实际映射到 FF++ test 目录，正式论文实验需要重新划分。
2. Image-level AUC 不是项目强项，当前最好 AUC 仍低于主流图像级检测方法。
3. 定位最佳 checkpoint 和 AUC 最佳 checkpoint 不是同一个，二者存在性能冲突。
4. HFAB 高频分支实验没有带来提升，因此不纳入最终方案。
5. 目前缺少系统性的可视化案例分析，例如成功案例、失败案例、边界响应图等。
6. 当前只主要评估 FF++ c23，跨压缩等级和跨数据集泛化还没有充分验证。

## 12. 最终可写入报告的总结

可以将本项目总结为：

> LF-Loc 是一个基于冻结 DINOv2 视觉主干的轻量化人脸伪造区域定位框架。项目通过轻量 FPN 将预训练语义特征转换为分割特征，并设计 FBAA 边界-区域协同注意力模块，显式建模伪造区域边界和区域内部响应。同时，利用 Sobel 生成的边界监督和 boundary-weighted BCE 强化模型对伪造边缘的学习。在 FaceForensics++ c23 测试集上，最终定位模型取得 IoU 0.8588、Dice 0.9224、PBCA 0.8976，说明该系统能够较准确地定位人脸伪造区域。项目还补充了分类头和横向评估脚本，可计算 Image-level AUC，但当前优势主要体现在像素级定位而非图像级真伪分类。

## 13. 推荐最终结论

最终报告建议采用以下口径：

1. 主任务：人脸伪造区域定位，不是单纯图像级检测。
2. 主模型：DINOv2 + LightFPN + improved FBAA + boundary-weighted BCE。
3. 主指标：IoU、Dice、PBCA、Boundary F1。
4. 主结果：IoU 0.858838，Dice 0.922383，PBCA 0.897616。
5. 主创新：边界-区域协同注意力、显式边界监督、边界加权损失、轻量参数高效训练。
6. AUC：作为补充指标，最好 0.799709，但不是项目优势。
7. HFAB：作为探索性实验，不纳入最终方案。

