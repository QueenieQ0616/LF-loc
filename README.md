# 📘 LF-Loc 开发文档（B 维护版）

## 1. 环境配置 (Setup)

本项目基于 **CPU** 训练（无 GPU 环境），请严格按照以下版本配置，避免依赖冲突。

### 1.1 创建 Conda 环境
```bash
conda create -n lfloc python=3.9 -y
conda activate lfloc
```

### 1.2 安装依赖
```bash
# 安装 PyTorch (CPU 版)
pip install torch==2.0.1+cpu torchvision==0.15.2+cpu --index-url https://download.pytorch.org/whl/cpu

# 安装核心库
pip install timm opencv-python numpy tqdm pyyaml scikit-learn

# 如果需要手动下载权重（可选）
pip install huggingface_hub
```

### 1.3 模型权重（重要）
由于网络限制，DINOv2 权重已手动下载并放入缓存。
**请勿删除以下目录：**
```
C:\Users\你的用户名\.cache\huggingface\hub\models--timm--vit_base_patch14_dinov2.lvd142m\
```

---

## 2. 项目结构 (Structure)

```
LF-Loc/
├── checkpoints/          # 训练权重保存目录（自动生成）
├── data/
│   └── dataset.py       # 数据集定义（FakeData，待 C 替换）
├── losses/
│   └── loss.py          # 损失函数（待 D 完善）
├── metrics/
│   └── metrics.py       # 评价指标（IoU, Dice）
├── models/
│   ├── backbone.py      # 冻结的 DINOv2 主干（B 维护）
│   ├── fpn.py           # FPN 特征金字塔（待 A 优化）
│   ├── fbaa.py          # 频域注意力（待 A 接入）
│   ├── head.py          # 分割头（待 A 优化）
│   └── lfloc.py         # 总模型组装
├── utils/
├── configs/
├── train.py             # 主训练脚本（B 维护）
└── README.md            # 本文档
```

---

## 3. 训练与推理 (Training)

### 3.1 启动训练
当前为 CPU 验证版本，直接运行：
```bash
python train.py
```

### 3.2 参数说明
训练参数位于 `train.py` 的 `parse_args()` 中，当前配置如下：

| 参数 | 值 | 说明 |
|----|----|----|
| `--device` | `cpu` | 强制使用 CPU |
| `--img_size` | `224` | **严禁修改**，改了会炸 |
| `--batch_size` | `2` | CPU 显存限制 |
| `--epochs` | `3` | 测试用，正式训练建议 50+ |
| `--num_workers` | `0` | Windows 必须为 0 |

### 3.3 预期输出
训练开始后，应看到如下日志：
```
Using device: cpu
============================================================
LF-Loc Model Summary
Backbone:      vit_base_patch14_dinov2
Image size:    224
...
============================================================
Epoch 1/3
Epoch 1 [Train]: ██████░░░░ 10%
Loss: 5.36
✅ Saved best model (IoU=...)
```

---

## 4. 接口规范 (API Contract)

**⚠️ 警告：这是对接红线，请勿随意更改，否则会导致下游报错。**

### 4.1 数据接口 (For C - 数据组)
`DataLoader` 输出的字典格式 **必须** 严格如下：
```python
batch = {
    "image": torch.Tensor,  # Shape: [B, 3, 224, 224], Range: [0, 1]
    "mask": torch.Tensor   # Shape: [B, 1, 224, 224], Value: {0, 1}
}
```

### 4.2 模型接口 (For A - 算法组)
- **Backbone 输出特征**：
  ```python
  feat: torch.Size([B, 768, 16, 16])
  ```
- **最终输出 Mask**：
  ```python
  pred_mask: torch.Size([B, 1, 224, 224])
  ```

---

## 5. 故障排查 (Troubleshooting)

### Q1: 报错 `RuntimeError: shape '[...]' is invalid`
**原因**：输入图片尺寸不是 224。
**解决**：检查 `train.py` 和 `dataset.py` 中的 `img_size` 是否均为 224。

### Q2: 报错 `Feature token number 257 is not a perfect square`
**原因**：DINOv2 输出包含 CLS Token。
**解决**：已修复于 `models/backbone.py`，确保使用了 `out = out[:, 1:, :]` 去除首 Token。

### Q3: 训练极慢
**原因**：CPU 训练且无 GPU 加速。
**解决**：正常现象。如需加速，需在带 GPU 的服务器上运行，并开启 `amp` (混合精度)。

---

## 6. 待办事项 (TODO)

- [ ] **C**：替换 `data/dataset.py` 中的 `FakeDataset` 为 FF++ 真实数据。
- [ ] **A**：接入并优化 `models/fbaa.py` (频域注意力模块)。
- [ ] **D**：优化 `losses/loss.py`，加入边界损失（Boundary Loss）。

---
*文档维护者：B (Backbone & Pipeline)*
*最后更新：2026-6-12*
