# LF-Loc 数据集与 DataLoader 使用文档

## 1. 数据集组织结构

我们使用的数据集为 **FaceForensics++ (c23 压缩版本)**，已预处理为人脸裁剪图片（PNG 格式），并按视频文件夹级别划分为 **训练集 (train/)** 和 **测试集 (test/)**。请将数据集解压到项目根目录下的 `data/FaceForensics++/`，其内部结构如下：

```
LF-Loc/
├── data/
│   └── FaceForensics++/                    # 数据集父目录
│       ├── train/                          # 训练集
│       │   ├── original_sequences/
│       │   │   ├── youtube/
│       │   │   │   └── c23/
│       │   │   │       └── frames/         # 真实人脸帧（YouTube）
│       │   │   │           └── <video_id>/
│       │   │   │               ├── 0000.png
│       │   │   │               └── ...
│       │   │   └── actors/
│       │   │       └── c23/
│       │   │           └── frames/         # 真实人脸帧（演员）
│       │   └── manipulated_sequences/
│       │       ├── Deepfakes/
│       │       │   └── c23/
│       │       │       └── frames/         # 伪造人脸帧
│       │       ├── Face2Face/
│       │       ├── FaceSwap/
│       │       └── ...                     # 其他伪造方法
│       └── test/                           # 测试集
│           ├── original_sequences/
│           └── manipulated_sequences/
├── dataloader/
│   ├── __init__.py
│   ├── ffpp_dataset.py
│   └── test_dataloader.py
└── ...
```

**关键说明**：
- 所有图片均为 `.png` 格式，代码中会统一 resize 到 224×224。
- 真实图片（`original_sequences`）没有 mask；伪造图片（`manipulated_sequences`）部分包含 mask（位于 `masks/` 同级目录，本数据集已移除 mask 以节省空间）。
- DataLoader 会自动根据路径中的 `original_sequences` 或 `manipulated_sequences` 分配标签（0 真实 / 1 伪造）。
- 由于没有 mask 文件，DataLoader 会返回**全零 mask**，确保训练时定位损失对无标注样本不产生干扰。

---

## 2. DataLoader 使用方法

DataLoader 类 `FFPPDataset` 位于 `dataloader/ffpp_dataset.py`。使用前请确保已激活 conda 环境 `lfloc`。

### 2.1 导入与实例化

```python
from torch.utils.data import DataLoader
import torchvision.transforms as T
from dataloader.ffpp_dataset import FFPPDataset

# 定义图像和 mask 的变换
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

mask_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor()
])

# 创建训练数据集
train_dataset = FFPPDataset(
    root_dir='./data/FaceForensics++',   # 父目录
    split='train',                        # 'train' 或 'test'
    transform=transform,
    mask_transform=mask_transform
)

# 创建 DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
```

### 2.2 参数说明

| 参数 | 类型 | 说明 |
|------|------|------|
| `root_dir` | `str` | 包含 `train/` 和 `test/` 的父目录（推荐相对路径 `'./data/FaceForensics++'`） |
| `split` | `str` | `'train'` 或 `'test'`，决定加载哪个子集 |
| `transform` | `callable` | 应用于图像的变换 |
| `mask_transform` | `callable` | 应用于 mask 的变换（通常只需 resize 和 totensor） |
| `max_videos_per_class` | `int` | 可选，每个类别最多加载的视频文件夹数（用于快速测试） |

### 2.3 输出格式

DataLoader 每次迭代返回一个三元组 `(image, mask, label)`：

- **image**：`torch.Tensor`，形状 `(C, H, W)`，值域取决于 `transform`
- **mask**：`torch.Tensor`，形状 `(1, H, W)`，值域 [0,1]（全零）
- **label**：`torch.LongTensor`，标量，`0` 真实，`1` 伪造

---

## 3. 测试脚本

项目提供了测试脚本 `dataloader/test_dataloader.py`，用于验证数据集划分是否正确、DataLoader 是否工作正常。

### 3.1 运行方式

在项目根目录 `D:\LF-loc` 下执行：

```bash
conda activate lfloc
python dataloader/test_dataloader.py
```

### 3.2 预期输出（示例）

```
=== 测试训练集 ===
训练集样本数: 16098
单样本: image torch.Size([3, 224, 224]), mask torch.Size([1, 224, 224]), label 0
Batch: images torch.Size([4, 3, 224, 224]), masks torch.Size([4, 1, 224, 224]), labels torch.Size([4])

=== 测试测试集 ===
测试集样本数: 3983
单样本: image torch.Size([3, 224, 224]), mask torch.Size([1, 224, 224]), label 0
Batch: images torch.Size([4, 3, 224, 224]), masks torch.Size([4, 1, 224, 224]), labels torch.Size([4])

✓ DataLoader 测试通过！
```

### 3.3 常见问题

| 现象 | 可能原因 | 解决方法 |
|------|----------|----------|
| `Split directory not found` | `root_dir` 下没有 `train/` 或 `test/` 子目录 | 确认数据集已按划分放入 `data/FaceForensics++/` |
| 样本数为 0 | 目录结构不正确或没有 `.png` 文件 | 检查 `train/original_sequences/.../frames/` 下是否有视频文件夹和图片 |
| `ModuleNotFoundError: No module named 'dataloader'` | 未在项目根目录运行 | 先 `cd D:\LF-loc` 再运行 |

---

## 4. 团队协作建议

- **数据路径统一**：所有队友将划分后的数据集解压到 `LF-Loc/data/FaceForensics++/`，DataLoader 使用相对路径，无需修改代码。
- **环境同步**：使用 `requirements.txt` 创建一致的 conda 环境。
- **自定义划分**：如需调整训练/测试比例，请在数据准备阶段重新划分文件夹，本 DataLoader 已适配。

---

**文档版本**：2.0  
**最后更新**：2026-06-16
```