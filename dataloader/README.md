# LF-Loc 数据集与 DataLoader 使用文档

## 1. 数据集组织结构

我们使用的数据集为 **FaceForensics++ (c23 压缩版本)**，已预处理为人脸裁剪图片（PNG 格式）。请将数据集解压到项目根目录下的 `data/FaceForensics++/`，其内部结构如下：

```
LF-Loc/
├── data/
│   └── FaceForensics++/
│       ├── original_sequences/
│       │   ├── actors/
│       │   │   └── c23/
│       │   │       └── frames/          # 真实人脸帧（演员场景）
│       │   │           └── <video_id>/
│       │   │               ├── 0000.png
│       │   │               └── ...
│       │   └── youtube/
│       │       └── c23/
│       │           └── frames/          # 真实人脸帧（YouTube 场景）
│       │               └── <video_id>/
│       │                   ├── 0000.png
│       │                   └── ...
│       └── manipulated_sequences/
│           ├── Deepfakes/
│           │   └── c23/
│           │       ├── frames/          # 伪造人脸帧（Deepfakes）
│           │       │   └── <video_id>/
│           │       │       ├── 0000.png
│           │       │       └── ...
│           │       └── masks/           # 可选：伪造区域标注
│           │           └── <video_id>/
│           │               ├── 0000.png
│           │               └── ...
│           ├── Face2Face/
│           ├── FaceSwap/
│           └── NeuralTextures/
```

**关键说明**：
- 所有图片均为 `.png` 格式，尺寸为 256×256（或原始尺寸，代码中会 resize 到 224×224）。
- 真实图片（`original_sequences`）**不包含 mask**；伪造图片（`manipulated_sequences`）部分包含 mask，位于同级的 `masks/` 目录下。
- DataLoader 会自动根据路径中的 `original_sequences` 或 `manipulated_sequences` 分配标签（0 真实 / 1 伪造）。
- 若 mask 文件不存在，DataLoader 会返回**全零 mask**，确保训练时定位损失对无标注样本不产生干扰。

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

# 创建训练数据集（自动随机划分为 train/val/test，比例可调）
train_dataset = FFPPDataset(
    root_dir='./data/FaceForensics++',      # 相对项目根目录的路径
    split='train',                           # 'train' / 'val' / 'test'
    transform=transform,
    mask_transform=mask_transform
)

# 创建 DataLoader
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4
)
```

### 2.2 参数说明

| 参数 | 类型 | 说明 |
|------|------|------|
| `root_dir` | `str` | 数据集根目录（推荐使用相对路径，如 `'./data/FaceForensics++'`） |
| `split` | `str` | 数据集划分：`'train'`, `'val'`, `'test'`。默认按照 **7:1.5:1.5** 随机划分（可修改 `split_ratio` 参数） |
| `transform` | `callable` | 应用于图像的变换（通常包含 resize、totensor、归一化） |
| `mask_transform` | `callable` | 应用于 mask 的变换（通常只需 resize 和 totensor，不需要归一化） |
| `split_ratio` | `tuple` | 可选，划分比例，默认为 `(0.7, 0.15, 0.15)` |

### 2.3 输出格式

DataLoader 每次迭代返回一个三元组 `(image, mask, label)`：

- **image**：`torch.Tensor`，形状 `(C, H, W)`，值域取决于 `transform`（若未归一化则为 [0,1]）
- **mask**：`torch.Tensor`，形状 `(1, H, W)`，值域 [0,1]。若原图无 mask，则返回全零张量
- **label**：`torch.LongTensor`，标量，`0` 表示真实，`1` 表示伪造

示例：
```python
for images, masks, labels in train_loader:
    print(images.shape)   # [batch, 3, 224, 224]
    print(masks.shape)    # [batch, 1, 224, 224]
    print(labels.shape)   # [batch]
    break
```

---

## 3. 测试脚本

项目提供了测试脚本 `dataloader/test_dataloader.py`，用于验证数据集是否放置正确、DataLoader 是否工作正常。

### 3.1 运行方式

在项目根目录 `D:\LF-loc` 下执行：

```bash
conda activate lfloc
python dataloader/test_dataloader.py
```

### 3.2 预期输出

```
数据集样本数: 359497

单样本: image torch.Size([3, 224, 224]), mask torch.Size([1, 224, 224]), label 1

Batch: images torch.Size([4, 3, 224, 224]), masks torch.Size([4, 1, 224, 224]), labels torch.Size([4])
✓ DataLoader 测试通过！
```

### 3.3 常见问题

| 现象 | 可能原因 | 解决方法 |
|------|----------|----------|
| `数据集目录不存在 - D:\data\FaceForensics++` | 相对路径错误 | 确认工作目录为项目根目录，且数据集在 `data/FaceForensics++` 下；修改 `data_root = './data/FaceForensics++'` |
| `ModuleNotFoundError: No module named 'dataloader'` | 未将项目根目录加入 Python 路径 | 在根目录运行脚本，或 `sys.path.append('.')` |
| `TypeError: expected Tensor as element 2 in argument 0, but got NoneType` | 部分样本的 mask 为 None | 已修复：现在所有样本均返回全零 mask 代替 None |
| 样本数为 0 | 目录下没有 .png 文件 | 检查 `data/FaceForensics++` 下是否存在 `original_sequences` 和 `manipulated_sequences` 文件夹，且内部包含图片 |

---

## 4. 团队协作建议

- **数据路径统一**：所有队友应将数据集解压到 `LF-Loc/data/FaceForensics++`，DataLoader 使用相对路径，无需修改代码。
- **环境同步**：使用 `requirements.txt` 创建一致的 conda 环境（`conda create -n lfloc python=3.8` + `pip install -r requirements.txt`）。
- **划分策略**：当前为随机划分，若需严格按视频 ID 划分（避免数据泄露），请使用官方提供的 `train/val/test` 视频列表。

---

**文档版本**：1.0  
**最后更新**：2026-06-14