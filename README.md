# LF-Loc：轻量级人脸伪造区域定位

**基于轻量分割网络的人脸伪造区域定位方法研究**

LF-Loc 是一个轻量、参数高效的**像素级人脸伪造区域定位**框架——不止做真假二分类，而是预测图像**被篡改的具体位置**。专为资源受限场景（单 GPU、小数据量）设计。

> 内容安全课程项目。架构灵感来自 [DeepFake-Adapter](https://github.com/rshaojimmy/DeepFake-Adapter)（冻结骨干 + 轻量适配模块的范式）和 [LAA-Net](https://github.com/10Ring/LAA-Net)（显式定位优于隐式注意力）。

---

## ✨ 项目亮点

- **像素级定位，而非仅分类**——预测伪造 Mask，并行一个分支输出真/假判断。
- **轻量化设计**——遵循"重用–冻结–轻训"：冻结大型预训练骨干，仅训练 **约 1.8M 参数**，远低于 ObjectFormer（>100M）等方法。
- **FBAA（伪造边界感知注意力）**——显式利用伪造融合边界信息，引导模型关注最关键的定位线索。
- **单卡友好**——单 GPU、小数据量即可快速训练。

---

## 🧠 方法概览

```
                 ┌─────────────────────┐
   输入图像   →  │   冻结骨干网络        │  (CLIP-ViT-B/16 或 DINOv2-ViT-B/14, requires_grad=False)
                 └──────────┬──────────┘
                            │ 多尺度特征
                 ┌──────────▼──────────┐
                 │    轻量 FPN Neck     │  (约 0.5M 参数)
                 └──────────┬──────────┘
                            │
              ┌─────────────┼─────────────┐
              ▼             ▼             ▼
        ┌──────────┐  ┌──────────┐  ┌──────────┐
        │  FBAA +  │  │  边界     │  │  全局     │
        │  分割头   │◄─│  检测分支 │  │  分类头   │
        └────┬─────┘  └──────────┘  └────┬─────┘
             │                            │
          伪造 Mask                    真 / 假
```

设计哲学是**"重用–冻结–轻训"**：最大化复用预训练模型，最小化可训练参数。FBAA 计算边界响应图，并将其作为注意力权重融入主分割特征。

### 损失函数

多任务损失：

```
L_total = λ1 · L_cls + λ2 · (L_bce + L_dice) + λ3 · L_boundary
```

---

## 📦 安装

```bash
# 克隆仓库
git clone https://github.com/<你的账号>/LF-Loc.git
cd LF-Loc

# 环境（推荐 Python 3.9+，CUDA 12.1，PyTorch 2.2.0）
conda create -n lfloc python=3.9 -y
conda activate lfloc
pip install -r requirements.txt
```

主要依赖：PyTorch、MMSegmentation、OpenCV、scikit-learn、dlib（人脸对齐）。

---

## 📂 数据准备

训练与验证使用 **FaceForensics++（FF++，c23 版本）**，跨域测试使用 **Celeb-DF** 和 **DFDC**。

1. 从[官方仓库](https://github.com/ondyari/FaceForensics)下载 FF++（c23）。
2. 运行人脸检测、对齐与裁剪（预处理流程参考 LAA-Net）：

```bash
python tools/preprocess.py --input /path/to/FF++ --output ./data/ffpp --version c23
```

期望的目录结构：

```
data/ffpp/
├── train/
│   ├── real/
│   └── fake/{DF,F2F,FS,NT}/
└── val/
    ├── real/
    └── fake/{DF,F2F,FS,NT}/
```

---

## 🚀 使用方法

### 训练

```bash
python train.py --config configs/lfloc_vitb.yaml --data ./data/ffpp
```

### 评估

```bash
python eval.py --config configs/lfloc_vitb.yaml --weights ./checkpoints/lfloc_best.pth
```

### 可视化预测结果

```bash
python tools/visualize.py --weights ./checkpoints/lfloc_best.pth --image samples/demo.png
```

---

## 📊 实验结果

> FF++（c23）上的结果。指标：AUC（分类）、IoU / Dice / F1@pixel（定位）。*（实验完成后更新）*

| 方法 | AUC ↑ | IoU ↑ | Dice ↑ | F1@pixel ↑ | 可训练参数 |
|------|-------|-------|--------|-----------|-----------|
| Face X-ray（论文值） | – | – | – | – | – |
| MVSS-Net（论文值） | – | – | – | – | – |
| **LF-Loc（本文）** | – | – | – | – | **约 1.8M** |

### 消融实验（FBAA 与 Dice Loss）

| FBAA | Dice Loss | AUC ↑ | IoU ↑ | Dice ↑ |
|:----:|:---------:|-------|-------|--------|
| ✗ | ✗ | – | – | – |
| ✓ | ✗ | – | – | – |
| ✓ | ✓ | – | – | – |

---

## 📁 项目结构

```
LF-Loc/
├── configs/           # 实验配置文件
├── data/              # 数据集（已 gitignore）
├── models/
│   ├── backbone.py    # 冻结的 ViT 骨干
│   ├── fpn.py         # 轻量 FPN Neck
│   ├── fbaa.py        # 伪造边界感知注意力
│   └── heads.py       # 分割头 + 分类头
├── tools/             # 预处理与可视化脚本
├── train.py
├── eval.py
└── requirements.txt
```

---

## 🙏 致谢

本工作基于以下研究的思路：
- **DeepFake-Adapter**（Shao 等）——冻结骨干 + 轻量双层适配器的范式。
- **LAA-Net**（Nguyen 等）——在多任务框架中对脆弱点/融合边界的显式注意力。

数据集：[FaceForensics++](https://github.com/ondyari/FaceForensics)、[Celeb-DF](https://github.com/yuezunli/celeb-deepfakeforensics)、[DFDC](https://ai.facebook.com/datasets/dfdc/)。

---

## 📄 许可证

仅供学术与教学使用。详见 `LICENSE`。
