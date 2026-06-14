import sys
import os
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from ffpp_dataset import FFPPDataset   # 注意文件名是 ffpp_dataset.py

def main():
    # 数据集相对于项目根目录的路径（项目根目录即 D:\LF-loc）
    # 使用 './data/FaceForensics++' 表示当前工作目录下的 data 文件夹
    data_root = './data/FaceForensics++'
    abs_root = os.path.abspath(data_root)
    
    if not os.path.exists(abs_root):
        print(f"错误：数据集目录不存在 - {abs_root}")
        print("请确保数据集解压在 LF-loc/data/FaceForensics++/")
        return

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
    ])

    dataset = FFPPDataset(
        root_dir=abs_root,
        split='train',
        transform=transform,
        mask_transform=transform
    )
    
    print(f"数据集样本数: {len(dataset)}")
    if len(dataset) == 0:
        print("警告：未找到任何图片，请检查目录结构")
        return

    image, mask, label = dataset[0]
    print(f"\n单样本: image {image.shape}, mask {mask.shape if mask is not None else 'None'}, label {label}")

    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
    images, masks, labels = next(iter(loader))
    print(f"\nBatch: images {images.shape}, masks {masks.shape if masks is not None else 'None'}, labels {labels.shape}")
    print("✓ DataLoader 测试通过！")

if __name__ == '__main__':
    main()