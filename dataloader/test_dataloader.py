import sys
import os
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from ffpp_dataset import FFPPDataset

def main():
    # 数据集父目录（包含 train/ 和 test/ 子文件夹）
    data_root = '../data/FaceForensics++'
    abs_root = os.path.abspath(data_root)
    
    if not os.path.exists(abs_root):
        print(f"错误：数据集目录不存在 - {abs_root}")
        print("请确保数据集位于 LF-loc/data/FaceForensics++/ 下，且包含 train/ 和 test/ 子目录")
        return

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
    ])

    # 测试训练集
    print("=== 测试训练集 ===")
    train_dataset = FFPPDataset(
        root_dir=abs_root,
        split='train',
        transform=transform,
        mask_transform=transform
    )
    print(f"训练集样本数: {len(train_dataset)}")
    if len(train_dataset) > 0:
        image, mask, label = train_dataset[0]
        print(f"单样本: image {image.shape}, mask {mask.shape}, label {label}")
        loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
        images, masks, labels = next(iter(loader))
        print(f"Batch: images {images.shape}, masks {masks.shape}, labels {labels.shape}")

    # 测试测试集
    print("\n=== 测试测试集 ===")
    test_dataset = FFPPDataset(
        root_dir=abs_root,
        split='test',
        transform=transform,
        mask_transform=transform
    )
    print(f"测试集样本数: {len(test_dataset)}")
    if len(test_dataset) > 0:
        image, mask, label = test_dataset[0]
        print(f"单样本: image {image.shape}, mask {mask.shape}, label {label}")
        loader = DataLoader(test_dataset, batch_size=4, shuffle=True, num_workers=0)
        images, masks, labels = next(iter(loader))
        print(f"Batch: images {images.shape}, masks {masks.shape}, labels {labels.shape}")

    print("\n✓ DataLoader 测试通过！")

if __name__ == '__main__':
    main()