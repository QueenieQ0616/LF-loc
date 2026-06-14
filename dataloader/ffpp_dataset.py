# D:\LF-loc\dataloader\ffpp_dataset.py
import os
import glob
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class FFPPDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None, mask_transform=None, split_ratio=(0.7, 0.15, 0.15)):
        """
        Args:
            root_dir: 数据集根目录，可以是相对路径（如 './data/FACEFORENSICS++-001'）或绝对路径
            split: 'train', 'val', 'test'
            transform: 图像变换
            mask_transform: mask 变换
            split_ratio: 如果没有预定义划分，按此比例随机划分 (train, val, test)
        """
        self.root_dir = os.path.abspath(root_dir)   # 转为绝对路径，方便后续拼接
        self.transform = transform
        self.mask_transform = mask_transform
        
        # 收集所有图片的相对路径和标签
        all_samples = self._collect_relative_samples()
        
        # 划分
        self.samples = self._split_samples(all_samples, split, split_ratio)
    
    def _collect_relative_samples(self):
        """收集所有 .png 文件相对于 root_dir 的路径，并分配标签"""
        samples = []
        # 真实图片：original_sequences 目录下所有 .png
        real_pattern = os.path.join(self.root_dir, 'original_sequences', '**', '*.png')
        for abs_path in glob.glob(real_pattern, recursive=True):
            rel_path = os.path.relpath(abs_path, self.root_dir)
            samples.append((rel_path, 0))
        # 伪造图片：manipulated_sequences 目录下所有 .png
        fake_pattern = os.path.join(self.root_dir, 'manipulated_sequences', '**', '*.png')
        for abs_path in glob.glob(fake_pattern, recursive=True):
            rel_path = os.path.relpath(abs_path, self.root_dir)
            samples.append((rel_path, 1))
        return samples
    
    def _split_samples(self, all_samples, split, split_ratio):
        """随机划分（可后续改进为基于视频的划分）"""
        np.random.seed(42)
        indices = np.random.permutation(len(all_samples))
        n_train = int(len(all_samples) * split_ratio[0])
        n_val = int(len(all_samples) * split_ratio[1])
        if split == 'train':
            idxs = indices[:n_train]
        elif split == 'val':
            idxs = indices[n_train:n_train+n_val]
        else:
            idxs = indices[n_train+n_val:]
        return [all_samples[i] for i in idxs]
    
    def _get_mask_relpath(self, img_relpath):
        """根据图片相对路径，推断 mask 的相对路径（如果存在）"""
        # 例如: 'manipulated_sequences/Deepfakes/c23/frames/001/0000.png'
        # 变为: 'manipulated_sequences/Deepfakes/c23/masks/001/0000.png'
        mask_relpath = img_relpath.replace('frames', 'masks')
        # 检查是否存在
        abs_mask_path = os.path.join(self.root_dir, mask_relpath)
        if os.path.exists(abs_mask_path):
            return mask_relpath
        return None
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_relpath, label = self.samples[idx]
        abs_img_path = os.path.join(self.root_dir, img_relpath)
        image = Image.open(abs_img_path).convert('RGB')

        # 尝试加载 mask
        mask = None
        mask_relpath = self._get_mask_relpath(img_relpath)
        if mask_relpath:
            abs_mask_path = os.path.join(self.root_dir, mask_relpath)
            if os.path.exists(abs_mask_path):
                mask = Image.open(abs_mask_path).convert('L')

        if self.transform:
            image = self.transform(image)

        if mask is not None:
            if self.mask_transform:
                mask = self.mask_transform(mask)
            else:
                mask = torch.from_numpy(np.array(mask)).float() / 255.0
                if mask.dim() == 2:
                    mask = mask.unsqueeze(0)
        else:
            # 没有 mask 时，创建全零 mask (1, H, W)
            H, W = image.shape[1], image.shape[2]
            mask = torch.zeros(1, H, W, dtype=torch.float32)

        return image, mask, label