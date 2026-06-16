# D:\LF-loc\dataloader\ffpp_dataset.py
import os
import glob
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class FFPPDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None, mask_transform=None, max_videos_per_class=None):
        """
        Args:
            root_dir: 包含 train/ 和 test/ 文件夹的父目录（例如 './data/FaceForensics++'）
            split: 'train' 或 'test'
            transform: 图像变换
            mask_transform: mask 变换
            max_videos_per_class: 每个类别（真实/每种伪造方法）最多加载的视频文件夹数，None表示全部
        """
        self.root_dir = os.path.abspath(root_dir)
        self.split = split
        self.data_dir = os.path.join(self.root_dir, split)
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"Split directory not found: {self.data_dir}")
        
        self.transform = transform
        self.mask_transform = mask_transform
        self.max_videos_per_class = max_videos_per_class
        
        # 收集所有样本（图片相对路径相对于 self.data_dir）
        self.samples = self._collect_samples()
    
    def _collect_samples(self):
        """收集 self.data_dir 下所有 .png 文件，并分配标签"""
        samples = []
        # 真实图片：original_sequences 目录
        real_dirs = [
            os.path.join(self.data_dir, 'original_sequences', 'youtube', 'c23', 'frames'),
            os.path.join(self.data_dir, 'original_sequences', 'actors', 'c23', 'frames')
        ]
        for real_dir in real_dirs:
            if os.path.exists(real_dir):
                samples += self._collect_from_directory(real_dir, label=0)
        
        # 伪造图片：manipulated_sequences 下的各种方法
        fake_methods = ['DeepFakeDetection', 'Deepfakes', 'Face2Face', 'FaceShifter', 'FaceSwap', 'NeuralTextures']
        for method in fake_methods:
            method_dir = os.path.join(self.data_dir, 'manipulated_sequences', method, 'c23', 'frames')
            if os.path.exists(method_dir):
                samples += self._collect_from_directory(method_dir, label=1)
        return samples
    
    def _collect_from_directory(self, base_path, label):
        """收集 base_path 下所有视频文件夹中的 .png 图片，返回 (rel_path, label) 列表"""
        samples = []
        # 获取所有视频文件夹
        video_folders = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
        if self.max_videos_per_class:
            video_folders = video_folders[:self.max_videos_per_class]
        for vid in video_folders:
            vid_path = os.path.join(base_path, vid)
            for img_file in glob.glob(os.path.join(vid_path, '*.png')):
                # 存储相对路径（相对于 self.data_dir）
                rel_path = os.path.relpath(img_file, self.data_dir)
                samples.append((rel_path, label))
        return samples
    
    def _get_mask_relpath(self, img_relpath):
        """根据图片相对路径，推断 mask 的相对路径（如果存在）"""
        mask_relpath = img_relpath.replace('frames', 'masks')
        abs_mask_path = os.path.join(self.data_dir, mask_relpath)
        if os.path.exists(abs_mask_path):
            return mask_relpath
        return None
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_relpath, label = self.samples[idx]
        abs_img_path = os.path.join(self.data_dir, img_relpath)
        image = Image.open(abs_img_path).convert('RGB')
        
        # 尝试加载 mask
        mask = None
        mask_relpath = self._get_mask_relpath(img_relpath)
        if mask_relpath:
            abs_mask_path = os.path.join(self.data_dir, mask_relpath)
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