"""
Dataset Module
Placeholder DataLoader - C will replace with real FF++ data.
"""
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class FakeDataset(Dataset):
    """
    Placeholder dataset that generates random data.
    Simulates: image + binary mask.
    C will replace this with real FF++ DataLoader.
    """

    def __init__(self, size=1000, img_size=224):
        self.size = size
        self.img_size = img_size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # Random image (simulating face crop)
        image = torch.randn(3, self.img_size, self.img_size)

        # Random binary mask (simulating forgery localization GT)
        mask = torch.rand(1, self.img_size, self.img_size)
        mask = (mask > 0.7).float()

        return {
            "image": image,
            "mask": mask,
        }


def build_dataloader(batch_size=8, num_workers=2, img_size=224, dataset_size=1000):
    """
    Build a placeholder DataLoader.
    C will replace this function with the real FF++ DataLoader.
    """
    dataset = FakeDataset(size=dataset_size, img_size=img_size)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    return loader


if __name__ == "__main__":
    loader = build_dataloader(batch_size=4, dataset_size=20)
    batch = next(iter(loader))
    print(f"Image batch: {batch['image'].shape}")
    print(f"Mask batch:  {batch['mask'].shape}")
    print("✅ DataLoader test passed!")
