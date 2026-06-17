"""
Segmentation Head Module
PLACEHOLDER VERSION - to be replaced/enhanced by A
"""
import torch
import torch.nn as nn


class SegHead(nn.Module):
    """
    Lightweight segmentation head.
    Input:  [B, C, H, W]
    Output: [B, 1, H, W]  (logits)
    """

    def __init__(self, in_channels=256):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 1, kernel_size=1, bias=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x


if __name__ == "__main__":
    x = torch.randn(2, 256, 56, 56)
    head = SegHead(256)
    out = head(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {out.shape}")
    assert out.shape == (2, 1, 56, 56)
    print("SegHead test passed!")
