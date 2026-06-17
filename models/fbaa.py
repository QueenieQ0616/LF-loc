"""
Forgery Boundary-Aware Attention module.
"""
import torch
import torch.nn as nn


class FBAA(nn.Module):
    """
    Boundary-region collaborative attention for forgery localization.

    Args:
        in_channels: input feature channels.
        hidden_channels: channels used by the lightweight attention branches.

    Returns:
        refined_feat: feature map with boundary-aware refinement.
        boundary_logits: one-channel boundary prediction logits.
    """

    def __init__(self, in_channels=256, hidden_channels=64):
        super().__init__()
        self.shared_proj = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )
        self.boundary_branch = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, kernel_size=1, bias=True),
        )
        self.region_branch = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, kernel_size=1, bias=True),
        )
        self.feature_proj = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )
        self.enhance = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels * 2 + 2, in_channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        shared = self.shared_proj(x)
        boundary_logits = self.boundary_branch(shared)
        region_logits = self.region_branch(shared)
        boundary_attention = torch.sigmoid(boundary_logits)
        region_attention = torch.sigmoid(region_logits)

        feat = self.feature_proj(x)
        enhanced = feat * (1.0 + boundary_attention) * region_attention
        enhanced = self.enhance(enhanced)

        gate_input = torch.cat([x, enhanced, boundary_attention, region_attention], dim=1)
        gate = self.gate(gate_input)
        refined_feat = x * (1.0 - gate) + enhanced * gate
        refined_feat = self.fuse(refined_feat)
        return refined_feat, boundary_logits


if __name__ == "__main__":
    x = torch.randn(2, 256, 224, 224)
    module = FBAA(256)
    feat, boundary = module(x)
    assert feat.shape == x.shape
    assert boundary.shape == (2, 1, 224, 224)
    print("FBAA test passed.")
