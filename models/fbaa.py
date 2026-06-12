"""
FBAA (Forgery Boundary-Aware Attention) Module
PLACEHOLDER VERSION - to be replaced by A (team leader)
"""
import torch
import torch.nn as nn


class FBAA(nn.Module):
    """
    Placeholder FBAA module.
    Passes input through unchanged (identity).
    A will replace this with the real implementation:
    - Boundary detection branch
    - Boundary response map as attention weights
    - Fusion with main features
    """

    def __init__(self, in_channels=256):
        super().__init__()
        # Placeholder: identity mapping
        self.identity = nn.Identity()

    def forward(self, x):
        return self.identity(x)


if __name__ == "__main__":
    x = torch.randn(2, 256, 56, 56)
    module = FBAA(256)
    out = module(x)
    assert out.shape == x.shape
    print("✅ FBAA placeholder test passed!")
