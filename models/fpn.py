"""
Lightweight FPN (Feature Pyramid Network)
Reduces channel dimension and upsamples to target resolution.
Total params: ~0.5M
"""
import torch
import torch.nn as nn
from .backbone import FrozenBackbone


class LightFPN(nn.Module):
    """
    Lightweight FPN: 1x1 conv channel reduction + bilinear upsample.
    Input:  [B, C_in, H_f, W_f]  (from ViT backbone)
    Output: [B, C_out, H_out, W_out]
    """

    def __init__(self, in_channels=768, out_channels=256, upscale_factor=14):
        super().__init__()
        self.reduce = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.upscale_factor = upscale_factor

    def forward(self, x):
        x = self.reduce(x)
        x = self.bn(x)
        x = self.relu(x)
        # Upsample to target resolution (e.g., 224x224)
        x = torch.nn.functional.interpolate(
            x,
            scale_factor=self.upscale_factor,
            mode="bilinear",
            align_corners=False,
        )
        return x


class FPNWithBackbone(nn.Module):
    """
    Convenience wrapper: Backbone + FPN in one module.
    Used as placeholder before A's FBAA is ready.
    """

    def __init__(self, backbone_name="vit_base_patch14_dinov2", img_size=224):
        super().__init__()
        self.backbone = FrozenBackbone(backbone_name, img_size)
        self.fpn = LightFPN(
            in_channels=self.backbone.out_channels,
            out_channels=256,
            upscale_factor=img_size // self.backbone.feat_size,
        )

    def forward(self, x):
        feat = self.backbone(x)
        feat = self.fpn(feat)
        return feat


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    fpn = FPNWithBackbone(img_size=224).to(device)
    x = torch.randn(2, 3, 224, 224).to(device)
    out = fpn(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {out.shape}")

    total_params = sum(p.numel() for p in fpn.parameters() if p.requires_grad)
    print(f"Trainable params: {total_params:,}")
    print("✅ FPN test passed!")
