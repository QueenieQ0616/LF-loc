"""
LF-Loc: Lightweight Forgery Localization Model
Main model file - assembles all components.
"""
import torch
import torch.nn as nn
from .backbone import FrozenBackbone
from .fpn import LightFPN
from .fbaa import FBAA
from .head import SegHead


class LFLOC(nn.Module):
    """
    LF-Loc main model.

    Pipeline:
    Image -> FrozenBackbone -> LightFPN -> FBAA -> SegHead -> Mask Logits

    Args:
        backbone_name: "vit_base_patch14_dinov2" or "clip_vit_base_patch16"
        img_size:      input image size (default 224)
        fpn_out:       FPN output channels (default 256)
        freeze_backbone: whether to freeze backbone (default True)
    """

    def __init__(
        self,
        backbone_name="vit_base_patch14_dinov2",
        img_size=224,
        fpn_out=256,
        freeze_backbone=True,
    ):
        super().__init__()
        # Backbone (frozen)
        self.backbone = FrozenBackbone(backbone_name, img_size)
        if not freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = True

        # Light FPN
        self.fpn = LightFPN(
            in_channels=self.backbone.out_channels,
            out_channels=fpn_out,
            upscale_factor=img_size // self.backbone.feat_size,
        )

        # FBAA module (placeholder -> A will replace)
        self.fbaa = FBAA(in_channels=fpn_out)

        # Segmentation head
        self.head = SegHead(in_channels=fpn_out)

        # Store config for reference
        self.config = {
            "backbone": backbone_name,
            "img_size": img_size,
            "fpn_out": fpn_out,
        }

    def forward(self, x):
        """
        Args:   x [B, 3, H, W]
        Return: logits [B, 1, H, W]
        """
        feat = self.backbone(x)
        feat = self.fpn(feat)
        feat = self.fbaa(feat)
        logits = self.head(feat)
        return logits

    def get_trainable_params(self):
        """Return number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = LFLOC(backbone_name="vit_base_patch14_dinov2", img_size=224)
    model = model.to(device)

    x = torch.randn(2, 3, 224, 224).to(device)
    logits = model(x)

    print(f"\n{'='*50}")
    print(f"Model: LF-Loc")
    print(f"{'='*50}")
    print(f"Input shape:   {x.shape}")
    print(f"Output shape:  {logits.shape}")
    print(f"Backbone:      {model.config['backbone']}")
    print(f"Img size:      {model.config['img_size']}")
    print(f"FPN out:       {model.config['fpn_out']}")
    print(f"Trainable params: {model.get_trainable_params():,}")
    print(f"{'='*50}")
    print("✅ LF-Loc model test passed!")
