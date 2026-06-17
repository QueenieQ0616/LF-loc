"""
Frozen Backbone Module
Supports: DINOv2-ViT-B/14 (default) and CLIP-ViT-B/16
"""
import torch
import torch.nn as nn
import timm


class FrozenBackbone(nn.Module):
    """
    Frozen ViT backbone.
    Output: feature tensor of shape [B, C, H, W]
    - DINOv2: C=768, H=W=16 (for 224 input)
    - CLIP:   C=768, H=W=14 (for 224 input)
    """

    def __init__(self, model_name="vit_base_patch14_dinov2", img_size=224, pretrained=True):
        super().__init__()
        self.model_name = model_name
        self.img_size = img_size
        self.pretrained = pretrained

        # Supported models
        supported = {
            "vit_base_patch14_dinov2": "vit_base_patch14_dinov2",
            "clip_vit_base_patch16": "vit_base_patch16_clip_224",
        }
        assert model_name in supported, f"Unsupported model: {model_name}"

        self.model = timm.create_model(
            supported[model_name],
            pretrained=pretrained,
            img_size=img_size,
        )

        # Freeze all parameters
        for p in self.model.parameters():
            p.requires_grad = False

        # Store output info for downstream modules
        if "dinov2" in model_name:
            self.out_channels = 768
            self.feat_size = img_size // 14  # 16 for 224
        else:  # CLIP
            self.out_channels = 768
            self.feat_size = img_size // 16  # 14 for 224

    @torch.no_grad()
    def forward(self, x):
        """
        Args: x [B, 3, H, W]
        Returns: features [B, C, feat_H, feat_W]
        """
        out = self.model.forward_features(x)

        B = out.shape[0]
        C = self.out_channels

        if out.dim() == 3:
            # [B, N, C]
            # Drop the first token (CLS / register token).
            out = out[:, 1:, :]  # 257 -> 256

            N = out.shape[1]
            H = W = int(N ** 0.5)
            out = out.transpose(1, 2).reshape(B, C, H, W)

        elif out.dim() == 4:
            # Already [B, C, H, W]
            pass

        return out


if __name__ == "__main__":
    # Quick test
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = FrozenBackbone(model_name="vit_base_patch14_dinov2", img_size=224)
    model = model.to(device)
    model.eval()

    x = torch.randn(2, 3, 224, 224).to(device)
    out = model(x)
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Output channels: {model.out_channels}, Feat size: {model.feat_size}")
    print("Backbone test passed!")
