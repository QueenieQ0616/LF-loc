"""
Loss functions for LF-Loc.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BCELoss(nn.Module):
    """Binary cross entropy loss for segmentation logits."""

    def __init__(self):
        super().__init__()

    def forward(self, pred, target, weight=None):
        loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
        if weight is not None:
            loss = loss * weight
        return loss.mean()


class DiceLoss(nn.Module):
    """Dice loss for binary segmentation."""

    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        pred = pred.flatten(1)
        target = target.flatten(1)

        intersection = (pred * target).sum(1)
        dice = (2.0 * intersection + self.smooth) / (
            pred.sum(1) + target.sum(1) + self.smooth
        )
        return 1.0 - dice.mean()


class BoundaryTarget(nn.Module):
    """Generate a binary boundary target from a mask with Sobel filters."""

    def __init__(self, threshold=0.1):
        super().__init__()
        self.threshold = threshold
        sobel_x = torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]], dtype=torch.float32)
        sobel_y = torch.tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]], dtype=torch.float32)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def forward(self, mask):
        sobel_x = self.sobel_x.to(device=mask.device, dtype=mask.dtype)
        sobel_y = self.sobel_y.to(device=mask.device, dtype=mask.dtype)
        grad_x = F.conv2d(mask, sobel_x, padding=1)
        grad_y = F.conv2d(mask, sobel_y, padding=1)
        boundary = torch.sqrt(grad_x**2 + grad_y**2 + 1e-6)
        max_per_sample = boundary.flatten(1).amax(dim=1).clamp_min(1e-6)
        boundary = boundary / max_per_sample.view(-1, 1, 1, 1)
        return (boundary > self.threshold).float()


class BoundaryLoss(nn.Module):
    """
    Boundary loss.

    If explicit boundary logits are provided by FBAA, supervise them with a
    Sobel-derived GT boundary map. Otherwise, fall back to a soft mask-boundary
    consistency loss for backward compatibility.
    """

    def __init__(self):
        super().__init__()
        self.target_builder = BoundaryTarget()
        self.boundary_bce = nn.BCEWithLogitsLoss()

    def forward(self, pred_mask, target_mask, pred_boundary=None):
        target_boundary = self.target_builder(target_mask)

        if pred_boundary is not None:
            if pred_boundary.shape[-2:] != target_boundary.shape[-2:]:
                pred_boundary = F.interpolate(
                    pred_boundary,
                    size=target_boundary.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            return self.boundary_bce(pred_boundary, target_boundary)

        pred_prob = torch.sigmoid(pred_mask)
        pred_boundary = self.target_builder(pred_prob)
        return F.mse_loss(pred_boundary, target_boundary)


class MultiTaskLoss(nn.Module):
    """
    Combined loss:
    L_total = lambda_cls * L_cls
            + lambda_seg * (L_bce + L_dice)
            + lambda_boundary * L_boundary
    """

    def __init__(
        self,
        lambda_cls=1.0,
        lambda_seg=1.0,
        lambda_boundary=1.0,
        boundary_bce_weight=0.0,
    ):
        super().__init__()
        self.lambda_cls = lambda_cls
        self.lambda_seg = lambda_seg
        self.lambda_boundary = lambda_boundary
        self.boundary_bce_weight = boundary_bce_weight

        self.bce = BCELoss()
        self.dice = DiceLoss()
        self.boundary = BoundaryLoss()
        self.boundary_target = BoundaryTarget()
        self.cls_loss = nn.BCEWithLogitsLoss()

    def forward(self, pred_mask, gt_mask=None, pred_cls=None, gt_cls=None, pred_boundary=None):
        """
        Args:
            pred_mask: segmentation logits [B, 1, H, W] or model output dict.
            gt_mask: binary mask [B, 1, H, W].
            pred_cls: optional classification logits [B, 1].
            gt_cls: optional binary image label [B, 1].
            pred_boundary: optional boundary logits [B, 1, H, W].
        """
        if isinstance(pred_mask, dict):
            outputs = pred_mask
            pred_mask = outputs["mask_logits"]
            pred_boundary = outputs.get("boundary_logits", pred_boundary)
            pred_cls = outputs.get("cls_logits", pred_cls)

        losses = {}
        bce_weight = None
        if self.boundary_bce_weight > 0:
            with torch.no_grad():
                boundary_weight = self.boundary_target(gt_mask)
                bce_weight = 1.0 + self.boundary_bce_weight * boundary_weight

        losses["bce"] = self.bce(pred_mask, gt_mask, weight=bce_weight)
        losses["dice"] = self.dice(pred_mask, gt_mask)

        seg_loss = self.lambda_seg * (losses["bce"] + losses["dice"])
        total_loss = seg_loss

        if self.lambda_boundary > 0:
            losses["boundary"] = self.boundary(pred_mask, gt_mask, pred_boundary)
            total_loss = total_loss + self.lambda_boundary * losses["boundary"]

        if pred_cls is not None and gt_cls is not None:
            losses["cls"] = self.cls_loss(pred_cls, gt_cls)
            total_loss += self.lambda_cls * losses["cls"]

        losses["total"] = total_loss
        return losses


if __name__ == "__main__":
    pred = torch.randn(2, 1, 224, 224)
    pred_boundary = torch.randn(2, 1, 224, 224)
    target = torch.randint(0, 2, (2, 1, 224, 224)).float()

    criterion = MultiTaskLoss()
    losses = criterion({"mask_logits": pred, "boundary_logits": pred_boundary}, target)
    for k, v in losses.items():
        print(f"{k}: {v.item():.4f}")
    print("Loss test passed.")
