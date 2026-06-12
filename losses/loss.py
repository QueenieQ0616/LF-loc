"""
Loss Functions
Multi-task loss: L_total = λ1*L_cls + λ2*(L_bce + L_dice) + λ3*L_boundary
D will enhance this file later.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BCELoss(nn.Module):
    """Binary Cross Entropy Loss"""

    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred, target):
        return self.bce(pred, target)


class DiceLoss(nn.Module):
    """Dice Loss for binary segmentation"""

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


class BoundaryLoss(nn.Module):
    """Boundary-aware loss using Sobel filter to detect edges"""

    def __init__(self):
        super().__init__()
        # Sobel kernels for edge detection
        sobel_x = torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]], dtype=torch.float32)
        sobel_y = torch.tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]], dtype=torch.float32)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def forward(self, pred, target):
        # pred: [B, 1, H, W] logits
        # target: [B, 1, H, W]
        pred_prob = torch.sigmoid(pred)

        # Compute gradients using Sobel
        pred_grad_x = F.conv2d(pred_prob, self.sobel_x, padding=1)
        pred_grad_y = F.conv2d(pred_prob, self.sobel_y, padding=1)
        pred_boundary = torch.sqrt(pred_grad_x**2 + pred_grad_y**2 + 1e-6)

        target_grad_x = F.conv2d(target, self.sobel_x, padding=1)
        target_grad_y = F.conv2d(target, self.sobel_y, padding=1)
        target_boundary = torch.sqrt(target_grad_x**2 + target_grad_y**2 + 1e-6)

        return F.mse_loss(pred_boundary, target_boundary)


class MultiTaskLoss(nn.Module):
    """
    Combined loss: L_total = λ1*L_cls + λ2*(L_bce + L_dice) + λ3*L_boundary
    Default: all λ = 1.0 (D will tune later)
    """

    def __init__(self, lambda_cls=1.0, lambda_seg=1.0, lambda_boundary=1.0):
        super().__init__()
        self.lambda_cls = lambda_cls
        self.lambda_seg = lambda_seg
        self.lambda_boundary = lambda_boundary

        self.bce = BCELoss()
        self.dice = DiceLoss()
        self.boundary = BoundaryLoss()
        self.cls_loss = nn.BCEWithLogitsLoss()

    def forward(self, pred_mask, gt_mask, pred_cls=None, gt_cls=None):
        """
        Args:
            pred_mask: [B, 1, H, W] segmentation logits
            gt_mask:   [B, 1, H, W] binary mask
            pred_cls:  [B, 1] classification logits (optional)
            gt_cls:    [B, 1] binary label (optional)
        """
        losses = {}

        # Segmentation losses
        losses["bce"] = self.bce(pred_mask, gt_mask)
        losses["dice"] = self.dice(pred_mask, gt_mask)
        losses["boundary"] = self.boundary(pred_mask, gt_mask)

        seg_loss = self.lambda_seg * (losses["bce"] + losses["dice"])

        total_loss = seg_loss + self.lambda_boundary * losses["boundary"]

        # Classification loss (if provided)
        if pred_cls is not None and gt_cls is not None:
            losses["cls"] = self.cls_loss(pred_cls, gt_cls)
            total_loss += self.lambda_cls * losses["cls"]

        losses["total"] = total_loss
        return losses


if __name__ == "__main__":
    pred = torch.randn(2, 1, 224, 224)
    target = torch.randint(0, 2, (2, 1, 224, 224)).float()

    criterion = MultiTaskLoss()
    losses = criterion(pred, target)
    for k, v in losses.items():
        print(f"{k}: {v.item():.4f}")
    print("✅ Loss test passed!")
