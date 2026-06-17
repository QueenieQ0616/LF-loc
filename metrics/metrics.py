"""
Evaluation metrics for LF-Loc.
"""
import numpy as np


def compute_iou(pred, target, threshold=0.5):
    """IoU for binary segmentation."""
    pred = (pred > threshold).astype(np.float32)
    target = (target > 0.5).astype(np.float32)

    intersection = (pred * target).sum()
    union = (pred + target).clip(0, 1).sum()
    if union < 1e-6:
        return 1.0 if intersection < 1e-6 else 0.0
    return intersection / union


def compute_dice(pred, target, threshold=0.5):
    """Dice coefficient for binary segmentation."""
    pred = (pred > threshold).astype(np.float32)
    target = (target > 0.5).astype(np.float32)

    intersection = (pred * target).sum()
    total = pred.sum() + target.sum()
    if total < 1e-6:
        return 1.0 if intersection < 1e-6 else 0.0
    return 2.0 * intersection / total


def compute_f1_pixel(pred, target, threshold=0.5):
    """Pixel-level F1 score."""
    pred = (pred > threshold).astype(np.int32)
    target = (target > 0.5).astype(np.int32)

    tp = (pred * target).sum()
    fp = (pred * (1 - target)).sum()
    fn = ((1 - pred) * target).sum()

    if tp + fp == 0 or tp + fn == 0:
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compute_pixel_acc(pred, target, threshold=0.5):
    """Pixel accuracy for binary segmentation."""
    pred = (pred > threshold).astype(np.int32)
    target = (target > 0.5).astype(np.int32)
    return (pred == target).mean()


def compute_auc(pred_scores, labels):
    """ROC-AUC for classification."""
    if len(np.unique(labels)) < 2:
        return 0.5
    try:
        from sklearn.metrics import roc_auc_score
    except Exception as exc:
        raise ImportError("scikit-learn is required for AUC but is not available.") from exc
    return roc_auc_score(labels, pred_scores)


class MetricTracker:
    """Tracks segmentation metrics across batches."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.iou_list = []
        self.dice_list = []
        self.f1_list = []
        self.pixel_acc_list = []
        self.losses = {}

    def update(self, pred_mask, gt_mask, loss_dict=None):
        """
        Args:
            pred_mask: numpy array [B, 1, H, W] logits.
            gt_mask: numpy array [B, 1, H, W].
        """
        pred_prob = 1.0 / (1.0 + np.exp(-pred_mask))

        for i in range(pred_mask.shape[0]):
            self.iou_list.append(compute_iou(pred_prob[i], gt_mask[i]))
            self.dice_list.append(compute_dice(pred_prob[i], gt_mask[i]))
            self.f1_list.append(compute_f1_pixel(pred_prob[i], gt_mask[i]))
            self.pixel_acc_list.append(compute_pixel_acc(pred_prob[i], gt_mask[i]))

        if loss_dict:
            for k, v in loss_dict.items():
                if k not in self.losses:
                    self.losses[k] = []
                self.losses[k].append(float(v.detach().cpu() if hasattr(v, "detach") else v))

    def get_results(self):
        results = {
            "IoU": np.mean(self.iou_list) if self.iou_list else 0.0,
            "Dice": np.mean(self.dice_list) if self.dice_list else 0.0,
            "F1@pixel": np.mean(self.f1_list) if self.f1_list else 0.0,
            "PixelAcc": np.mean(self.pixel_acc_list) if self.pixel_acc_list else 0.0,
        }
        for k, v in self.losses.items():
            results[f"Loss/{k}"] = np.mean(v)
        return results


if __name__ == "__main__":
    pred = np.random.randn(2, 1, 224, 224)
    target = np.random.randint(0, 2, (2, 1, 224, 224)).astype(np.float32)

    tracker = MetricTracker()
    tracker.update(pred, target)
    results = tracker.get_results()

    print("Metrics test:")
    for k, v in results.items():
        print(f"  {k}: {v:.4f}")
    print("Metrics test passed!")
