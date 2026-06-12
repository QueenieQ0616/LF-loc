"""
Visualization Utilities
D will enhance this later for prediction vs GT visualization.
"""
import matplotlib.pyplot as plt
import numpy as np
import torch


def visualize_batch(images, pred_masks, gt_masks, save_path="viz_batch.png"):
    """
    Visualize a batch of images with predictions and ground truth.
    Args:
        images:    [B, 3, H, W] tensor or numpy
        pred_masks: [B, 1, H, W] tensor or numpy (logits)
        gt_masks:  [B, 1, H, W] tensor or numpy
    """
    if isinstance(images, torch.Tensor):
        images = images.detach().cpu().numpy()
    if isinstance(pred_masks, torch.Tensor):
        pred_masks = pred_masks.detach().cpu().numpy()
    if isinstance(gt_masks, torch.Tensor):
        gt_masks = gt_masks.detach().cpu().numpy()

    B = min(images.shape[0], 4)  # Show at most 4 samples

    fig, axes = plt.subplots(B, 3, figsize=(12, 3 * B))
    if B == 1:
        axes = axes.reshape(1, -1)

    for i in range(B):
        # Denormalize image (assuming ImageNet normalization)
        img = images[i].transpose(1, 2, 0)
        img = np.clip(img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]), 0, 1)

        pred = torch.sigmoid(torch.tensor(pred_masks[i][0])).numpy()
        gt = gt_masks[i][0]

        axes[i, 0].imshow(img)
        axes[i, 0].set_title("Input Image")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(pred, cmap="hot", vmin=0, vmax=1)
        axes[i, 1].set_title("Prediction")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(gt, cmap="hot", vmin=0, vmax=1)
        axes[i, 2].set_title("Ground Truth")
        axes[i, 2].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved visualization to {save_path}")


if __name__ == "__main__":
    # Quick test
    images = torch.randn(4, 3, 224, 224)
    preds = torch.randn(4, 1, 224, 224)
    gts = torch.randint(0, 2, (4, 1, 224, 224)).float()
    visualize_batch(images, preds, gts, "test_viz.png")
