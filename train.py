"""
LF-Loc Training Script
Main training loop with AMP, checkpointing, and logging.
"""
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import yaml

from models.lfloc import LFLOC
from data.dataset import build_dataloader
from losses.loss import MultiTaskLoss
from metrics.metrics import MetricTracker


def parse_args():
    parser = argparse.ArgumentParser(description="LF-Loc Training")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=2)   # ✅ CPU 用 2
    parser.add_argument("--epochs", type=int, default=3)        # ✅ 先跑 3 个 epoch 验证
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=0)   # ✅ CPU 必须为 0
    parser.add_argument("--device", type=str, default="cpu")    # ✅ 无 GPU
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--backbone", type=str, default="vit_base_patch14_dinov2")
    parser.add_argument("--fpn_out", type=int, default=256)
    parser.add_argument("--lambda_seg", type=float, default=1.0)
    parser.add_argument("--lambda_boundary", type=float, default=1.0)
    parser.add_argument("--lambda_cls", type=float, default=1.0)
    return parser.parse_args()


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, epoch):
    model.train()
    tracker = MetricTracker()
    total_loss = 0.0

    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]")
    for batch in pbar:
        images = batch["image"].to(device)
        # print(f"DEBUG: Input image shape: {images.shape}")
        masks = batch["mask"].to(device)

        optimizer.zero_grad()

        with autocast():
            pred_mask = model(images)
            losses = criterion(pred_mask, masks)

        scaler.scale(losses["total"]).backward()
        scaler.step(optimizer)
        scaler.update()

        # Track metrics
        pred_np = pred_mask.detach().cpu().numpy()
        mask_np = masks.detach().cpu().numpy()
        tracker.update(pred_np, mask_np, losses)

        total_loss += losses["total"].item()
        pbar.set_postfix(loss=losses["total"].item())

    results = tracker.get_results()
    results["Loss/total"] = total_loss / len(loader)
    return results


def validate(model, loader, criterion, device):
    model.eval()
    tracker = MetricTracker()
    total_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(loader, desc="[Val]"):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            with autocast():
                pred_mask = model(images)
                losses = criterion(pred_mask, masks)

            pred_np = pred_mask.detach().cpu().numpy()
            mask_np = masks.detach().cpu().numpy()
            tracker.update(pred_np, mask_np, losses)
            total_loss += losses["total"].item()

    results = tracker.get_results()
    results["Loss/total"] = total_loss / len(loader)
    return results


def main():
    args = parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Create directories
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # Model
    model = LFLOC(
        backbone_name=args.backbone,
        img_size=args.img_size,
        fpn_out=args.fpn_out,
    ).to(device)

    # Print model info
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"\n{'='*60}")
    print(f"LF-Loc Model Summary")
    print(f"{'='*60}")
    print(f"Backbone:      {args.backbone}")
    print(f"Image size:    {args.img_size}")
    print(f"FPN out ch:    {args.fpn_out}")
    print(f"Total params:  {total:,}")
    print(f"Trainable:     {trainable:,} ({trainable/total*100:.1f}%)")
    print(f"{'='*60}\n")

    # Loss
    criterion = MultiTaskLoss(
        lambda_cls=args.lambda_cls,
        lambda_seg=args.lambda_seg,
        lambda_boundary=args.lambda_boundary,
    )

    # Optimizer
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # LR scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # AMP scaler
    scaler = GradScaler()

    # DataLoaders (placeholder - C will replace with real FF++ data)
    train_loader = build_dataloader(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=args.img_size,
        dataset_size=200,
    )
    val_loader = build_dataloader(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=args.img_size,
        dataset_size=50,
    )

    # Training loop
    best_iou = 0.0
    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"{'='*60}")

        # Train
        train_results = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, epoch
        )

        # Validate
        val_results = validate(model, val_loader, criterion, device)

        # Update scheduler
        scheduler.step()

        # Print results
        print(f"\n{'Metric':<20} {'Train':>12} {'Val':>12}")
        print(f"{'-'*44}")
        for k in ["Loss/total", "IoU", "Dice", "F1@pixel"]:
            train_v = train_results.get(k, 0)
            val_v = val_results.get(k, 0)
            print(f"{k:<20} {train_v:>12.4f} {val_v:>12.4f}")

        # Save best model
        if val_results["IoU"] > best_iou:
            best_iou = val_results["IoU"]
            ckpt_path = os.path.join(args.save_dir, "best_model.pth")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_iou": best_iou,
                    "args": vars(args),
                },
                ckpt_path,
            )
            print(f"✅ Saved best model (IoU={best_iou:.4f})")

        # Save latest
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_iou": best_iou,
                "args": vars(args),
            },
            os.path.join(args.save_dir, "latest.pth"),
        )

    print(f"\n🎉 Training complete! Best IoU: {best_iou:.4f}")
    print(f"Checkpoints saved to: {args.save_dir}")


if __name__ == "__main__":
    main()
