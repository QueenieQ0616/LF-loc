"""
LF-Loc Training Script
Main training loop with AMP, checkpointing, and logging.
"""
import os
import argparse
import torch
import torch.optim as optim
from contextlib import nullcontext
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
    parser.add_argument("--batch_size", type=int, default=2)   # CPU smoke-test default
    parser.add_argument("--epochs", type=int, default=3)        # Smoke-test default
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=0)   # Windows-safe default
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--dataset", type=str, default="fake", choices=["fake", "ffpp"])
    parser.add_argument("--data_root", type=str, default="data/FaceForensics++/FaceForensics++")
    parser.add_argument("--compression", type=str, default="c23")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"],
    )
    parser.add_argument("--include_originals", action="store_true")
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--backbone", type=str, default="vit_base_patch14_dinov2")
    parser.add_argument("--no_pretrained_backbone", action="store_true")
    parser.add_argument("--disable_fbaa", action="store_true")
    parser.add_argument("--fpn_out", type=int, default=256)
    parser.add_argument("--train_size", type=int, default=200)
    parser.add_argument("--val_size", type=int, default=50)
    parser.add_argument("--lambda_seg", type=float, default=1.0)
    parser.add_argument("--lambda_boundary", type=float, default=1.0)
    parser.add_argument("--lambda_cls", type=float, default=1.0)
    parser.add_argument("--boundary_bce_weight", type=float, default=0.0)
    return parser.parse_args()


def amp_context(enabled):
    return autocast() if enabled else nullcontext()


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, epoch):
    model.train()
    tracker = MetricTracker()
    total_loss = 0.0
    use_amp = device.type == "cuda"

    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]")
    for batch in pbar:
        images = batch["image"].to(device)
        # print(f"DEBUG: Input image shape: {images.shape}")
        masks = batch["mask"].to(device)

        optimizer.zero_grad()

        with amp_context(use_amp):
            outputs = model(images, return_dict=True)
            losses = criterion(outputs, masks)

        if use_amp:
            scaler.scale(losses["total"]).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses["total"].backward()
            optimizer.step()

        # Track metrics
        pred_mask = outputs["mask_logits"]
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
    use_amp = device.type == "cuda"

    with torch.no_grad():
        for batch in tqdm(loader, desc="[Val]"):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            with amp_context(use_amp):
                outputs = model(images, return_dict=True)
                losses = criterion(outputs, masks)

            pred_mask = outputs["mask_logits"]
            pred_np = pred_mask.detach().cpu().numpy()
            mask_np = masks.detach().cpu().numpy()
            tracker.update(pred_np, mask_np, losses)
            total_loss += losses["total"].item()

    results = tracker.get_results()
    results["Loss/total"] = total_loss / len(loader)
    return results


def main():
    args = parse_args()
    requested_device = args.device
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        requested_device = "cpu"
    device = torch.device(requested_device)
    print(f"Using device: {device}")

    # Create directories
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # Model
    model = LFLOC(
        backbone_name=args.backbone,
        img_size=args.img_size,
        fpn_out=args.fpn_out,
        pretrained_backbone=not args.no_pretrained_backbone,
        use_fbaa=not args.disable_fbaa,
    ).to(device)

    # Print model info
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"\n{'='*60}")
    print(f"LF-Loc Model Summary")
    print(f"{'='*60}")
    print(f"Backbone:      {args.backbone}")
    print(f"Pretrained:    {not args.no_pretrained_backbone}")
    print(f"FBAA:          {not args.disable_fbaa}")
    print(f"Image size:    {args.img_size}")
    print(f"FPN out ch:    {args.fpn_out}")
    print(f"Boundary BCE:  {args.boundary_bce_weight}")
    print(f"Dataset:       {args.dataset}")
    if args.dataset == "ffpp":
        print(f"Data root:     {args.data_root}")
        print(f"Compression:   {args.compression}")
        print(f"Methods:       {', '.join(args.methods)}")
    print(f"Total params:  {total:,}")
    print(f"Trainable:     {trainable:,} ({trainable/total*100:.1f}%)")
    print(f"{'='*60}\n")

    # Loss
    criterion = MultiTaskLoss(
        lambda_cls=args.lambda_cls,
        lambda_seg=args.lambda_seg,
        lambda_boundary=args.lambda_boundary,
        boundary_bce_weight=args.boundary_bce_weight,
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
    scaler = GradScaler(enabled=device.type == "cuda")

    # DataLoaders (placeholder - C will replace with real FF++ data)
    train_loader = build_dataloader(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=args.img_size,
        dataset_size=args.train_size,
        pin_memory=device.type == "cuda",
        dataset_type=args.dataset,
        data_root=args.data_root,
        split="train",
        compression=args.compression,
        methods=args.methods,
        include_originals=args.include_originals,
    )
    val_loader = build_dataloader(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=args.img_size,
        dataset_size=args.val_size,
        pin_memory=device.type == "cuda",
        dataset_type=args.dataset,
        data_root=args.data_root,
        split="val",
        compression=args.compression,
        methods=args.methods,
        include_originals=args.include_originals,
        shuffle=False,
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
        for k in ["Loss/total", "IoU", "Dice", "F1@pixel", "PixelAcc"]:
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
            print(f"Saved best model (IoU={best_iou:.4f})")

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

    print(f"\nTraining complete! Best IoU: {best_iou:.4f}")
    print(f"Checkpoints saved to: {args.save_dir}")


if __name__ == "__main__":
    main()
