"""
Evaluation script for LF-Loc.

Reports:
1. Image-level AUC when cls_logits are available.
2. Fake-only mean IoU.
3. Fake-only mean Dice.
4. Fake-only Boundary F1 with 2-pixel tolerance.
5. Trainable parameters.
"""
import argparse
import csv
import math
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from data.dataset import DEFAULT_FFPP_METHODS, FFPPDataset
from models.lfloc import LFLOC


METHOD_ALIASES = {
    "Deepfakes": "DF",
    "Face2Face": "F2F",
    "FaceSwap": "FS",
    "NeuralTextures": "NT",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate LF-Loc checkpoints")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-root", type=str, default="data/FaceForensics++")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output-dir", type=str, default="eval_results")
    parser.add_argument("--dataset", type=str, default="ffpp", choices=["ffpp"])
    parser.add_argument("--compression", type=str, default="c23")
    parser.add_argument("--methods", nargs="+", default=list(DEFAULT_FFPP_METHODS))
    parser.add_argument("--include-originals", action="store_true")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--method-name", type=str, default="LF-Loc")
    return parser.parse_args()


def load_config(path):
    if not path or not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def build_model(args, config, checkpoint):
    ckpt_args = checkpoint.get("args", {}) if isinstance(checkpoint, dict) else {}
    model_cfg = config.get("model", {})

    backbone = ckpt_args.get("backbone", model_cfg.get("backbone", "vit_base_patch14_dinov2"))
    img_size = int(ckpt_args.get("img_size", model_cfg.get("img_size", 224)))
    fpn_out = int(ckpt_args.get("fpn_out", model_cfg.get("fpn_out", 256)))
    pretrained_backbone = not bool(ckpt_args.get("no_pretrained_backbone", False))
    use_fbaa = not bool(ckpt_args.get("disable_fbaa", False))

    model = LFLOC(
        backbone_name=backbone,
        img_size=img_size,
        fpn_out=fpn_out,
        pretrained_backbone=pretrained_backbone,
        use_fbaa=use_fbaa,
    )
    return model, img_size


def get_state_dict(checkpoint):
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    return checkpoint


def sigmoid_np(logits):
    return torch.sigmoid(logits).detach().float().cpu().numpy()


def binary_counts(pred_bin, gt_bin):
    pred_bin = pred_bin.astype(bool)
    gt_bin = gt_bin.astype(bool)
    tp = np.logical_and(pred_bin, gt_bin).sum(dtype=np.float64)
    fp = np.logical_and(pred_bin, ~gt_bin).sum(dtype=np.float64)
    fn = np.logical_and(~pred_bin, gt_bin).sum(dtype=np.float64)
    return tp, fp, fn


def compute_iou_dice(pred_prob, gt_mask, threshold):
    pred_bin = pred_prob > threshold
    gt_bin = gt_mask > 0.5
    tp, fp, fn = binary_counts(pred_bin, gt_bin)

    iou_den = tp + fp + fn
    dice_den = 2.0 * tp + fp + fn
    iou = tp / iou_den if iou_den > 0 else math.nan
    dice = 2.0 * tp / dice_den if dice_den > 0 else math.nan
    return iou, dice


def max_pool_binary(mask, radius):
    tensor = torch.from_numpy(mask.astype(np.float32))[None, None]
    pooled = F.max_pool2d(tensor, kernel_size=2 * radius + 1, stride=1, padding=radius)
    return pooled[0, 0].numpy() > 0.5


def make_boundary(mask):
    mask = mask.astype(bool)
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)
    dilated = max_pool_binary(mask, radius=1)
    eroded = ~max_pool_binary(~mask, radius=1)
    return np.logical_and(dilated, ~eroded)


def compute_boundary_f1(pred_prob, gt_mask, threshold, tolerance=2):
    pred_boundary = make_boundary(pred_prob > threshold)
    gt_boundary = make_boundary(gt_mask > 0.5)

    pred_count = pred_boundary.sum(dtype=np.float64)
    gt_count = gt_boundary.sum(dtype=np.float64)
    if pred_count == 0 and gt_count == 0:
        return 1.0
    if pred_count == 0 or gt_count == 0:
        return 0.0

    gt_boundary_dil = max_pool_binary(gt_boundary, radius=tolerance)
    pred_boundary_dil = max_pool_binary(pred_boundary, radius=tolerance)

    precision = np.logical_and(pred_boundary, gt_boundary_dil).sum(dtype=np.float64) / pred_count
    recall = np.logical_and(gt_boundary, pred_boundary_dil).sum(dtype=np.float64) / gt_count
    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def safe_auc(labels, scores):
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    valid = ~np.isnan(scores)
    labels = labels[valid]
    scores = scores[valid]
    if len(labels) == 0 or len(np.unique(labels)) < 2:
        return math.nan
    try:
        from sklearn.metrics import roc_auc_score
    except Exception as exc:
        raise ImportError("scikit-learn is required to compute Image AUC.") from exc
    return float(roc_auc_score(labels, scores))


def mean_or_nan(values):
    values = [v for v in values if v is not None and not math.isnan(v)]
    return float(np.mean(values)) if values else math.nan


def format_float(value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return f"{value:.6f}"


def trainable_params_m(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1_000_000.0


def get_cls_probability(outputs):
    if not isinstance(outputs, dict):
        return None
    cls_logits = outputs.get("cls_logits")
    if cls_logits is None:
        return None
    if cls_logits.ndim == 2 and cls_logits.shape[1] == 2:
        return torch.softmax(cls_logits, dim=1)[:, 1]
    return torch.sigmoid(cls_logits.reshape(cls_logits.shape[0], -1)[:, 0])


def get_metadata(dataset):
    metadata = {}
    for sample in getattr(dataset, "samples", []):
        image_path = str(sample.get("image", ""))
        method = sample.get("method", "")
        metadata[image_path] = {
            "manipulation_type": METHOD_ALIASES.get(method, method),
            "label": float(sample.get("label", 0.0)),
        }
    return metadata


def summarize(rows, method, dataset_name, params_text):
    labels = [int(r["gt_label"]) for r in rows]
    scores = [float(r["fake_probability"]) if r["fake_probability"] != "" else math.nan for r in rows]
    return {
        "Method": method,
        "Dataset": dataset_name,
        "Samples": len(rows),
        "Image AUC": safe_auc(labels, scores),
        "IoU": mean_or_nan([float(r["iou"]) if r["iou"] != "" else math.nan for r in rows]),
        "Dice": mean_or_nan([float(r["dice"]) if r["dice"] != "" else math.nan for r in rows]),
        "Boundary F1": mean_or_nan(
            [float(r["boundary_f1"]) if r["boundary_f1"] != "" else math.nan for r in rows]
        ),
        "Trainable Params": params_text,
    }


def main():
    args = parse_args()
    config = load_config(args.config)
    device_name = args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model, img_size = build_model(args, config, checkpoint)
    model.load_state_dict(get_state_dict(checkpoint), strict=True)
    model.to(device)
    model.eval()

    params_m = trainable_params_m(model)
    params_text = f"{params_m:.2f}M"

    dataset = FFPPDataset(
        data_root=args.data_root,
        split=args.split,
        img_size=img_size,
        compression=args.compression,
        methods=args.methods,
        include_originals=args.include_originals,
        max_samples=None,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    metadata = get_metadata(dataset)

    rows = []
    cls_available = False

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            gt_masks = batch["mask"]
            labels = batch["label"].view(-1).cpu().numpy().astype(int)
            image_paths = list(batch.get("image_path", [""] * len(labels)))

            outputs = model(images, return_dict=True)
            mask_logits = outputs["mask_logits"] if isinstance(outputs, dict) else outputs
            if mask_logits.shape[-2:] != gt_masks.shape[-2:]:
                mask_logits = F.interpolate(
                    mask_logits,
                    size=gt_masks.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            mask_probs = sigmoid_np(mask_logits)

            cls_prob_tensor = get_cls_probability(outputs)
            if cls_prob_tensor is not None:
                cls_available = True
                fake_probs = cls_prob_tensor.detach().float().cpu().numpy()
            else:
                fake_probs = [math.nan] * len(labels)

            for i, image_path in enumerate(image_paths):
                meta = metadata.get(str(image_path), {})
                gt_mask = gt_masks[i, 0].float().cpu().numpy()
                gt_mask = (gt_mask > 0.5).astype(np.float32)
                pred_prob = mask_probs[i, 0]

                is_fake_nonempty = labels[i] == 1 and gt_mask.sum() > 0
                if is_fake_nonempty:
                    iou, dice = compute_iou_dice(pred_prob, gt_mask, args.threshold)
                    boundary_f1 = compute_boundary_f1(pred_prob, gt_mask, args.threshold)
                else:
                    iou, dice, boundary_f1 = math.nan, math.nan, math.nan

                rows.append(
                    {
                        "image_path": str(image_path),
                        "gt_label": int(labels[i]),
                        "fake_probability": "" if math.isnan(float(fake_probs[i])) else f"{float(fake_probs[i]):.8f}",
                        "manipulation_type": meta.get("manipulation_type", ""),
                        "iou": format_float(iou),
                        "dice": format_float(dice),
                        "boundary_f1": format_float(boundary_f1),
                    }
                )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    per_image_path = output_dir / "per_image_results.csv"
    with per_image_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image_path",
                "gt_label",
                "fake_probability",
                "manipulation_type",
                "iou",
                "dice",
                "boundary_f1",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    dataset_name = f"FF++-{args.split}"
    summary_rows = [summarize(rows, args.method_name, dataset_name, params_text)]

    group_names = ["DF", "F2F", "FS", "NT"]
    for group_name in group_names:
        group_rows = [r for r in rows if r["manipulation_type"] == group_name]
        if group_rows:
            summary_rows.append(summarize(group_rows, f"{args.method_name}-{group_name}", dataset_name, params_text))

    summary_path = output_dir / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "Method",
                "Dataset",
                "Samples",
                "Image AUC",
                "IoU",
                "Dice",
                "Boundary F1",
                "Trainable Params",
            ],
        )
        writer.writeheader()
        for row in summary_rows:
            row = row.copy()
            for key in ["Image AUC", "IoU", "Dice", "Boundary F1"]:
                row[key] = format_float(row[key])
            writer.writerow(row)

    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Dataset samples: {len(dataset)}")
    print(f"Classification head available: {cls_available}")
    if not cls_available:
        print("Image AUC left blank because LFLOC does not output cls_logits.")
    print(f"Trainable Parameters: {params_text}")
    print(f"Saved summary: {summary_path}")
    print(f"Saved per-image results: {per_image_path}")
    print("\nSummary:")
    for row in summary_rows:
        print(
            f"{row['Method']}: samples={row['Samples']}, "
            f"AUC={format_float(row['Image AUC']) or 'N/A'}, "
            f"IoU={format_float(row['IoU']) or 'N/A'}, "
            f"Dice={format_float(row['Dice']) or 'N/A'}, "
            f"BoundaryF1={format_float(row['Boundary F1']) or 'N/A'}, "
            f"Params={row['Trainable Params']}"
        )


if __name__ == "__main__":
    main()
