"""
FF++ comparison evaluation for LF-Loc.

This script reports the metrics used for comparison with Face X-ray and
FakeLocator-style results:
  - Image-level AUC, only when the model provides cls_logits.
  - Fake-only mean IoU at thresholds 0.1 and 0.5.
  - Fake-only mean Dice at thresholds 0.1 and 0.5.
  - PBCA over all valid pixels at thresholds 0.1 and 0.5.
"""
import argparse
import csv
import math
import os
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
THRESHOLDS = (0.1, 0.5)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate LF-Loc on FF++ comparison metrics")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-root", type=str, default="data/FaceForensics++")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--output-dir", type=str, default="eval_results")
    parser.add_argument("--compression", type=str, default="c23")
    parser.add_argument("--methods", nargs="+", default=list(DEFAULT_FFPP_METHODS))
    parser.add_argument("--include-originals", action="store_true")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--method-name", type=str, default="LF-Loc")
    parser.add_argument("--dataset-name", type=str, default="FF++ c23")
    return parser.parse_args()


def load_config(path):
    if not path or not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_checkpoint(path):
    return torch.load(path, map_location="cpu")


def get_state_dict(checkpoint):
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    return checkpoint


def build_model(config, checkpoint):
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


def get_cls_probability(outputs):
    if not isinstance(outputs, dict):
        return None
    cls_logits = outputs.get("cls_logits")
    if cls_logits is None:
        return None
    if cls_logits.ndim == 2 and cls_logits.shape[1] == 2:
        return torch.softmax(cls_logits, dim=1)[:, 1]
    return torch.sigmoid(cls_logits.reshape(cls_logits.shape[0], -1)[:, 0])


def safe_auc(labels, scores):
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    valid = np.isfinite(scores)
    labels = labels[valid]
    scores = scores[valid]
    if labels.size == 0 or np.unique(labels).size < 2:
        return None
    try:
        from sklearn.metrics import roc_auc_score
    except Exception as exc:
        raise ImportError("scikit-learn is required for Image-level AUC.") from exc
    return float(roc_auc_score(labels, scores))


def counts_for_threshold(pred_prob, gt_mask, threshold):
    pred = pred_prob > threshold
    gt = gt_mask > 0.5
    tp = np.logical_and(pred, gt).sum(dtype=np.float64)
    tn = np.logical_and(~pred, ~gt).sum(dtype=np.float64)
    fp = np.logical_and(pred, ~gt).sum(dtype=np.float64)
    fn = np.logical_and(~pred, gt).sum(dtype=np.float64)
    return tp, tn, fp, fn


def iou_dice_from_counts(tp, fp, fn):
    iou_den = tp + fp + fn
    dice_den = 2.0 * tp + fp + fn
    iou = tp / iou_den if iou_den > 0 else None
    dice = 2.0 * tp / dice_den if dice_den > 0 else None
    return iou, dice


def pbca_from_counts(tp, tn, fp, fn):
    den = tp + tn + fp + fn
    return (tp + tn) / den if den > 0 else None


def fmt(value):
    if value is None:
        return ""
    return f"{value:.6f}"


def metadata_by_path(dataset):
    result = {}
    for sample in getattr(dataset, "samples", []):
        image_path = str(sample.get("image", ""))
        method = sample.get("method", "")
        result[image_path] = {
            "manipulation_type": METHOD_ALIASES.get(method, method),
            "label": int(float(sample.get("label", 0.0))),
        }
    return result


def summarize_rows(rows, threshold):
    iou_key = f"iou_at_{int(threshold * 10):02d}"
    dice_key = f"dice_at_{int(threshold * 10):02d}"
    pbca_key = f"pbca_at_{int(threshold * 10):02d}"

    ious = [float(row[iou_key]) for row in rows if row[iou_key] != ""]
    dices = [float(row[dice_key]) for row in rows if row[dice_key] != ""]

    tp = tn = fp = fn = 0.0
    count_key = f"counts_at_{int(threshold * 10):02d}"
    for row in rows:
        c = row.get(count_key)
        if c is None:
            continue
        tp += c[0]
        tn += c[1]
        fp += c[2]
        fn += c[3]

    return {
        "iou": float(np.mean(ious)) if ious else None,
        "dice": float(np.mean(dices)) if dices else None,
        "pbca": pbca_from_counts(tp, tn, fp, fn),
    }


def main():
    args = parse_args()
    config = load_config(args.config)
    device_name = args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)

    checkpoint = load_checkpoint(args.checkpoint)
    model, img_size = build_model(config, checkpoint)
    model.load_state_dict(get_state_dict(checkpoint), strict=True)
    model.to(device)
    model.eval()

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
    metadata = metadata_by_path(dataset)

    rows = []
    cls_available = False
    auc_labels = []
    auc_scores = []
    group_auc = {name: {"labels": [], "scores": []} for name in ["DF", "F2F", "FS", "NT"]}

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            gt_masks = batch["mask"]
            labels = batch["label"].view(-1).cpu().numpy().astype(np.int64)
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
            mask_probs = torch.sigmoid(mask_logits).detach().float().cpu().numpy()

            cls_probs = get_cls_probability(outputs)
            if cls_probs is not None:
                cls_available = True
                fake_probs = cls_probs.detach().float().cpu().numpy()
            else:
                fake_probs = [None] * len(labels)

            for i, image_path in enumerate(image_paths):
                image_path = str(image_path)
                meta = metadata.get(image_path, {})
                manipulation_type = meta.get("manipulation_type", "")
                gt_label = int(labels[i])
                gt_mask = (gt_masks[i, 0].float().cpu().numpy() > 0.5).astype(np.float32)
                pred_prob = mask_probs[i, 0]
                fake_prob = fake_probs[i]

                if fake_prob is not None:
                    auc_labels.append(gt_label)
                    auc_scores.append(float(fake_prob))
                    if gt_label == 1 and manipulation_type in group_auc:
                        group_auc[manipulation_type]["labels"].append(1)
                        group_auc[manipulation_type]["scores"].append(float(fake_prob))
                    elif gt_label == 0:
                        for group in group_auc.values():
                            group["labels"].append(0)
                            group["scores"].append(float(fake_prob))

                row = {
                    "image_path": image_path,
                    "gt_label": gt_label,
                    "fake_probability": "" if fake_prob is None else f"{float(fake_prob):.8f}",
                    "manipulation_type": manipulation_type,
                }

                is_fake_nonempty = gt_label == 1 and gt_mask.sum() > 0
                for threshold in THRESHOLDS:
                    suffix = f"{int(threshold * 10):02d}"
                    tp, tn, fp, fn = counts_for_threshold(pred_prob, gt_mask, threshold)
                    iou, dice = iou_dice_from_counts(tp, fp, fn) if is_fake_nonempty else (None, None)
                    pbca = pbca_from_counts(tp, tn, fp, fn)
                    row[f"iou_at_{suffix}"] = fmt(iou)
                    row[f"dice_at_{suffix}"] = fmt(dice)
                    row[f"pbca_at_{suffix}"] = fmt(pbca)
                    row[f"counts_at_{suffix}"] = (tp, tn, fp, fn)

                rows.append(row)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    per_image_rows = []
    for row in rows:
        per_image_rows.append(
            {
                "image_path": row["image_path"],
                "gt_label": row["gt_label"],
                "fake_probability": row["fake_probability"],
                "manipulation_type": row["manipulation_type"],
                "iou_at_01": row["iou_at_01"],
                "dice_at_01": row["dice_at_01"],
                "pbca_at_01": row["pbca_at_01"],
                "iou_at_05": row["iou_at_05"],
                "dice_at_05": row["dice_at_05"],
                "pbca_at_05": row["pbca_at_05"],
            }
        )

    per_image_path = output_dir / "per_image_results.csv"
    with per_image_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image_path",
                "gt_label",
                "fake_probability",
                "manipulation_type",
                "iou_at_01",
                "dice_at_01",
                "pbca_at_01",
                "iou_at_05",
                "dice_at_05",
                "pbca_at_05",
            ],
        )
        writer.writeheader()
        writer.writerows(per_image_rows)

    overall_auc = safe_auc(auc_labels, auc_scores)
    comparison_rows = []
    for threshold in THRESHOLDS:
        summary = summarize_rows(rows, threshold)
        comparison_rows.append(
            {
                "Method": args.method_name,
                "Dataset": args.dataset_name,
                "Threshold": threshold,
                "Image AUC": fmt(overall_auc),
                "IoU": fmt(summary["iou"]),
                "Dice": fmt(summary["dice"]),
                "PBCA": fmt(summary["pbca"]),
            }
        )

    comparison_path = output_dir / "ffpp_comparison.csv"
    with comparison_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["Method", "Dataset", "Threshold", "Image AUC", "IoU", "Dice", "PBCA"],
        )
        writer.writeheader()
        writer.writerows(comparison_rows)

    auc_rows = [
        {
            "Method": args.method_name,
            "DF AUC": fmt(safe_auc(group_auc["DF"]["labels"], group_auc["DF"]["scores"])),
            "F2F AUC": fmt(safe_auc(group_auc["F2F"]["labels"], group_auc["F2F"]["scores"])),
            "FS AUC": fmt(safe_auc(group_auc["FS"]["labels"], group_auc["FS"]["scores"])),
            "NT AUC": fmt(safe_auc(group_auc["NT"]["labels"], group_auc["NT"]["scores"])),
            "Overall AUC": fmt(overall_auc),
        }
    ]
    auc_path = output_dir / "ffpp_auc_by_type.csv"
    with auc_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["Method", "DF AUC", "F2F AUC", "FS AUC", "NT AUC", "Overall AUC"],
        )
        writer.writeheader()
        writer.writerows(auc_rows)

    real_count = sum(1 for row in rows if int(row["gt_label"]) == 0)
    fake_count = sum(1 for row in rows if int(row["gt_label"]) == 1)
    method_counts = {}
    for row in rows:
        key = row["manipulation_type"] or "unknown"
        method_counts[key] = method_counts.get(key, 0) + 1

    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Total samples: {len(rows)}")
    print(f"Real samples: {real_count}")
    print(f"Fake samples: {fake_count}")
    print(f"Manipulation counts: {method_counts}")
    print(f"Classification head available: {cls_available}")
    if not cls_available:
        print("Image-level AUC is blank because LFLOC does not output cls_logits.")
    print(f"Saved comparison CSV: {comparison_path}")
    print(f"Saved per-image CSV: {per_image_path}")
    print(f"Saved AUC-by-type CSV: {auc_path}")
    print("\nAUC by manipulation type:")
    for row in auc_rows:
        print(row)
    print("\nComparison metrics:")
    for row in comparison_rows:
        print(row)


if __name__ == "__main__":
    main()
