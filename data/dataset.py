"""
Dataset module for LF-Loc.

The default FakeDataset is kept for smoke tests. FFPPDataset reads extracted
FaceForensics++ frame/mask pairs from the local data folder.
"""
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


DEFAULT_FFPP_METHODS = ("Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures")


class FakeDataset(Dataset):
    """
    Placeholder dataset that generates random image-mask pairs.

    Output contract:
        image: [3, H, W], float tensor
        mask:  [1, H, W], binary float tensor
    """

    def __init__(self, size=1000, img_size=224):
        self.size = size
        self.img_size = img_size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        image = torch.randn(3, self.img_size, self.img_size)
        mask = (torch.rand(1, self.img_size, self.img_size) > 0.7).float()
        return {
            "image": image,
            "mask": mask,
            "label": torch.tensor([1.0], dtype=torch.float32),
        }


class FFPPDataset(Dataset):
    """
    FaceForensics++ frame-level localization dataset.

    Expected directory layout:
        data_root/
          train/
            manipulated_sequences/<method>/<compression>/frames/<video>/<frame>.png
            manipulated_sequences/<method>/<compression>/masks/<video>/<frame>.png
            original_sequences/<source>/<compression>/frames/<video>/<frame>.png
          test/
            ...

    For manipulated samples, masks are loaded from the matching mask path.
    For optional original samples, a zero mask is generated.
    """

    def __init__(
        self,
        data_root="data/FaceForensics++/FaceForensics++",
        split="train",
        img_size=224,
        compression="c23",
        methods=DEFAULT_FFPP_METHODS,
        include_originals=False,
        max_samples=None,
    ):
        self.data_root = Path(data_root)
        self.split = split
        self.img_size = img_size
        self.compression = compression
        self.methods = tuple(methods)
        self.include_originals = include_originals

        if not self.data_root.exists():
            raise FileNotFoundError(f"FF++ data root not found: {self.data_root}")

        scan_split = "test" if split in {"val", "test"} else "train"
        self.split_dir = self.data_root / scan_split
        if not self.split_dir.exists():
            raise FileNotFoundError(f"FF++ split directory not found: {self.split_dir}")

        self.samples = self._collect_samples()
        if max_samples is not None and max_samples > 0:
            self.samples = self.samples[:max_samples]
        if not self.samples:
            raise RuntimeError(
                f"No FF++ samples found under {self.split_dir} "
                f"for methods={self.methods}, compression={self.compression}"
            )

    def _collect_samples(self):
        samples = []
        manipulated_root = self.split_dir / "manipulated_sequences"

        for method in self.methods:
            method_root = manipulated_root / method / self.compression
            frames_root = method_root / "frames"
            masks_root = method_root / "masks"
            if not frames_root.exists() or not masks_root.exists():
                continue

            for video_dir in sorted(p for p in frames_root.iterdir() if p.is_dir()):
                mask_video_dir = masks_root / video_dir.name
                if not mask_video_dir.exists():
                    continue
                for image_path in sorted(video_dir.glob("*.png")):
                    mask_path = mask_video_dir / image_path.name
                    if mask_path.exists():
                        samples.append(
                            {
                                "image": image_path,
                                "mask": mask_path,
                                "label": 1.0,
                                "method": method,
                                "video": video_dir.name,
                            }
                        )

        if self.include_originals:
            original_root = self.split_dir / "original_sequences"
            if original_root.exists():
                for source_dir in sorted(p for p in original_root.iterdir() if p.is_dir()):
                    frames_root = source_dir / self.compression / "frames"
                    if not frames_root.exists():
                        continue
                    for video_dir in sorted(p for p in frames_root.iterdir() if p.is_dir()):
                        for image_path in sorted(video_dir.glob("*.png")):
                            samples.append(
                                {
                                    "image": image_path,
                                    "mask": None,
                                    "label": 0.0,
                                    "method": "original",
                                    "video": video_dir.name,
                                }
                            )

        return samples

    def __len__(self):
        return len(self.samples)

    def _load_image(self, path):
        image = Image.open(path).convert("RGB")
        image = image.resize((self.img_size, self.img_size), Image.BILINEAR)
        array = np.asarray(image, dtype=np.float32) / 255.0
        return torch.from_numpy(array).permute(2, 0, 1)

    def _load_mask(self, path):
        if path is None:
            return torch.zeros(1, self.img_size, self.img_size, dtype=torch.float32)
        mask = Image.open(path).convert("L")
        mask = mask.resize((self.img_size, self.img_size), Image.NEAREST)
        array = (np.asarray(mask, dtype=np.float32) > 127.0).astype(np.float32)
        return torch.from_numpy(array).unsqueeze(0)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            "image": self._load_image(sample["image"]),
            "mask": self._load_mask(sample["mask"]),
            "label": torch.tensor([sample["label"]], dtype=torch.float32),
            "image_path": str(sample["image"]),
        }


def build_dataloader(
    batch_size=8,
    num_workers=0,
    img_size=224,
    dataset_size=1000,
    pin_memory=False,
    dataset_type="fake",
    data_root="data/FaceForensics++/FaceForensics++",
    split="train",
    compression="c23",
    methods=DEFAULT_FFPP_METHODS,
    include_originals=False,
    shuffle=None,
):
    """Build a DataLoader for fake data or FaceForensics++."""
    if dataset_type == "fake":
        if dataset_size is None or dataset_size <= 0:
            dataset_size = 1000
        dataset = FakeDataset(size=dataset_size, img_size=img_size)
    elif dataset_type == "ffpp":
        dataset = FFPPDataset(
            data_root=data_root,
            split=split,
            img_size=img_size,
            compression=compression,
            methods=methods,
            include_originals=include_originals,
            max_samples=dataset_size,
        )
    else:
        raise ValueError(f"Unsupported dataset_type: {dataset_type}")

    if shuffle is None:
        shuffle = split == "train"

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


if __name__ == "__main__":
    fake_loader = build_dataloader(batch_size=4, num_workers=0, dataset_size=20)
    batch = next(iter(fake_loader))
    print(f"Fake image batch: {batch['image'].shape}")
    print(f"Fake mask batch:  {batch['mask'].shape}")

    default_root = Path("data/FaceForensics++/FaceForensics++")
    if default_root.exists():
        ffpp_loader = build_dataloader(
            dataset_type="ffpp",
            data_root=str(default_root),
            split="train",
            batch_size=2,
            num_workers=0,
            dataset_size=4,
        )
        batch = next(iter(ffpp_loader))
        print(f"FF++ image batch: {batch['image'].shape}")
        print(f"FF++ mask batch:  {batch['mask'].shape}")
        print(f"FF++ label batch: {batch['label'].shape}")

    print("DataLoader test passed.")
