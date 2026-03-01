from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import transforms

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class FolderClassificationDataset(Dataset):
    def __init__(self, root: Path, classes: Sequence[str], transform=None):
        self.root = Path(root)
        self.classes = list(classes)
        self.class_to_idx = {name: i for i, name in enumerate(self.classes)}
        self.transform = transform
        self.samples = []

        for class_name in self.classes:
            class_dir = self.root / class_name
            if not class_dir.exists():
                continue
            for file_path in sorted(class_dir.rglob("*")):
                if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS:
                    self.samples.append((file_path, self.class_to_idx[class_name]))

        if not self.samples:
            raise ValueError(
                f"No images found under {self.root} for classes {self.classes}. "
                "Please verify dataset path and folder names."
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        image_path, label = self.samples[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def build_transforms(image_size: int) -> Tuple[transforms.Compose, transforms.Compose]:
    train_tfms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    eval_tfms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_tfms, eval_tfms


def _resolve_data_root(root: Path) -> Path:
    if (root / "train").exists() or (root / "val").exists():
        return root

    tom_jerry_nested = root / "Tom_and_Jerry_Dataset" / "tom_and_jerry" / "tom_and_jerry"
    if tom_jerry_nested.exists():
        return tom_jerry_nested

    return root


def _discover_classes(root: Path, expected_classes: Optional[Sequence[str]] = None) -> List[str]:
    subdirs = [d.name for d in root.iterdir() if d.is_dir()]
    if expected_classes:
        filtered = [name for name in expected_classes if (root / name).exists()]
        if filtered:
            return filtered
    return sorted(subdirs)


def _train_val_from_single_dir(
    base_dir: Path,
    classes: Sequence[str],
    train_tfms: transforms.Compose,
    eval_tfms: transforms.Compose,
    val_split: float,
    seed: int,
):
    full_dataset = FolderClassificationDataset(base_dir, classes=classes, transform=train_tfms)
    val_size = max(1, int(len(full_dataset) * val_split))
    train_size = len(full_dataset) - val_size
    if train_size < 1:
        raise ValueError("Dataset is too small after split. Reduce val_split or add more training images.")

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size], generator=generator)

    # Validation subset should use eval transforms.
    eval_dataset = FolderClassificationDataset(base_dir, classes=classes, transform=eval_tfms)
    val_ds = Subset(eval_dataset, val_ds.indices)
    return train_ds, val_ds, list(classes)


def create_dataloaders(
    data_dir: str,
    image_size: int,
    batch_size: int,
    num_workers: int,
    val_split: float,
    seed: int,
    expected_classes: Optional[Sequence[str]] = None,
) -> Tuple[Dict[str, DataLoader], list]:
    root = _resolve_data_root(Path(data_dir))
    train_tfms, eval_tfms = build_transforms(image_size)

    train_dir = root / "train"
    val_dir = root / "val"

    if train_dir.exists() and val_dir.exists() and any(train_dir.iterdir()) and any(val_dir.iterdir()):
        classes = _discover_classes(train_dir, expected_classes)
        train_ds = FolderClassificationDataset(train_dir, classes=classes, transform=train_tfms)
        val_ds = FolderClassificationDataset(val_dir, classes=classes, transform=eval_tfms)
    else:
        classes = _discover_classes(root, expected_classes)
        train_ds, val_ds, classes = _train_val_from_single_dir(
            root,
            classes=classes,
            train_tfms=train_tfms,
            eval_tfms=eval_tfms,
            val_split=val_split,
            seed=seed,
        )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return {"train": train_loader, "val": val_loader}, classes


def create_eval_loader(
    data_dir: str,
    image_size: int,
    batch_size: int,
    num_workers: int,
    classes: Sequence[str],
) -> DataLoader:
    root = _resolve_data_root(Path(data_dir))
    _, eval_tfms = build_transforms(image_size)
    dataset = FolderClassificationDataset(root, classes=classes, transform=eval_tfms)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
