from pathlib import Path
from typing import List, Optional, Sequence

import pandas as pd
from sklearn.model_selection import train_test_split

from src.data import IMAGE_EXTENSIONS


def discover_image_root(data_dir: str) -> Path:
    root = Path(data_dir)
    if (root / "tom").exists() and (root / "jerry").exists():
        return root

    candidates = [
        root / "tom_and_jerry" / "tom_and_jerry",
        root / "Tom_and_Jerry_Dataset" / "tom_and_jerry" / "tom_and_jerry",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return root


def build_index(data_dir: str, classes: Optional[Sequence[str]] = None) -> pd.DataFrame:
    root = discover_image_root(data_dir)
    if classes:
        target_classes = list(classes)
    else:
        target_classes = sorted([d.name for d in root.iterdir() if d.is_dir()])

    records = []
    for class_name in target_classes:
        class_dir = root / class_name
        if not class_dir.exists():
            continue
        for path in sorted(class_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                records.append({"path": str(path), "label": class_name})

    if not records:
        raise ValueError("No images were found. Please verify data_dir and class names.")

    return pd.DataFrame.from_records(records)


def stratified_split(
    df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> pd.DataFrame:
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-8:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    train_df, temp_df = train_test_split(
        df,
        test_size=(1.0 - train_ratio),
        random_state=seed,
        stratify=df["label"],
    )

    val_portion = val_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1.0 - val_portion),
        random_state=seed,
        stratify=temp_df["label"],
    )

    train_df = train_df.assign(split="train")
    val_df = val_df.assign(split="val")
    test_df = test_df.assign(split="test")

    return pd.concat([train_df, val_df, test_df], ignore_index=True)


def write_split_csv(df: pd.DataFrame, output_csv: str):
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
