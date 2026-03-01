import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.optim import AdamW
from tqdm import tqdm

from src.data import create_dataloaders
from src.model import build_model


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_epoch(model, loader, criterion, optimizer, device, is_train: bool):
    model.train(is_train)
    epoch_loss, correct, total = 0.0, 0, 0

    loop = tqdm(loader, leave=False)
    for images, labels in loop:
        images = images.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(is_train):
            logits = model(images)
            loss = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        epoch_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return epoch_loss / total, correct / total


def main(config_path: str):
    cfg = load_config(config_path)
    set_seed(cfg["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loaders, classes = create_dataloaders(
        data_dir=cfg["data"]["data_dir"],
        image_size=cfg["data"]["image_size"],
        batch_size=cfg["training"]["batch_size"],
        num_workers=cfg["training"]["num_workers"],
        val_split=cfg["data"]["val_split"],
        seed=cfg["seed"],
        expected_classes=cfg["data"].get("classes"),
    )

    print(f"Detected classes: {classes}")

    model = build_model(
        model_name=cfg["model"]["name"],
        num_classes=len(classes),
        pretrained=cfg["model"]["pretrained"],
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    checkpoint_dir = Path(cfg["training"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_val_acc = 0.0
    for epoch in range(cfg["training"]["epochs"]):
        train_loss, train_acc = run_epoch(
            model, loaders["train"], criterion, optimizer, device, is_train=True
        )
        val_loss, val_acc = run_epoch(
            model, loaders["val"], criterion, optimizer, device, is_train=False
        )

        print(
            f"Epoch {epoch + 1}/{cfg['training']['epochs']} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt_path = checkpoint_dir / "best.pt"
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "classes": classes,
                    "model_name": cfg["model"]["name"],
                    "image_size": cfg["data"]["image_size"],
                },
                ckpt_path,
            )
            print(f"Saved best checkpoint: {ckpt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml", help="Path to config file")
    args = parser.parse_args()

    os.environ.setdefault("PYTHONPATH", ".")
    main(args.config)
