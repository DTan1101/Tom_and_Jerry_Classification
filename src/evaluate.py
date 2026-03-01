import argparse

import torch
from torch import nn
from tqdm import tqdm

from src.data import create_eval_loader
from src.model import build_model


def main(data_dir: str, checkpoint: str, batch_size: int = 32, num_workers: int = 2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(checkpoint, map_location=device)
    classes = ckpt["classes"]
    image_size = ckpt.get("image_size", 224)

    loader = create_eval_loader(
        data_dir=data_dir,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
        classes=classes,
    )

    model = build_model(ckpt["model_name"], num_classes=len(classes), pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    criterion = nn.CrossEntropyLoss()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in tqdm(loader):
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            total_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    print(f"Eval loss: {total_loss / total:.4f}")
    print(f"Eval acc : {correct / total:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        default="data/Tom_and_Jerry_Dataset/tom_and_jerry/tom_and_jerry",
        help="Path to evaluation dataset root",
    )
    parser.add_argument("--checkpoint", default="outputs/checkpoints/best.pt", help="Checkpoint path")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()

    main(args.data_dir, args.checkpoint, args.batch_size, args.num_workers)
