import argparse

import torch
from PIL import Image
from torchvision import transforms

from src.model import build_model


def main(image_path: str, checkpoint: str, top_k: int = 3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(checkpoint, map_location=device)
    classes = ckpt["classes"]
    image_size = ckpt.get("image_size", 224)

    model = build_model(ckpt["model_name"], num_classes=len(classes), pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    tfms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image = Image.open(image_path).convert("RGB")
    tensor = tfms(image).unsqueeze(0).to(device)

    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0]

    values, indices = torch.topk(probs, k=min(top_k, len(classes)))
    for score, idx in zip(values.tolist(), indices.tolist()):
        print(f"{classes[idx]}: {score:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to image file")
    parser.add_argument("--checkpoint", default="outputs/checkpoints/best.pt", help="Checkpoint path")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    main(args.image, args.checkpoint, args.top_k)
