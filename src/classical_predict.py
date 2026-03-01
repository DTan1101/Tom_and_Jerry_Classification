import argparse

import joblib

from src.classical.features import extract_feature


def main(image_path: str, model_path: str):
    artifact = joblib.load(model_path)
    pipeline = artifact["pipeline"]
    classes = artifact["classes"]
    feature_method = artifact["feature_method"]
    image_size = artifact["image_size"]

    x = extract_feature(image_path, method=feature_method, image_size=image_size).reshape(1, -1)
    probs = pipeline.predict_proba(x)[0]

    pairs = sorted(zip(classes, probs), key=lambda z: z[1], reverse=True)
    for cls, p in pairs:
        print(f"{cls}: {p:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--model", default="outputs/models/classical_model.joblib")
    args = parser.parse_args()
    main(args.image, args.model)
