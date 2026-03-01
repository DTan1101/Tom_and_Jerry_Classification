import argparse
from pathlib import Path

import joblib
import pandas as pd
import yaml
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.classical.features import batch_extract
from src.classical.models import build_estimator


def evaluate_split(name: str, pipeline, x, y_true):
    y_pred = pipeline.predict(x)
    acc = accuracy_score(y_true, y_pred)
    print(f"[{name}] accuracy: {acc:.4f}")
    print(classification_report(y_true, y_pred, digits=4))


def main(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    split_df = pd.read_csv(cfg["split"]["output_csv"])
    feature_cfg = cfg["classical"]["feature"]
    model_cfg = cfg["classical"]["model"]

    train_df = split_df[split_df["split"] == "train"].reset_index(drop=True)
    val_df = split_df[split_df["split"] == "val"].reset_index(drop=True)
    test_df = split_df[split_df["split"] == "test"].reset_index(drop=True)

    image_size = feature_cfg["image_size"]
    method = feature_cfg["name"]

    x_train = batch_extract(train_df["path"].tolist(), method=method, image_size=image_size)
    x_val = batch_extract(val_df["path"].tolist(), method=method, image_size=image_size)
    x_test = batch_extract(test_df["path"].tolist(), method=method, image_size=image_size)

    encoder = LabelEncoder()
    y_train = encoder.fit_transform(train_df["label"])
    y_val = encoder.transform(val_df["label"])
    y_test = encoder.transform(test_df["label"])

    steps = [("scaler", StandardScaler())]
    if model_cfg.get("use_pca", False):
        steps.append(("pca", PCA(n_components=model_cfg.get("pca_components", 128), random_state=cfg["seed"])))
    steps.append(("estimator", build_estimator(model_cfg["name"], seed=cfg["seed"])))

    pipeline = Pipeline(steps)
    pipeline.fit(x_train, y_train)

    evaluate_split("val", pipeline, x_val, y_val)
    evaluate_split("test", pipeline, x_test, y_test)

    out_path = Path(cfg["classical"]["output_model_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "pipeline": pipeline,
            "classes": encoder.classes_.tolist(),
            "feature_method": method,
            "image_size": image_size,
        },
        out_path,
    )
    print(f"Saved model: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/classical.yaml")
    args = parser.parse_args()
    main(args.config)
