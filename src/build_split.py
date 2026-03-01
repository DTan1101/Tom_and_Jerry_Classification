import argparse

import yaml

from src.splits import build_index, stratified_split, write_split_csv


def main(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    split_cfg = cfg["split"]
    data_cfg = cfg["data"]

    index_df = build_index(data_cfg["data_dir"], classes=data_cfg.get("classes"))
    split_df = stratified_split(
        index_df,
        train_ratio=split_cfg["train_ratio"],
        val_ratio=split_cfg["val_ratio"],
        test_ratio=split_cfg["test_ratio"],
        seed=cfg["seed"],
    )
    write_split_csv(split_df, split_cfg["output_csv"])

    print(f"Wrote split CSV: {split_cfg['output_csv']}")
    print(split_df.groupby(["split", "label"]).size())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/classical.yaml")
    args = parser.parse_args()
    main(args.config)
