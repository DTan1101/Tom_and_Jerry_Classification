import argparse
import re

import pandas as pd


FRAME_RE = re.compile(r"frame(\d+)\.jpg")


def frame_to_second(filename: str) -> int:
    match = FRAME_RE.fullmatch(filename)
    if not match:
        return -1
    return int(match.group(1))


def label_state(row) -> str:
    tom = int(row["tom"])
    jerry = int(row["jerry"])
    if tom == 1 and jerry == 1:
        return "tom_jerry_1"
    if tom == 1 and jerry == 0:
        return "tom_only"
    if tom == 0 and jerry == 1:
        return "jerry_only"
    return "tom_jerry_0"


def longest_both_streak_seconds(df: pd.DataFrame) -> int:
    both = df[df["state"] == "tom_jerry_1"].sort_values("second")
    if both.empty:
        return 0

    longest = 1
    current = 1
    seconds = both["second"].tolist()
    for i in range(1, len(seconds)):
        if seconds[i] == seconds[i - 1] + 1:
            current += 1
            longest = max(longest, current)
        else:
            current = 1
    return longest


def main(ground_truth_csv: str):
    df = pd.read_csv(ground_truth_csv)
    df["second"] = df["filename"].apply(frame_to_second)
    df = df[df["second"] >= 0].copy()
    df["state"] = df.apply(label_state, axis=1)

    counts = df["state"].value_counts().to_dict()
    longest = longest_both_streak_seconds(df)

    print("Screen time summary (seconds):")
    print(f"tom_only   : {counts.get('tom_only', 0)}")
    print(f"jerry_only : {counts.get('jerry_only', 0)}")
    print(f"tom_jerry_1: {counts.get('tom_jerry_1', 0)}")
    print(f"tom_jerry_0: {counts.get('tom_jerry_0', 0)}")
    print(f"Longest continuous tom_jerry_1 streak: {longest} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ground-truth", default="data/Tom_and_Jerry_Dataset/ground_truth.csv")
    args = parser.parse_args()
    main(args.ground_truth)
