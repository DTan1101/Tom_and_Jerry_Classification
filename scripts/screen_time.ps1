param(
    [string]$GroundTruth = "data/Tom_and_Jerry_Dataset/ground_truth.csv"
)

python -m src.screen_time --ground-truth $GroundTruth
