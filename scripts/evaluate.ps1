param(
    [string]$DataDir = "data/Tom_and_Jerry_Dataset/tom_and_jerry/tom_and_jerry",
    [string]$Checkpoint = "outputs/checkpoints/best.pt"
)

python -m src.evaluate --data-dir $DataDir --checkpoint $Checkpoint
