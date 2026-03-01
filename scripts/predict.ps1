param(
    [Parameter(Mandatory=$true)][string]$Image,
    [string]$Checkpoint = "outputs/checkpoints/best.pt",
    [int]$TopK = 3
)

python -m src.predict --image $Image --checkpoint $Checkpoint --top-k $TopK
