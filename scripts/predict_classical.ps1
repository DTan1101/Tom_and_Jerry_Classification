param(
    [Parameter(Mandatory=$true)][string]$Image,
    [string]$Model = "outputs/models/classical_model.joblib"
)

python -m src.classical_predict --image $Image --model $Model
