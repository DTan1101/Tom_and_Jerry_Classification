param(
    [string]$Config = "configs/classical.yaml"
)

python -m src.classical_train --config $Config
