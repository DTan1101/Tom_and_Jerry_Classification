param(
    [string]$Config = "configs/default.yaml"
)

python -m src.train --config $Config
