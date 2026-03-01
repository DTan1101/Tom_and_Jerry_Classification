param(
    [string]$Config = "configs/classical.yaml"
)

python -m src.build_split --config $Config
