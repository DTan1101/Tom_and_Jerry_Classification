# Tom and Jerry Classification - Extensible Project Design

This project supports both classical machine learning and deep learning for the Tom and Jerry dataset.

## 1) Dataset summary

- Total images: 5478 (1 FPS frames)
- Original labeled folders:
  - `tom`: images containing only Tom
  - `jerry`: images containing only Jerry
  - `tom_jerry_1`: images containing both Tom and Jerry
  - `tom_jerry_0`: images containing neither
- Metadata:
  - `ground_truth.csv`: frame-level labels (`tom`, `jerry`)
  - `challenges.csv`: difficult/distorted samples for error analysis

Default binary classification setup in this repository uses only:
- `tom`
- `jerry`

## 2) Default dataset path

```text
data/Tom_and_Jerry_Dataset/tom_and_jerry/tom_and_jerry/
```

The loader auto-resolves this nested path from `data/Tom_and_Jerry_Dataset`.

## 3) Project layout

```text
.
├── configs/
│   ├── deep.yaml
│   ├── classical.yaml
│   └── default.yaml
├── scripts/
│   ├── train.ps1
│   ├── evaluate.ps1
│   ├── predict.ps1
│   ├── build_split.ps1
│   ├── train_classical.ps1
│   ├── predict_classical.ps1
│   └── screen_time.ps1
├── src/
│   ├── data.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   ├── splits.py
│   ├── build_split.py
│   ├── classical_train.py
│   ├── classical_predict.py
│   ├── screen_time.py
│   └── classical/
│       ├── features.py
│       └── models.py
└── outputs/
```

## 4) Install

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 5) Deep learning workflow (CNN)

Train:

```bash
python -m src.train --config configs/deep.yaml
```

Evaluate:

```bash
python -m src.evaluate --data-dir data/Tom_and_Jerry_Dataset/tom_and_jerry/tom_and_jerry --checkpoint outputs/checkpoints/best.pt
```

Predict single image:

```bash
python -m src.predict --image path/to/image.jpg --checkpoint outputs/checkpoints/best.pt --top-k 3
```

## 6) Classical ML workflow (HOG/SIFT + SVM/RF/NB)

Build stratified train/val/test split:

```bash
python -m src.build_split --config configs/classical.yaml
```

Train + evaluate:

```bash
python -m src.classical_train --config configs/classical.yaml
```

Predict single image:

```bash
python -m src.classical_predict --image path/to/image.jpg --model outputs/models/classical_model.joblib
```

## 7) Screen-time analysis from ground truth

```bash
python -m src.screen_time --ground-truth data/Tom_and_Jerry_Dataset/ground_truth.csv
```

This prints total seconds in 4 states:
- `tom_only`
- `jerry_only`
- `tom_jerry_1`
- `tom_jerry_0`

Also prints the longest continuous streak where both Tom and Jerry appear together.

## 8) Config knobs for extension

- `configs/classical.yaml`
  - `feature.name`: `raw | hog | sift`
  - `model.name`: `svm | rf | nb`
  - `use_pca`, `pca_components`
  - split ratios: train/val/test
- `configs/deep.yaml`
  - model backbone, image size, epochs, batch size, learning rate

## 9) Suggested extension direction

1. Add challenge-set evaluation by filtering images listed in `challenges.csv`.
2. Add 4-class mode using folders `tom`, `jerry`, `tom_jerry_1`, `tom_jerry_0`.
3. Add experiment tracking and confusion-matrix exports under `outputs/runs/`.
