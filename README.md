# Attention-Based Multimodal Emotion Recognition

This project is the separated, runnable version of the uploaded `Multimodal_Emotion_Recognition.ipynb` notebook.

## Current implementation

The notebook uses a processed MELD CSV with 7-dimensional modality-level prediction vectors for:

- Text
- Audio
- Video

The project implements:

1. Attention Fusion model
2. Text-only baseline
3. Audio-only baseline
4. Video-only baseline
5. Concatenation baseline
6. Attention visualization
7. Confusion matrix
8. Missing-modality ablation study

> Important: this version does **not** train raw audio/video/text encoders. It fuses the pre-extracted 7-D modality prediction vectors contained in the CSV.

## Folder structure

```text
multimodal_emotion_project/
├── train.py
├── evaluate.py
├── requirements.txt
├── README.md
├── data/
│   └── MELD_dataset_with_emotions.csv
├── models/
├── outputs/
└── src/
    ├── config.py
    ├── preprocessing.py
    ├── model.py
    ├── training.py
    ├── evaluation.py
    ├── experiments.py
    └── visualization.py
```

## Setup

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Train

From the project root:

```bash
python train.py
```

This creates model files inside `models/`.

## Evaluate

```bash
python evaluate.py
```

Results are saved to `outputs/`.

## Next step: application

The current model accepts the three 7-D modality vectors. To build a user-facing Streamlit demo that accepts raw text/audio/video, separate encoders or a feature-extraction layer must be added first. That is a future upgrade of this project.
