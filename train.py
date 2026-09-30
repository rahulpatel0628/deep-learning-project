import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import torch
import torch.nn as nn

from src.config import *
from src.preprocessing import load_data, prepare_features, make_splits, make_tensors, make_multimodal_loaders, make_unimodal_loaders
from src.model import AttentionFusionModel, UnimodalClassifier, ConcatenationModel
from src.training import train_attention_model, train_unimodal_model
from src.evaluation import evaluate_unimodal_model
from src.experiments import train_concatenation_model


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    df = load_data()
    _, X_text, X_audio, X_video, y, label_encoder = prepare_features(df)
    train_idx, val_idx, test_idx = make_splits(y)
    text_tensor, audio_tensor, video_tensor, label_tensor = make_tensors(X_text, X_audio, X_video, y)

    train_loader, val_loader, test_loader = make_multimodal_loaders(
        text_tensor, audio_tensor, video_tensor, label_tensor,
        train_idx, val_idx, test_idx
    )

    # ---------------- Attention Fusion ----------------
    model = AttentionFusionModel(INPUT_DIM, EMBEDDING_DIM, NUM_HEADS, len(label_encoder.classes_)).to(device)
    history = train_attention_model(
        model, train_loader, val_loader, device,
        epochs=EPOCHS, lr=LEARNING_RATE,
        save_path=MODEL_DIR / "best_attention_model.pth"
    )
    print("Best attention validation accuracy:", history["best_val_accuracy"])

    # ---------------- Unimodal baselines ----------------
    train_text, val_text, test_text = make_unimodal_loaders(text_tensor, label_tensor, train_idx, val_idx, test_idx)
    train_audio, val_audio, test_audio = make_unimodal_loaders(audio_tensor, label_tensor, train_idx, val_idx, test_idx)
    train_video, val_video, test_video = make_unimodal_loaders(video_tensor, label_tensor, train_idx, val_idx, test_idx)

    text_model = UnimodalClassifier(INPUT_DIM, len(label_encoder.classes_)).to(device)
    audio_model = UnimodalClassifier(INPUT_DIM, len(label_encoder.classes_)).to(device)
    video_model = UnimodalClassifier(INPUT_DIM, len(label_encoder.classes_)).to(device)

    train_unimodal_model(text_model, train_text, val_text, device, EPOCHS, LEARNING_RATE)
    train_unimodal_model(audio_model, train_audio, val_audio, device, EPOCHS, LEARNING_RATE)
    train_unimodal_model(video_model, train_video, val_video, device, EPOCHS, LEARNING_RATE)

    torch.save(text_model.state_dict(), MODEL_DIR / "text_model.pth")
    torch.save(audio_model.state_dict(), MODEL_DIR / "audio_model.pth")
    torch.save(video_model.state_dict(), MODEL_DIR / "video_model.pth")

    # ---------------- Concatenation baseline ----------------
    concat_model = ConcatenationModel(INPUT_DIM, len(label_encoder.classes_)).to(device)
    train_concatenation_model(
        concat_model, train_loader, val_loader, device,
        epochs=EPOCHS, lr=LEARNING_RATE,
        save_path=MODEL_DIR / "best_concat_model.pth"
    )

    # Save class mapping for inference
    import json
    with open(MODEL_DIR / "label_mapping.json", "w", encoding="utf-8") as f:
        json.dump({str(i): name for i, name in enumerate(label_encoder.classes_)}, f, indent=2)

    print("\nTraining complete.")
    print("Saved models to:", MODEL_DIR)


if __name__ == "__main__":
    main()
