import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import torch
import pandas as pd

from src.config import *
from src.preprocessing import load_data, prepare_features, make_splits, make_tensors, make_multimodal_loaders, make_unimodal_loaders
from src.model import AttentionFusionModel, UnimodalClassifier, ConcatenationModel
from src.training import evaluate
from src.evaluation import get_classification_metrics, print_classification_report, get_confusion_matrix, evaluate_unimodal_model, evaluate_missing_modality
from src.experiments import evaluate_concat_model, build_results_dataframe


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = load_data()
    _, X_text, X_audio, X_video, y, label_encoder = prepare_features(df)
    train_idx, val_idx, test_idx = make_splits(y)
    text_tensor, audio_tensor, video_tensor, label_tensor = make_tensors(X_text, X_audio, X_video, y)
    train_loader, val_loader, test_loader = make_multimodal_loaders(text_tensor, audio_tensor, video_tensor, label_tensor, train_idx, val_idx, test_idx)

    criterion = torch.nn.CrossEntropyLoss()
    attention_model = AttentionFusionModel(INPUT_DIM, EMBEDDING_DIM, NUM_HEADS, len(label_encoder.classes_)).to(device)
    attention_model.load_state_dict(torch.load(MODEL_DIR / "best_attention_model.pth", map_location=device))
    _, test_acc, y_true, y_pred = evaluate(attention_model, test_loader, criterion, device)
    attention_metrics = get_classification_metrics(y_true, y_pred)
    print("Attention Fusion Metrics:", attention_metrics)
    print_classification_report(y_true, y_pred, label_encoder.classes_)
    print("Confusion Matrix:\n", get_confusion_matrix(y_true, y_pred))

    # Unimodal models
    text_train, text_val, text_test = make_unimodal_loaders(text_tensor, label_tensor, train_idx, val_idx, test_idx)
    audio_train, audio_val, audio_test = make_unimodal_loaders(audio_tensor, label_tensor, train_idx, val_idx, test_idx)
    video_train, video_val, video_test = make_unimodal_loaders(video_tensor, label_tensor, train_idx, val_idx, test_idx)

    text_model = UnimodalClassifier(INPUT_DIM, len(label_encoder.classes_)).to(device)
    audio_model = UnimodalClassifier(INPUT_DIM, len(label_encoder.classes_)).to(device)
    video_model = UnimodalClassifier(INPUT_DIM, len(label_encoder.classes_)).to(device)
    text_model.load_state_dict(torch.load(MODEL_DIR / "text_model.pth", map_location=device))
    audio_model.load_state_dict(torch.load(MODEL_DIR / "audio_model.pth", map_location=device))
    video_model.load_state_dict(torch.load(MODEL_DIR / "video_model.pth", map_location=device))

    text_metrics = evaluate_unimodal_model(text_model, text_test, device)
    audio_metrics = evaluate_unimodal_model(audio_model, audio_test, device)
    video_metrics = evaluate_unimodal_model(video_model, video_test, device)

    concat_model = ConcatenationModel(INPUT_DIM, len(label_encoder.classes_)).to(device)
    concat_model.load_state_dict(torch.load(MODEL_DIR / "best_concat_model.pth", map_location=device))
    concat_metrics = evaluate_concat_model(concat_model, test_loader, device)

    results = build_results_dataframe(text_metrics, audio_metrics, video_metrics, concat_metrics, attention_metrics)
    print("\nModel Comparison:\n", results)
    results.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)

    # Ablation
    rows = []
    for name, missing in [("Full Model", None), ("Without Text", "text"), ("Without Audio", "audio"), ("Without Video", "video")]:
        acc, f1 = evaluate_missing_modality(attention_model, test_loader, device, missing)
        rows.append({"Experiment": name, "Accuracy": acc, "Macro-F1": f1})
    ablation = pd.DataFrame(rows)
    print("\nAblation Study:\n", ablation)
    ablation.to_csv(OUTPUT_DIR / "ablation_results.csv", index=False)


if __name__ == "__main__":
    main()
