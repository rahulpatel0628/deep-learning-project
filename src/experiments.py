import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch import nn, optim

from model import AttentionFusionModel, UnimodalClassifier, ConcatenationModel
from training import train_unimodal_model
from evaluation import evaluate_unimodal_model, evaluate_missing_modality


def train_concatenation_model(model, train_loader, val_loader, device, epochs=10, lr=0.001, save_path=None):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_val_acc = 0.0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for text, audio, video, labels in train_loader:
            text, audio, video, labels = (
                text.to(device), audio.to(device), video.to(device), labels.to(device)
            )
            optimizer.zero_grad()
            outputs = model(text, audio, video)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            predictions = torch.argmax(outputs, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total

        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for text, audio, video, labels in val_loader:
                text, audio, video, labels = (
                    text.to(device), audio.to(device), video.to(device), labels.to(device)
                )
                outputs = model(text, audio, video)
                predictions = torch.argmax(outputs, dim=1)
                val_correct += (predictions == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total
        print(f"Epoch [{epoch + 1}/{epochs}] Train Acc: {train_acc:.4f} Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            if save_path:
                torch.save(model.state_dict(), save_path)

    return best_val_acc


def evaluate_concat_model(model, test_loader, device):
    model.eval()
    all_predictions, all_labels = [], []
    with torch.no_grad():
        for text, audio, video, labels in test_loader:
            outputs = model(text.to(device), audio.to(device), video.to(device))
            predictions = torch.argmax(outputs, dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.numpy())

    return {
        "accuracy": accuracy_score(all_labels, all_predictions),
        "precision": precision_score(all_labels, all_predictions, average="macro", zero_division=0),
        "recall": recall_score(all_labels, all_predictions, average="macro", zero_division=0),
        "f1": f1_score(all_labels, all_predictions, average="macro", zero_division=0),
    }


def build_results_dataframe(text_metrics, audio_metrics, video_metrics, concat_metrics, attention_metrics):
    return pd.DataFrame({
        "Model": ["Text Only", "Audio Only", "Video Only", "Concatenation", "Attention Fusion"],
        "Accuracy": [text_metrics["accuracy"], audio_metrics["accuracy"], video_metrics["accuracy"], concat_metrics["accuracy"], attention_metrics["accuracy"]],
        "Precision": [text_metrics["precision"], audio_metrics["precision"], video_metrics["precision"], concat_metrics["precision"], attention_metrics["precision"]],
        "Recall": [text_metrics["recall"], audio_metrics["recall"], video_metrics["recall"], concat_metrics["recall"], attention_metrics["recall"]],
        "Macro-F1": [text_metrics["f1"], audio_metrics["f1"], video_metrics["f1"], concat_metrics["f1"], attention_metrics["f1"]],
    })
