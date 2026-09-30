import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)


def evaluate_unimodal_model(model, loader, device):
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for X, labels in loader:
            X = X.to(device)
            outputs = model(X)
            predictions = torch.argmax(outputs, dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.numpy())

    return {
        "accuracy": accuracy_score(all_labels, all_predictions),
        "precision": precision_score(all_labels, all_predictions, average="macro", zero_division=0),
        "recall": recall_score(all_labels, all_predictions, average="macro", zero_division=0),
        "f1": f1_score(all_labels, all_predictions, average="macro", zero_division=0),
    }


def evaluate_missing_modality(model, loader, device, missing_modality=None):
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for text, audio, video, labels in loader:
            text = text.to(device)
            audio = audio.to(device)
            video = video.to(device)

            if missing_modality == "text":
                text = torch.zeros_like(text)
            elif missing_modality == "audio":
                audio = torch.zeros_like(audio)
            elif missing_modality == "video":
                video = torch.zeros_like(video)

            outputs, _ = model(text, audio, video)
            predictions = torch.argmax(outputs, dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.numpy())

    return (
        accuracy_score(all_labels, all_predictions),
        f1_score(all_labels, all_predictions, average="macro", zero_division=0),
    )


def get_classification_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def print_classification_report(y_true, y_pred, class_names):
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))


def get_confusion_matrix(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)
