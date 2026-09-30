import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_training_curves(history):
    plt.figure(figsize=(10, 5))
    plt.plot(history["train_losses"], label="Training Loss")
    plt.plot(history["val_losses"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(history["train_accuracies"], label="Training Accuracy")
    plt.plot(history["val_accuracies"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(cm, class_names):
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Emotion")
    plt.ylabel("Actual Emotion")
    plt.title("Confusion Matrix - Attention Fusion Model")
    plt.tight_layout()
    plt.show()


def plot_model_comparison(results):
    plt.figure(figsize=(12, 6))
    x = np.arange(len(results["Model"]))
    width = 0.2
    for offset, metric in zip([-1.5, -0.5, 0.5, 1.5], ["Accuracy", "Precision", "Recall", "Macro-F1"]):
        plt.bar(x + offset * width, results[metric], width, label=metric)
    plt.xticks(x, results["Model"], rotation=20)
    plt.ylabel("Score")
    plt.title("Comparison of Unimodal and Multimodal Models")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_attention_weights(attention_weights):
    attention_matrix = attention_weights[0].detach().cpu().numpy()
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        attention_matrix,
        annot=True,
        fmt=".3f",
        xticklabels=["Text", "Audio", "Video"],
        yticklabels=["Text", "Audio", "Video"],
    )
    plt.xlabel("Key Modality")
    plt.ylabel("Query Modality")
    plt.title("Attention Weights Between Modalities")
    plt.tight_layout()
    plt.show()


def plot_ablation(ablation_results):
    plt.figure(figsize=(10, 6))
    x = np.arange(len(ablation_results["Experiment"]))
    width = 0.35
    plt.bar(x - width / 2, ablation_results["Accuracy"], width, label="Accuracy")
    plt.bar(x + width / 2, ablation_results["Macro-F1"], width, label="Macro-F1")
    plt.xticks(x, ablation_results["Experiment"], rotation=20)
    plt.ylabel("Score")
    plt.title("Ablation Study - Missing Modalities")
    plt.legend()
    plt.tight_layout()
    plt.show()
