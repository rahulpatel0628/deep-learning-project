import json
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import TensorDataset, DataLoader

from config import DATA_PATH, BATCH_SIZE, RANDOM_STATE


def load_data(data_path=DATA_PATH):
    """Load the processed MELD CSV."""
    df = pd.read_csv(data_path)
    return df


def parse_predictions(x):
    """Parse Multi_Model_Predictions JSON stored in the CSV."""
    return json.loads(x)


def prepare_features(df):
    """Extract 7-D video, audio and text modality-level prediction vectors."""
    df = df.copy()

    df["predictions"] = df["Multi_Model_Predictions"].apply(parse_predictions)
    df["video_features"] = df["predictions"].apply(lambda x: x["video"])
    df["audio_features"] = df["predictions"].apply(lambda x: x["audio"])
    df["text_features"] = df["predictions"].apply(lambda x: x["text"])

    X_video = np.array(df["video_features"].tolist(), dtype=np.float32)
    X_audio = np.array(df["audio_features"].tolist(), dtype=np.float32)
    X_text = np.array(df["text_features"].tolist(), dtype=np.float32)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df["Emotion"])

    return df, X_text, X_audio, X_video, y, label_encoder


def make_splits(y, random_state=RANDOM_STATE):
    """Create the notebook's stratified 80/10/10 split."""
    indices = np.arange(len(y))

    train_idx, temp_idx = train_test_split(
        indices,
        test_size=0.20,
        random_state=random_state,
        stratify=y,
    )

    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.50,
        random_state=random_state,
        stratify=y[temp_idx],
    )

    return train_idx, val_idx, test_idx


def make_tensors(X_text, X_audio, X_video, y):
    return (
        torch.tensor(X_text),
        torch.tensor(X_audio),
        torch.tensor(X_video),
        torch.tensor(y, dtype=torch.long),
    )


def make_multimodal_loaders(text_tensor, audio_tensor, video_tensor, label_tensor,
                            train_idx, val_idx, test_idx, batch_size=BATCH_SIZE):
    train_dataset = TensorDataset(
        text_tensor[train_idx], audio_tensor[train_idx],
        video_tensor[train_idx], label_tensor[train_idx]
    )
    val_dataset = TensorDataset(
        text_tensor[val_idx], audio_tensor[val_idx],
        video_tensor[val_idx], label_tensor[val_idx]
    )
    test_dataset = TensorDataset(
        text_tensor[test_idx], audio_tensor[test_idx],
        video_tensor[test_idx], label_tensor[test_idx]
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def make_unimodal_loaders(feature_tensor, label_tensor, train_idx, val_idx, test_idx,
                          batch_size=BATCH_SIZE):
    train_dataset = TensorDataset(feature_tensor[train_idx], label_tensor[train_idx])
    val_dataset = TensorDataset(feature_tensor[val_idx], label_tensor[val_idx])
    test_dataset = TensorDataset(feature_tensor[test_idx], label_tensor[test_idx])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
