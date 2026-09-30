import torch
import torch.nn as nn


class AttentionFusionModel(nn.Module):
    """Attention fusion model copied from the project notebook architecture."""

    def __init__(self, input_dim=7, embedding_dim=64, num_heads=4, num_classes=7):
        super().__init__()

        self.text_projection = nn.Sequential(
            nn.Linear(input_dim, embedding_dim),
            nn.ReLU(),
        )

        self.audio_projection = nn.Sequential(
            nn.Linear(input_dim, embedding_dim),
            nn.ReLU(),
        )

        self.video_projection = nn.Sequential(
            nn.Linear(input_dim, embedding_dim),
            nn.ReLU(),
        )

        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            batch_first=True,
        )

        self.norm = nn.LayerNorm(embedding_dim)

        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, text, audio, video):
        text_emb = self.text_projection(text).unsqueeze(1)
        audio_emb = self.audio_projection(audio).unsqueeze(1)
        video_emb = self.video_projection(video).unsqueeze(1)

        modalities = torch.cat(
            [text_emb, audio_emb, video_emb], dim=1
        )

        attended, attention_weights = self.attention(
            modalities, modalities, modalities
        )

        fused = attended.mean(dim=1)
        fused = self.norm(fused)
        output = self.classifier(fused)

        return output, attention_weights


class UnimodalClassifier(nn.Module):
    """Baseline classifier used for Text-only, Audio-only and Video-only models."""

    def __init__(self, input_dim=7, num_classes=7):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        return self.network(x)


class ConcatenationModel(nn.Module):
    """Simple concatenation baseline used in the notebook."""

    def __init__(self, input_dim=7, num_classes=7):
        super().__init__()
        total_dim = input_dim * 3
        self.classifier = nn.Sequential(
            nn.Linear(total_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, text, audio, video):
        combined = torch.cat([text, audio, video], dim=1)
        return self.classifier(combined)
