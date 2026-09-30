import json
import torch

from config import INPUT_DIM, EMBEDDING_DIM, NUM_HEADS, MODEL_DIR
from model import AttentionFusionModel


def load_attention_model(device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(MODEL_DIR / "label_mapping.json", "r", encoding="utf-8") as f:
        mapping = json.load(f)
    class_names = [mapping[str(i)] for i in range(len(mapping))]

    model = AttentionFusionModel(
        input_dim=INPUT_DIM,
        embedding_dim=EMBEDDING_DIM,
        num_heads=NUM_HEADS,
        num_classes=len(class_names),
    ).to(device)
    model.load_state_dict(torch.load(MODEL_DIR / "best_attention_model.pth", map_location=device))
    model.eval()
    return model, class_names, device


def predict_from_vectors(text_vector, audio_vector, video_vector, model=None, class_names=None, device=None):
    if model is None:
        model, class_names, device = load_attention_model(device)

    text = torch.tensor([text_vector], dtype=torch.float32, device=device)
    audio = torch.tensor([audio_vector], dtype=torch.float32, device=device)
    video = torch.tensor([video_vector], dtype=torch.float32, device=device)

    with torch.no_grad():
        logits, attention = model(text, audio, video)
        probabilities = torch.softmax(logits, dim=1)[0]
        index = int(torch.argmax(probabilities).item())

    return {
        "emotion": class_names[index],
        "confidence": float(probabilities[index].item()),
        "probabilities": {
            name: float(probabilities[i].item()) for i, name in enumerate(class_names)
        },
        "attention": attention[0].detach().cpu().numpy(),
    }
