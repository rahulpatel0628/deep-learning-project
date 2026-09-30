from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "MELD_dataset_with_emotions.csv"
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

INPUT_DIM = 7
EMBEDDING_DIM = 64
NUM_HEADS = 4
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001
RANDOM_STATE = 42
NUM_CLASSES = 7

EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
