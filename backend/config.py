import os
from dotenv import load_dotenv

# -----------------------------
# Load environment variables
# -----------------------------
load_dotenv()

# -----------------------------
# Folder paths
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
DATA_FOLDER = os.path.join(BASE_DIR, "data")
MODEL_FOLDER = os.path.join(BASE_DIR, "models")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DATA_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)

# -----------------------------
# Model path
# -----------------------------
MODEL_PATH = os.getenv(
    "MODEL_PATH",
    os.path.join(BASE_DIR, "..", "models", "animal_fd_cnn_ca.h5")
)

# -----------------------------
# MongoDB Configuration
# -----------------------------
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("DB_NAME", "animal_detection_db")

# -----------------------------
# Allowed file types
# -----------------------------
ALLOWED_EXTENSIONS = {'.wav', '.mp3', '.ogg', '.webm', '.m4a'}

# -----------------------------
# Audio Preprocessing Parameters
# -----------------------------
# These constants are required by model_utils.py
SAMPLE_RATE = 22050          # Sampling rate for audio files
DURATION = 4.0               # Clip length in seconds
N_MELS = 128                 # Number of Mel bands for spectrogram
FIXED_FRAMES = 128           # Fixed number of time frames for CNN input

# -----------------------------
# Debug info
# -----------------------------
print(f"[CONFIG] MODEL_PATH = {MODEL_PATH}")
print(f"[CONFIG] UPLOAD_FOLDER = {UPLOAD_FOLDER}")
print(f"[CONFIG] MONGO_URI = {MONGO_URI}")
