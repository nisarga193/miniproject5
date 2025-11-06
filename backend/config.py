import os
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.environ.get('MONGO_URI', 'mongodb://localhost:27017')
DB_NAME = os.environ.get('DB_NAME', 'animal_detection')
MODEL_PATH = os.environ.get('MODEL_PATH', 'models/animal_cnn.h5')
UPLOAD_FOLDER = os.environ.get('UPLOAD_FOLDER', 'uploads')
ALLOWED_EXTENSIONS = {'.wav', '.mp3', '.ogg', '.flac', '.webm'}

# Preprocessing params
SAMPLE_RATE = int(os.environ.get('SAMPLE_RATE', 22050))
DURATION = float(os.environ.get('DURATION', 3.0))  # seconds
N_MELS = int(os.environ.get('N_MELS', 128))
FIXED_FRAMES = int(os.environ.get('FIXED_FRAMES', 128))
