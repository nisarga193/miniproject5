import numpy as np
import librosa
import tensorflow as tf
from config import SAMPLE_RATE, DURATION, N_MELS, FIXED_FRAMES


# ---------------------------------------------
# ✅ Preprocessing function for inference
# ---------------------------------------------
def preprocess_file_to_input(filepath, is_bytes=False):
    """
    Preprocess an audio file into a normalized Mel-spectrogram tensor 
    with shape (1, 128, 128, 1) for FD_CNN_CA model inference.
    """

    # 1️⃣ Load the audio file
    y, sr = librosa.load(filepath, sr=SAMPLE_RATE, duration=DURATION, mono=True)

    # 2️⃣ Pad or truncate to fixed length
    target_len = int(DURATION * SAMPLE_RATE)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    elif len(y) > target_len:
        y = y[:target_len]

    # 3️⃣ Compute Mel-spectrogram
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=N_MELS, fmax=sr / 2
    )

    # 4️⃣ Convert to decibel (log) scale
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # 5️⃣ Normalize to [0, 1] range
    mel_db -= mel_db.min()
    mel_db /= mel_db.max() + 1e-6

    # 6️⃣ Ensure consistent shape → (128, 128)
    if mel_db.shape[1] < FIXED_FRAMES:
        # pad with zeros (time axis)
        pad_width = FIXED_FRAMES - mel_db.shape[1]
        mel_db = np.pad(mel_db, ((0, 0), (0, pad_width)), mode="constant")
    elif mel_db.shape[1] > FIXED_FRAMES:
        # trim to required size
        mel_db = mel_db[:, :FIXED_FRAMES]

    # 7️⃣ Add channel and batch dimensions
    mel_db = mel_db[..., np.newaxis]        # (128, 128, 1)
    mel_db = np.expand_dims(mel_db, axis=0) # (1, 128, 128, 1)

    return mel_db


# ---------------------------------------------
# 🧩 (Optional) Dummy functions for training imports
# ---------------------------------------------
# These prevent ImportError when loading custom layers from train_fd_cnn_ca.py
def load_audio_file(path, sr=SAMPLE_RATE, duration=DURATION):
    """Simple loader used for training imports (not used in inference)."""
    y, sr = librosa.load(path, sr=sr, duration=duration, mono=True)
    return y, sr


def extract_log_mel(y, sr, n_mels=N_MELS):
    """Extracts log-Mel spectrogram for compatibility with training scripts."""
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db
