import os
import io
import numpy as np
import librosa
import soundfile as sf
from config import SAMPLE_RATE, DURATION, N_MELS, FIXED_FRAMES

# Audio utils: load and convert to log-mel spectrogram of fixed size (N_MELS x FIXED_FRAMES)

def load_audio_file(path, sr=SAMPLE_RATE, duration=DURATION):
    # returns numpy array y
    y, _ = librosa.load(path, sr=sr, mono=True, duration=duration)
    # ensure length
    expected_len = int(sr * duration)
    if len(y) < expected_len:
        y = np.pad(y, (0, expected_len - len(y)))
    elif len(y) > expected_len:
        y = y[:expected_len]
    return y


def load_audio_bytes(b, sr=SAMPLE_RATE, duration=DURATION):
    # b: bytes-like
    bio = io.BytesIO(b)
    y, _ = sf.read(bio, dtype='float32')
    # if stereo
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    # resample if needed
    if sr is not None:
        y = librosa.resample(y, orig_sr=_, target_sr=sr) if _ != sr else y
    expected_len = int(sr * duration)
    if len(y) < expected_len:
        y = np.pad(y, (0, expected_len - len(y)))
    elif len(y) > expected_len:
        y = y[:expected_len]
    return y


def extract_log_mel(y, sr=SAMPLE_RATE, n_mels=N_MELS, fixed_frames=FIXED_FRAMES):
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    S_db = librosa.power_to_db(S, ref=np.max)
    # S_db shape is (n_mels, t)
    t = S_db.shape[1]
    if t < fixed_frames:
        pad_width = fixed_frames - t
        S_db = np.pad(S_db, ((0,0),(0,pad_width)), mode='constant', constant_values=(S_db.min(),))
    elif t > fixed_frames:
        S_db = S_db[:, :fixed_frames]
    # normalize to 0-1
    S_norm = (S_db - S_db.min()) / (S_db.max() - S_db.min() + 1e-6)
    return S_norm


def preprocess_file_to_input(path_or_file, is_bytes=False):
    """Returns array shaped (1, n_mels, fixed_frames, 1) ready for model.predict
    path_or_file: filepath if is_bytes=False, else raw bytes
    """
    if is_bytes:
        y = load_audio_bytes(path_or_file)
    else:
        y = load_audio_file(path_or_file)
    mel = extract_log_mel(y)
    mel = mel.astype(np.float32)
    mel = np.expand_dims(mel, -1)  # channel
    mel = np.expand_dims(mel, 0)   # batch
    return mel


if __name__ == '__main__':
    # quick local test (requires a sample.wav in project root)
    if os.path.exists('sample.wav'):
        x = preprocess_file_to_input('sample.wav')
        print('shape', x.shape)
    else:
        print('No sample.wav found for quick test')
