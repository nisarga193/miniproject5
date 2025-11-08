"""
🎤 Live Animal Sound Detection
Record real-time audio using your microphone and classify it using the FD-CNN-CA model.
"""

import os
import tempfile
import numpy as np
import sounddevice as sd
import soundfile as sf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import custom_object_scope
from model_utils import load_audio_file, extract_log_mel
from train_fd_cnn_ca import CoordinateAttention, FrequencyDynamicConv
import json


# ======================================================================================
# Config
# ======================================================================================
MODEL_PATH = "models/animal_fd_cnn_ca.h5"
DURATION = 5           # seconds of recording
SAMPLE_RATE = 22050    # Hz
CUSTOM_OBJECTS = {
    "CoordinateAttention": CoordinateAttention,
    "FrequencyDynamicConv": FrequencyDynamicConv
}


# ======================================================================================
# Load Model and Class Names
# ======================================================================================
print("📦 Loading model...")
with custom_object_scope(CUSTOM_OBJECTS):
    model = load_model(MODEL_PATH)

classes_file = MODEL_PATH + ".classes.json"
if os.path.exists(classes_file):
    with open(classes_file, "r") as f:
        CLASSES = json.load(f)
else:
    CLASSES = [f"class_{i}" for i in range(model.output_shape[-1])]
print(f"✅ Model loaded with {len(CLASSES)} classes.\n")


# ======================================================================================
# Record from Microphone
# ======================================================================================
def record_audio(duration=DURATION, sample_rate=SAMPLE_RATE):
    print(f"🎙️ Recording for {duration} seconds... Speak or play the sound now!")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype="float32")
    sd.wait()
    print("✅ Recording complete!\n")

    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    sf.write(tmp_file.name, audio, sample_rate)
    return tmp_file.name


# ======================================================================================
# Predict Function
# ======================================================================================
def predict_audio(file_path):
    print(f"🎵 Processing audio: {file_path}")
    y_audio = load_audio_file(file_path)
    mel = extract_log_mel(y_audio)
    mel = np.expand_dims(mel, axis=(0, -1))

    preds = model.predict(mel)
    top_idx = int(np.argmax(preds[0]))
    confidence = float(preds[0][top_idx])
    animal = CLASSES[top_idx]

    top3_idx = np.argsort(preds[0])[-3:][::-1]
    print("\n🎯 Prediction Results:")
    for i in top3_idx:
        print(f"  {CLASSES[i]:<15} : {preds[0][i]*100:.2f}%")

    print(f"\n✅ Final Prediction: {animal.upper()} ({confidence*100:.2f}%)")
    return animal, confidence


# ======================================================================================
# Main
# ======================================================================================
if __name__ == "__main__":
    try:
        recorded_file = record_audio()
        predict_audio(recorded_file)
        os.remove(recorded_file)
    except KeyboardInterrupt:
        print("\n🛑 Recording cancelled by user.")
    except Exception as e:
        print(f"❌ Error: {e}")
