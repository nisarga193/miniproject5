"""
Predict animal sound using the trained FD-CNN-CA model.
Usage:
    python backend/predict_sound.py --model models/animal_fd_cnn_ca.h5 --file samples/test_dog.wav
"""

import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import custom_object_scope
from model_utils import load_audio_file, extract_log_mel
from train_fd_cnn_ca import CoordinateAttention, FrequencyDynamicConv

# ======================================================================================
# Custom Layers: Needed for model loading
# ======================================================================================
CUSTOM_OBJECTS = {
    "CoordinateAttention": CoordinateAttention,
    "FrequencyDynamicConv": FrequencyDynamicConv
}


# ======================================================================================
# Predict function
# ======================================================================================
def predict_animal_sound(model_path, audio_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Model not found: {model_path}")
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"❌ Audio file not found: {audio_path}")

    # Load model safely with custom layers
    print(f"📦 Loading model from: {model_path}")
    with custom_object_scope(CUSTOM_OBJECTS):
        model = load_model(model_path)

    # Load classes file (for readable labels)
    classes_file = model_path + ".classes.json"
    if os.path.exists(classes_file):
        import json
        with open(classes_file, "r") as f:
            classes = json.load(f)
    else:
        classes = [f"class_{i}" for i in range(model.output_shape[-1])]
    print(f"✅ Classes loaded: {classes}")

    # Process the input audio
    print(f"🎵 Processing audio file: {audio_path}")
    y_audio = load_audio_file(audio_path)
    mel = extract_log_mel(y_audio)
    mel = np.expand_dims(mel, axis=(0, -1))  # (1, freq, time, 1)

    # Predict
    print("🔮 Predicting...")
    preds = model.predict(mel)
    top_idx = np.argmax(preds[0])
    top_class = classes[top_idx]
    confidence = preds[0][top_idx]

    # Display top-3 predictions
    top3_idx = np.argsort(preds[0])[-3:][::-1]
    print("\n🎯 Prediction Results:")
    for i in top3_idx:
        print(f"  {classes[i]:<15} : {preds[0][i]*100:.2f}%")

    print(f"\n✅ Final Prediction: {top_class.upper()} ({confidence*100:.2f}%)")
    return top_class, confidence


# ======================================================================================
# CLI
# ======================================================================================
def main():
    parser = argparse.ArgumentParser(description="Predict animal from an audio file")
    parser.add_argument("--model", required=True, help="Path to trained .h5 model file")
    parser.add_argument("--file", required=True, help="Path to .wav or .mp3 file")
    args = parser.parse_args()

    predict_animal_sound(args.model, args.file)


if __name__ == "__main__":
    main()
