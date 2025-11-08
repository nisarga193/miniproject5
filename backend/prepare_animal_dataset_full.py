import os
import shutil
import zipfile
import pandas as pd
import urllib.request
import random
from pydub import AudioSegment

# ---------------- CONFIG ----------------
PROJECT_DIR = r"C:\Users\lenovo\Documents\miniproject5"
ESC50_ZIP_URL = "https://github.com/karoldvl/ESC-50/archive/refs/heads/master.zip"
ESC50_ZIP_PATH = os.path.join(PROJECT_DIR, "ESC-50.zip")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "data")
ESC50_EXTRACT_DIR = os.path.join(PROJECT_DIR, "ESC-50")

ANIMAL_CLASSES = [
    "dog", "cat", "cow", "pig", "sheep", "rooster", "hen", "crow",
    "bird", "insects", "frog", "horse"
]

TARGET_SAMPLE_RATE = 16000
TARGET_CHANNELS = 1
TRAIN_SPLIT = 0.8
# ----------------------------------------


def download_esc50():
    """Download and extract ESC-50 dataset safely."""
    possible_dirs = [
        os.path.join(PROJECT_DIR, "ESC-50"),
        os.path.join(PROJECT_DIR, "ESC-50-master"),
        os.path.join(PROJECT_DIR, "ESC-50-main"),
    ]
    for d in possible_dirs:
        if os.path.exists(d):
            print(f"✅ Found existing ESC-50 dataset at: {d}")
            return d

    print("⬇️ Downloading ESC-50 dataset (≈600 MB)...")
    urllib.request.urlretrieve(ESC50_ZIP_URL, ESC50_ZIP_PATH)
    print("✅ Download complete!")

    print("📦 Extracting dataset...")
    with zipfile.ZipFile(ESC50_ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall(PROJECT_DIR)
    print("✅ Extraction complete!")

    for d in possible_dirs:
        if os.path.exists(d):
            print(f"📁 ESC-50 dataset extracted to: {d}")
            return d

    raise FileNotFoundError("❌ Could not locate extracted ESC-50 folder after extraction.")


def convert_to_wav(src_path, dst_path):
    """Convert audio file to mono 16 kHz WAV using pydub."""
    try:
        audio = AudioSegment.from_file(src_path)
        audio = audio.set_frame_rate(TARGET_SAMPLE_RATE).set_channels(TARGET_CHANNELS)
        audio.export(dst_path, format="wav")
    except Exception as e:
        print(f"⚠️ Conversion failed for {src_path}: {e}")


def prepare_animal_dataset():
    """Extract only animal classes and convert them to clean WAV files."""
    extracted_dir = download_esc50()

    # Locate meta and audio directories
    meta_file = os.path.join(extracted_dir, "meta", "esc50.csv")
    audio_dir = os.path.join(extracted_dir, "audio")

    if not os.path.exists(meta_file):
        raise FileNotFoundError(f"Metadata file not found at expected path: {meta_file}")

    df = pd.read_csv(meta_file)
    df_animals = df[df["category"].isin(ANIMAL_CLASSES)]
    print(f"📊 Found {len(df_animals)} animal sound samples in ESC-50")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for cls in ANIMAL_CLASSES:
        os.makedirs(os.path.join(OUTPUT_DIR, cls), exist_ok=True)

    copied = 0
    for _, row in df_animals.iterrows():
        src = os.path.join(audio_dir, row["filename"])
        dest_folder = os.path.join(OUTPUT_DIR, row["category"])
        dest_file = os.path.join(dest_folder, os.path.splitext(row["filename"])[0] + ".wav")

        if os.path.exists(src):
            convert_to_wav(src, dest_file)
            copied += 1
        else:
            print(f"⚠️ Missing audio file: {src}")

    print(f"\n✅ {copied} animal audio files processed and converted to WAV.")
    print(f"🎉 Raw dataset ready under: {OUTPUT_DIR}")

    split_train_test()


def split_train_test():
    """Automatically split dataset into train/test folders."""
    print("\n📂 Splitting dataset into train/test sets (80/20)...")

    train_dir = os.path.join(OUTPUT_DIR, "train")
    test_dir = os.path.join(OUTPUT_DIR, "test")

    # Remove old splits if they exist
    if os.path.exists(train_dir):
        shutil.rmtree(train_dir)
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    for cls in ANIMAL_CLASSES:
        src_folder = os.path.join(OUTPUT_DIR, cls)
        files = [f for f in os.listdir(src_folder) if f.endswith(".wav")]
        random.shuffle(files)

        split_idx = int(len(files) * TRAIN_SPLIT)
        train_files = files[:split_idx]
        test_files = files[split_idx:]

        os.makedirs(os.path.join(train_dir, cls), exist_ok=True)
        os.makedirs(os.path.join(test_dir, cls), exist_ok=True)

        for f in train_files:
            shutil.copy(os.path.join(src_folder, f), os.path.join(train_dir, cls, f))
        for f in test_files:
            shutil.copy(os.path.join(src_folder, f), os.path.join(test_dir, cls, f))

        print(f"✅ {cls}: {len(train_files)} train, {len(test_files)} test")

    print(f"\n🎯 Train/Test split complete!")
    print(f"📁 Train folder: {train_dir}")
    print(f"📁 Test folder: {test_dir}")


if __name__ == "__main__":
    print("🐾 Preparing ESC-50 animal dataset with auto conversion + train/test split...")
    prepare_animal_dataset()
    print("✅ All done! You can now train your FD-CNN-CA model on 'data/train'")
