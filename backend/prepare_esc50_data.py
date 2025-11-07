import os
import zipfile
import requests
import numpy as np
import pandas as pd
import librosa
from sklearn.model_selection import train_test_split

# Constants
ESC50_URL = "https://github.com/karoldvl/ESC-50/archive/master.zip"
SAMPLE_RATE = 22050
DURATION = 5  # Duration of audio files in seconds
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512

def download_and_extract_esc50():
    """Download and extract the ESC-50 dataset."""
    # Get the root project directory
    project_root = os.path.dirname(os.path.dirname(__file__))
    data_dir = os.path.join(project_root, 'data')
    esc50_dir = os.path.join(data_dir, 'ESC-50')
    
    if not os.path.exists(esc50_dir):
        print("Downloading ESC-50 dataset...")
        response = requests.get(ESC50_URL)
        zip_path = os.path.join(project_root, "esc50.zip")
        with open(zip_path, "wb") as f:
            f.write(response.content)
        
        print("Extracting dataset...")
        os.makedirs(data_dir, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(data_dir)
        
        # Rename the extracted folder
        os.rename(os.path.join(data_dir, "ESC-50-master"), esc50_dir)
        os.remove(zip_path)
        print("Dataset downloaded and extracted successfully!")
    else:
        print("ESC-50 dataset already exists.")

def load_audio_file(file_path):
    """Load and preprocess audio file."""
    try:
        # Load audio file
        audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
        
        # Pad or truncate audio to fixed length
        target_length = DURATION * SAMPLE_RATE
        if len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)))
        else:
            audio = audio[:target_length]
        
        # Extract mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=SAMPLE_RATE,
            n_mels=N_MELS,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH
        )
        
        # Convert to log scale
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normalize
        mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min())
        
        return mel_spec_norm
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None

def prepare_dataset():
    """Prepare the ESC-50 dataset for training."""
    # Get the root project directory
    project_root = os.path.dirname(os.path.dirname(__file__))
    data_dir = os.path.join(project_root, 'data')
    esc50_dir = os.path.join(data_dir, 'ESC-50')
    
    # Download and extract if needed
    download_and_extract_esc50()
    
    # Read metadata
    meta_file = os.path.join(esc50_dir, "meta/esc50.csv")
    df = pd.read_csv(meta_file)
    
    # Initialize lists to store features and labels
    features = []
    labels = []
    
    # Process each audio file
    for idx, row in df.iterrows():
        file_path = os.path.join(esc50_dir, "audio", row['filename'])
        print(f"Processing {idx+1}/{len(df)}: {file_path}")
        
        mel_spec = load_audio_file(file_path)
        if mel_spec is not None:
            features.append(mel_spec)
            labels.append(row['category'])
    
    # Convert to numpy arrays
    X = np.array(features)
    y = np.array(labels)
    
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Save processed data
    np.save(os.path.join(data_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(data_dir, 'X_test.npy'), X_test)
    np.save(os.path.join(data_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(data_dir, 'y_test.npy'), y_test)
    
    print("Dataset preparation completed!")
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    prepare_dataset()