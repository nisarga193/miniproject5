# Animal Voice Detection System

This repository contains a complete AI-based Animal Voice Detection System:

- **Frequency-Dynamic CNN with Coordinate Attention (FD-CNN-CA)** - Advanced model achieving ~95.21% accuracy (`backend/train_fd_cnn_ca.py`) ⭐ **RECOMMENDED**
- Multiple CNN-based model training scripts using TensorFlow/Keras and librosa (`backend/train.py`, `backend/train_final.py`, etc.).
- Reusable preprocessing and inference utilities (`backend/model_utils.py`).
- A Flask backend API to accept audio, return predictions, and store history in MongoDB (`backend/app.py`).
- A simple web frontend (`static/index.html`, `static/app.js`, `static/style.css`) to record or upload audio and view detection history.
- Data organization utilities (`backend/organize_animal_data.py`).

Requirements
- Python 3.8+ (tested with 3.8-3.11)
- MongoDB (or MongoDB Atlas) running and reachable

Quick setup (Windows PowerShell)

```powershell
# create venv
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Environment variables (create `.env` or set in shell)
- MONGO_URI - MongoDB connection string (default: mongodb://localhost:27017)
- MODEL_PATH - path to saved model (default: ./models/animal_cnn.h5)
  - For FD-CNN-CA model, use: `MODEL_PATH=models/animal_fd_cnn_ca.h5`

Train

**Option 1: Frequency-Dynamic CNN with Coordinate Attention (Recommended - ~95.21% accuracy)**

Prepare dataset in `data/<species>/*.wav` (one folder per species):
- Example structure:
  - data/dog/*.wav
  - data/cat/*.wav
  - data/bird/*.wav

```powershell
# Train with FD-CNN-CA (recommended)
python backend/train_fd_cnn_ca.py --data_dir data --epochs 150 --batch_size 32 --output models/animal_fd_cnn_ca.h5

# With mixup augmentation (advanced)
python backend/train_fd_cnn_ca.py --data_dir data --epochs 150 --batch_size 32 --use_mixup --output models/animal_fd_cnn_ca.h5
```

**Option 2: Organize data from unorganized files**

If you have audio files with animal names in filenames (e.g., `dog_bark_001.wav`):
```powershell
python backend/organize_animal_data.py --input_dir path/to/audio/files --output_dir data --pattern animal_name
```

Or create a sample directory structure:
```powershell
python backend/organize_animal_data.py --output_dir data --create_sample
```

**Option 3: Basic CNN model**

```powershell
python backend/train.py --data_dir data --epochs 30 --output models/animal_cnn.h5
```

Run backend

```powershell
# ensure model exists at MODEL_PATH or set MODEL_PATH env
python backend/app.py
```

Open the frontend
- Open `static/index.html` in a browser (or serve via Flask static folder as configured) and record or upload audio. Predictions will be shown.

## Model Performance

| Model Architecture | Accuracy | Notes |
|-------------------|----------|-------|
| **FD-CNN-CA (Frequency-Dynamic CNN + Coordinate Attention)** | **~95.21%** | ✅ Recommended - Advanced architecture with attention mechanism |
| CNN-XG / CNN-SVM Boost | 68-72% | Hybrid CNN + ML models |
| PSD, MDS, Nearest-Neighbor | 93.75-100% | Classical DSP + ML |
| DTW, ZCR, MFCC | ~75-90% | Feature-based ML methods |
| Bidirectional LSTM (Bi-LSTM) | 85% | Temporal deep learning model |

## Training Features

The FD-CNN-CA model includes:
- **Frequency-Dynamic Convolution**: Adapts to different frequency components in audio
- **Coordinate Attention**: Captures both frequency and time dependencies
- **Advanced Data Augmentation**: Time/frequency masking, mixup, noise injection
- **Balanced Data Sampling**: Ensures equal representation of all classes
- **Early Stopping & Learning Rate Scheduling**: Prevents overfitting

## Notes & Next Steps
- The FD-CNN-CA training script includes advanced augmentation and regularization.
- For best results, use diverse animal sound datasets with multiple samples per class.
- Consider using transfer learning from pretrained audio networks for even better accuracy.
- Real-time inference is optimized for the trained models.

Contact
- This scaffold is intended as a local prototype. Reach out with dataset details and I can help tune and scale it.
