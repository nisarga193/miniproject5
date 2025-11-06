# Animal Voice Detection System

This repository contains a complete AI-based Animal Voice Detection System:

- A CNN-based model training script using TensorFlow/Keras and librosa (`backend/train.py`).
- Reusable preprocessing and inference utilities (`backend/model_utils.py`).
- A Flask backend API to accept audio, return predictions, and store history in MongoDB (`backend/app.py`).
- A simple web frontend (`static/index.html`, `static/app.js`, `static/style.css`) to record or upload audio and view detection history.

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

Train
- Prepare dataset in `data/<species>/*.wav` (one folder per species)
- Example structure:
  - data/dog/*.wav
  - data/cat/*.wav
  - data/bird/*.wav

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

Notes & next steps
- The training script is a starting point — you'll need labeled audio and hyperparameter tuning for production accuracy.
- Consider augmenting data (noise, pitch shift) and using more advanced architectures (pretrained audio networks).

Contact
- This scaffold is intended as a local prototype. Reach out with dataset details and I can help tune and scale it.
