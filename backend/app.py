import os
import json
from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import numpy as np
from datetime import datetime
from pymongo import MongoClient
from config import MODEL_PATH, UPLOAD_FOLDER, MONGO_URI, DB_NAME, ALLOWED_EXTENSIONS
from model_utils import preprocess_file_to_input

# load model lazily
from tensorflow.keras.models import load_model

app = Flask(__name__, static_folder='../static', static_url_path='/')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Mongo
client = None
collection = None
try:
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=2000)
    # quick ping to check availability
    client.admin.command('ping')
    db = client[DB_NAME]
    collection = db['detections']
    print('Connected to MongoDB at', MONGO_URI)
except Exception as e:
    print('Warning: Could not connect to MongoDB at', MONGO_URI, '-', e)
    collection = None

# Load model and classes mapping
print('Attempting to load model. Config MODEL_PATH=', MODEL_PATH)
model = None
classes = None
def _load_model_and_classes(path):
    m = load_model(path)
    cls_path = path + '.classes.json'
    if os.path.exists(cls_path):
        with open(cls_path, 'r') as f:
            cls = json.load(f)
    else:
        cls = [str(i) for i in range(m.output_shape[-1])]
    return m, cls

if os.path.exists(MODEL_PATH):
    try:
        model, classes = _load_model_and_classes(MODEL_PATH)
        print('Loaded model from', MODEL_PATH)
    except Exception as e:
        print('Failed to load model from', MODEL_PATH, 'error:', e)
        model = None

else:
    # Try to find any .h5 model in ./models as a fallback (useful for demo/demo name)
    models_dir = os.path.dirname(MODEL_PATH) or 'models'
    found = []
    if os.path.isdir(models_dir):
        for fname in os.listdir(models_dir):
            if fname.lower().endswith('.h5'):
                found.append(os.path.join(models_dir, fname))
    if found:
        chosen = found[0]
        try:
            model, classes = _load_model_and_classes(chosen)
            print('Model not found at configured path. Loaded fallback model:', chosen)
        except Exception as e:
            print('Found model at', chosen, 'but failed to load:', e)
            model = None
    else:
        print('Model not found at', MODEL_PATH, 'and no .h5 found in', models_dir, "- /predict will fail until a model is available")


def allowed_file(filename):
    ext = os.path.splitext(filename)[1].lower()
    return ext in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    return app.send_static_file('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'no file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'no selected file'}), 400
    if not allowed_file(file.filename):
        return jsonify({'error': 'file type not allowed'}), 400
    filename = secure_filename(file.filename)
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(save_path)

    if model is None:
        return jsonify({'error': 'model not loaded on server'}), 500

    # preprocess
    try:
        x = preprocess_file_to_input(save_path, is_bytes=False)
    except Exception as e:
        return jsonify({'error': 'preprocessing failed', 'detail': str(e)}), 500

    preds = model.predict(x)[0]
    top_idx = int(np.argmax(preds))
    top_label = classes[top_idx] if classes else str(top_idx)
    confidences = {classes[i] if classes else str(i): float(preds[i]) for i in range(len(preds))}

    # store in MongoDB (if available). If MongoDB is not reachable, skip storing.
    rec = {
        'timestamp': datetime.utcnow(),
        'filename': filename,
        'top_label': top_label,
        'confidences': confidences
    }
    if collection is not None:
        try:
            collection.insert_one(rec)
        except Exception as e:
            print('Warning: failed to insert detection into MongoDB:', e)
    else:
        # fallback: write to a local file so history is preserved
        try:
            local_path = os.path.join(os.path.dirname(__file__), '..', 'detections_local.json')
            local_path = os.path.abspath(local_path)
            existing = []
            if os.path.exists(local_path):
                try:
                    with open(local_path, 'r', encoding='utf-8') as f:
                        existing = json.load(f)
                except Exception:
                    existing = []
            # convert timestamp to iso for JSON
            rec_copy = rec.copy()
            rec_copy['timestamp'] = rec_copy['timestamp'].isoformat()
            existing.insert(0, rec_copy)
            with open(local_path, 'w', encoding='utf-8') as f:
                json.dump(existing[:200], f, ensure_ascii=False, indent=2)
        except Exception as e:
            print('Warning: failed to write local detection file:', e)

    return jsonify({'label': top_label, 'confidences': confidences})


@app.route('/history', methods=['GET'])
def history():
    limit = int(request.args.get('limit', 50))
    if collection is not None:
        try:
            docs = list(collection.find().sort('timestamp', -1).limit(limit))
            for d in docs:
                d['_id'] = str(d['_id'])
                if isinstance(d.get('timestamp'), datetime):
                    d['timestamp'] = d['timestamp'].isoformat()
            return jsonify(docs)
        except Exception as e:
            print('Warning: failed to read history from MongoDB:', e)
    # fallback: read local file
    local_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'detections_local.json'))
    if os.path.exists(local_path):
        try:
            with open(local_path, 'r', encoding='utf-8') as f:
                docs = json.load(f)
            return jsonify(docs[:limit])
        except Exception as e:
            print('Warning: failed to read local detection file:', e)
    return jsonify([])


@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
