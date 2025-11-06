"""
Transfer learning based model with signal processing features
and time-frequency attention for audio classification.
"""
import os
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.layers import (
    Layer, Dense, Dropout, GlobalAveragePooling2D, Input,
    Reshape, Permute, Multiply, Lambda, Add, BatchNormalization, Concatenate
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import librosa
from model_utils import load_audio_file, extract_log_mel

class TimeFrequencyAttention(Layer):
    """Simple time-frequency attention that produces separate
    attention maps over time and frequency and applies them to
    the input feature map. This implementation avoids learnable
    weight shape mismatches and is robust across different
    channel sizes.
    """
    def __init__(self, **kwargs):
        super(TimeFrequencyAttention, self).__init__(**kwargs)

    def call(self, x):
        # x: (batch, freq, time, channels)
        # Time attention: collapse frequency and channel, keep time
        time_att = tf.reduce_mean(x, axis=[1, 3], keepdims=True)  # (batch, 1, time, 1)
        time_att = tf.nn.softmax(time_att, axis=2)
        time_weighted = x * time_att

        # Frequency attention: collapse time and channel, keep frequency
        freq_att = tf.reduce_mean(x, axis=[2, 3], keepdims=True)  # (batch, freq, 1, 1)
        freq_att = tf.nn.softmax(freq_att, axis=1)
        freq_weighted = x * freq_att

        return time_weighted + freq_weighted

def extract_features(audio, sr):
    """Extract additional audio features beyond mel spectrogram."""
    features = []
    
    # Spectral centroid
    cent = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
    cent = librosa.util.normalize(cent)
    features.append(cent)
    
    # Spectral rolloff
    rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
    rolloff = librosa.util.normalize(rolloff)
    features.append(rolloff)
    
    # Zero crossing rate
    zcr = librosa.feature.zero_crossing_rate(audio)[0]
    zcr = librosa.util.normalize(zcr)
    features.append(zcr)
    
    # RMS energy
    rms = librosa.feature.rms(y=audio)[0]
    rms = librosa.util.normalize(rms)
    features.append(rms)
    
    # Stack features
    features = np.stack(features)
    
    # Interpolate to match spectrogram length
    target_len = 128  # Match mel spectrogram width
    features = np.array([
        np.interp(
            np.linspace(0, len(f), target_len),
            np.arange(len(f)),
            f
        )
        for f in features
    ])
    
    # Reshape to (4, 128, 1) - adding channel dimension
    features = np.expand_dims(features, axis=-1)
    
    return features

def build_model(input_shape, n_classes):
    # Main input for spectrogram (single-channel)
    input_mel = Input(shape=input_shape)

    # Convert single-channel mel spectrogram to 3 channels so we can
    # use ImageNet-pretrained ResNet which expects 3-channel input.
    x_rgb = Lambda(lambda x: tf.image.grayscale_to_rgb(x), name='gray_to_rgb')(input_mel)

    # Base ResNet model (pre-trained on ImageNet) expecting 3 channels
    base_model = ResNet50V2(
        include_top=False,
        weights='imagenet',
        input_shape=(input_shape[0], input_shape[1], 3)
    )

    # Freeze early layers
    for layer in base_model.layers[:100]:
        layer.trainable = False
    
    # Additional input for signal features
    input_features = Input(shape=(4, 128, 1))
    
    # Process mel spectrogram (converted to RGB)
    x = base_model(x_rgb)
    
    # Add time-frequency attention
    x = TimeFrequencyAttention()(x)
    
    # Process signal features
    features = TimeFrequencyAttention()(input_features)
    features = GlobalAveragePooling2D()(features)
    features = Dense(64, activation='relu')(features)
    features = BatchNormalization()(features)
    
    # Combine features
    x = GlobalAveragePooling2D()(x)
    x = Concatenate(axis=-1)([x, features])
    
    # Final classification layers
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    outputs = Dense(n_classes, activation='softmax')(x)
    
    model = Model([input_mel, input_features], outputs)
    
    model.compile(
        optimizer=Adam(1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

class AudioDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, X_mel, X_features, y, batch_size=32, augment=False):
        self.X_mel = X_mel
        self.X_features = X_features
        self.y = y
        self.batch_size = batch_size
        self.n_classes = len(np.unique(y))
        self.augment = augment
        
        # Create class indices for balanced sampling
        self.class_indices = [np.where(y == i)[0] for i in range(self.n_classes)]
        self.min_class_size = min(len(indices) for indices in self.class_indices)
        
    def __len__(self):
        return int(np.ceil(self.min_class_size * self.n_classes / self.batch_size))
    
    def __getitem__(self, idx):
        # For each batch, sample equal numbers from each class
        samples_per_class = self.batch_size // self.n_classes
        batch_indices = []
        
        for class_idx in range(self.n_classes):
            indices = np.random.choice(
                self.class_indices[class_idx],
                size=samples_per_class,
                replace=True
            )
            batch_indices.extend(indices)
        
        # Shuffle within the batch
        np.random.shuffle(batch_indices)
        
        # Get the data
        batch_X_mel = self.X_mel[batch_indices]
        batch_X_features = self.X_features[batch_indices]
        batch_y = tf.keras.utils.to_categorical(self.y[batch_indices], self.n_classes)
        
        # Apply augmentation if enabled
        if self.augment:
            batch_X_mel, batch_X_features = self.augment_batch(
                batch_X_mel, batch_X_features
            )
        
        return [batch_X_mel, batch_X_features], batch_y
    
    def augment_batch(self, mel_specs, features):
        """Apply augmentation to both mel spectrograms and features."""
        augmented_mel = []
        augmented_features = []
        
        for mel, feat in zip(mel_specs, features):
            # Random time shift
            if np.random.random() < 0.8:
                shift = int(mel.shape[1] * np.random.uniform(-0.2, 0.2))
                if shift > 0:
                    mel = np.roll(mel, shift, axis=1)
                    feat = np.roll(feat, shift, axis=1)
                elif shift < 0:
                    mel = np.roll(mel, shift, axis=1)
                    feat = np.roll(feat, shift, axis=1)
            
            # Random frequency masking
            if np.random.random() < 0.8:
                n_masks = np.random.randint(1, 3)
                for _ in range(n_masks):
                    mask_size = int(mel.shape[0] * np.random.uniform(0.05, 0.2))
                    mask_start = np.random.randint(0, mel.shape[0] - mask_size)
                    mel[mask_start:mask_start + mask_size, :, :] = 0
                    feat[mask_start % feat.shape[0]:
                         (mask_start + mask_size) % feat.shape[0], :, :] = 0
            
            # Random time masking
            if np.random.random() < 0.8:
                n_masks = np.random.randint(1, 3)
                for _ in range(n_masks):
                    mask_size = int(mel.shape[1] * np.random.uniform(0.05, 0.2))
                    mask_start = np.random.randint(0, mel.shape[1] - mask_size)
                    mel[:, mask_start:mask_start + mask_size, :] = 0
                    feat[:, mask_start:mask_start + mask_size, :] = 0
            
            # Add noise
            if np.random.random() < 0.8:
                noise_level = np.random.uniform(0.001, 0.005)
                mel = mel + np.random.normal(0, noise_level, mel.shape)
                feat = feat + np.random.normal(0, noise_level, feat.shape)
                mel = np.clip(mel, 0, 1)
                feat = np.clip(feat, 0, 1)
            
            augmented_mel.append(mel)
            augmented_features.append(feat)
        
        return np.array(augmented_mel), np.array(augmented_features)

def gather_data(data_dir):
    classes = sorted([d for d in os.listdir(data_dir) 
                     if os.path.isdir(os.path.join(data_dir,d))])
    
    X_mel = []
    X_features = []
    y = []
    files_used = []
    
    print('Loading audio files from classes:', classes)
    
    for idx, cls in enumerate(classes):
        cls_dir = os.path.join(data_dir, cls)
        print(f'Processing {cls} files...')
        n_processed = 0
        
        for fname in os.listdir(cls_dir):
            if not fname.lower().endswith(('.wav', '.mp3', '.flac', '.ogg')):
                continue
                
            path = os.path.join(cls_dir, fname)
            try:
                audio = load_audio_file(path)
                
                # Extract mel spectrogram
                mel = extract_log_mel(audio)
                
                # Extract additional features
                features = extract_features(audio, sr=22050)  # Using default sr
                
                X_mel.append(mel)
                X_features.append(features)
                y.append(idx)
                files_used.append(path)
                n_processed += 1
            except Exception as e:
                print(f'Error processing {path}: {e}')
        
        print(f'Successfully processed {n_processed} files for class {cls}')
    
    X_mel = np.array(X_mel)
    X_mel = np.expand_dims(X_mel, -1)  # Add channel dimension
    X_features = np.array(X_features)  # Already has shape (N, 4, 128, 1)
    y = np.array(y)
    
    print('\nDataset summary:')
    print('Mel spectrogram shape:', X_mel.shape)
    print('Features shape:', X_features.shape)
    print('Label shape:', y.shape)
    print('\nClass distribution:')
    for cls, count in zip(classes, np.bincount(y)):
        print(f'{cls}: {count} samples')
    
    return X_mel, X_features, y, classes, files_used

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', required=True)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch_size', type=int, default=24)
    p.add_argument('--output', default='models/animal_cnn.h5')
    args = p.parse_args()

    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)

    # Gather data with both mel spectrograms and additional features
    X_mel, X_features, y, classes, files_used = gather_data(args.data_dir)
    print('\nLoaded dataset with classes:', classes)

    if len(classes) < 2:
        raise SystemExit('Need >=2 classes in dataset')

    # Split preserving class distribution
    splits = train_test_split(
        X_mel, X_features, y, files_used,
        test_size=0.2, random_state=42, stratify=y
    )
    X_mel_train, X_mel_test, X_feat_train, X_feat_test, y_train, y_test, files_train, files_test = splits
    
    splits = train_test_split(
        X_mel_train, X_feat_train, y_train, files_train,
        test_size=0.2, random_state=42, stratify=y_train
    )
    X_mel_train, X_mel_val, X_feat_train, X_feat_val, y_train, y_val, files_train, files_val = splits

    print('\nSplit sizes:')
    print('Train:', X_mel_train.shape[0])
    print('Validation:', X_mel_val.shape[0])
    print('Test:', X_mel_test.shape[0])

    model = build_model(X_mel_train.shape[1:], len(classes))
    model.summary()

    # Create output directory
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Create balanced training set by oversampling minority classes
    def oversample(X_mel, X_feat, y):
        classes = np.unique(y)
        counts = [np.sum(y == c) for c in classes]
        max_count = max(counts)
        X_mel_res = []
        X_feat_res = []
        y_res = []
        for c in classes:
            idx = np.where(y == c)[0]
            if len(idx) == 0:
                continue
            reps = np.random.choice(idx, size=max_count, replace=True)
            X_mel_res.append(X_mel[reps])
            X_feat_res.append(X_feat[reps])
            y_res.append(y[reps])
        X_mel_res = np.concatenate(X_mel_res, axis=0)
        X_feat_res = np.concatenate(X_feat_res, axis=0)
        y_res = np.concatenate(y_res, axis=0)
        p = np.random.permutation(len(y_res))
        return X_mel_res[p], X_feat_res[p], y_res[p]

    X_mel_train_bal, X_feat_train_bal, y_train_bal = oversample(
        X_mel_train, X_feat_train, y_train
    )

    # Callbacks
    callbacks = [
        ModelCheckpoint(
            args.output,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_accuracy',
            patience=20,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_accuracy',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]

    # Train using numpy arrays (balanced)
    history = model.fit(
        [X_mel_train_bal, X_feat_train_bal],
        tf.keras.utils.to_categorical(y_train_bal, len(classes)),
        validation_data=(
            [X_mel_val, X_feat_val],
            tf.keras.utils.to_categorical(y_val, len(classes))
        ),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        shuffle=True
    )

    # Evaluate on test set
    print('\nEvaluating on test set...')
    y_test_cat = tf.keras.utils.to_categorical(y_test, len(classes))
    test_loss, test_acc = model.evaluate(
        [X_mel_test, X_feat_test],
        y_test_cat
    )
    print(f'Test accuracy: {test_acc:.4f}')

    # Get predictions and compute confusion matrix
    y_pred = model.predict([X_mel_test, X_feat_test])
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    print('\nConfusion Matrix:')
    cm = confusion_matrix(y_test, y_pred_classes)
    for i, row in enumerate(cm):
        print(f'{classes[i]:>10}:', ' '.join(f'{x:>4}' for x in row))
    
    print('\nClassification Report:')
    print(classification_report(y_test, y_pred_classes, target_names=classes))

    # Save classes mapping
    import json
    classes_file = args.output + '.classes.json'
    with open(classes_file, 'w') as f:
        json.dump(classes, f)
    print(f'\nSaved model to {args.output}')
    print(f'Saved classes to {classes_file}')

    # Save error analysis
    errors_file = args.output + '.errors.txt'
    with open(errors_file, 'w') as f:
        f.write('Misclassified examples:\n\n')
        for i, (true, pred) in enumerate(zip(y_test, y_pred_classes)):
            if true != pred:
                f.write(f'File: {files_test[i]}\n')
                f.write(f'True: {classes[true]}, Predicted: {classes[pred]}\n')
                probs = y_pred[i]
                f.write('Class probabilities:\n')
                for cls, prob in zip(classes, probs):
                    f.write(f'  {cls}: {prob:.4f}\n')
                f.write('\n')
    print(f'Saved error analysis to {errors_file}')

if __name__ == '__main__':
    main()