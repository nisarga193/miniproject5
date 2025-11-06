"""
Final attempt at training script with stronger regularization
and simpler architecture to prevent overfitting.
"""
import os
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization,
    GaussianNoise, Input, LeakyReLU, GlobalAveragePooling2D
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from model_utils import load_audio_file, extract_log_mel

# Data augmentation for audio spectrograms
class AudioAugmentation:
    @staticmethod
    def add_noise(mel_spec, noise_level=0.005):
        noise = np.random.normal(0, noise_level, mel_spec.shape)
        return np.clip(mel_spec + noise, 0, 1)
    
    @staticmethod
    def time_shift(mel_spec, max_shift_pct=0.3):
        rows, cols = mel_spec.shape
        shift = int(cols * np.random.uniform(-max_shift_pct, max_shift_pct))
        if shift > 0:
            mel_spec = np.hstack([mel_spec[:, -shift:], mel_spec[:, :-shift]])
        elif shift < 0:
            mel_spec = np.hstack([mel_spec[:, -shift:], mel_spec[:, :-shift]])
        return mel_spec
    
    @staticmethod
    def frequency_mask(mel_spec, max_mask_pct=0.2, num_masks=2):
        result = mel_spec.copy()
        rows, cols = mel_spec.shape
        for _ in range(num_masks):
            mask_size = int(rows * np.random.uniform(0, max_mask_pct))
            mask_start = np.random.randint(0, rows - mask_size)
            result[mask_start:mask_start + mask_size, :] = 0
        return result

    @staticmethod
    def time_mask(mel_spec, max_mask_pct=0.2, num_masks=2):
        result = mel_spec.copy()
        rows, cols = mel_spec.shape
        for _ in range(num_masks):
            mask_size = int(cols * np.random.uniform(0, max_mask_pct))
            mask_start = np.random.randint(0, cols - mask_size)
            result[:, mask_start:mask_start + mask_size] = 0
        return result

    @staticmethod
    def augment(mel_spec, p=0.8):
        if np.random.random() < p:
            mel_spec = AudioAugmentation.add_noise(mel_spec)
        if np.random.random() < p:
            mel_spec = AudioAugmentation.time_shift(mel_spec)
        if np.random.random() < p:
            mel_spec = AudioAugmentation.frequency_mask(mel_spec)
        if np.random.random() < p:
            mel_spec = AudioAugmentation.time_mask(mel_spec)
        return mel_spec

def build_model(input_shape, n_classes):
    # Much simpler architecture with stronger regularization
    model = Sequential([
        Input(shape=input_shape),
        
        # Initial noise for robustness
        GaussianNoise(0.02),
        
        # First block - keep simple
        Conv2D(32, (5,5), padding='same'),
        LeakyReLU(0.2),
        BatchNormalization(),
        MaxPooling2D((2,2)),
        Dropout(0.3),

        # Second block
        Conv2D(64, (3,3), padding='same'),
        LeakyReLU(0.2),
        BatchNormalization(),
        MaxPooling2D((2,2)),
        Dropout(0.3),

        # Third block
        Conv2D(128, (3,3), padding='same'),
        LeakyReLU(0.2),
        BatchNormalization(),
        MaxPooling2D((2,2)),
        Dropout(0.3),

        # Global pooling instead of flatten
        GlobalAveragePooling2D(),
        
        # Single dense layer with strong dropout
        Dense(128),
        LeakyReLU(0.2),
        BatchNormalization(),
        Dropout(0.5),
        
        Dense(n_classes, activation='softmax')
    ])

    # Use much lower learning rate
    model.compile(
        optimizer=Adam(5e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

class BalancedDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, X, y, batch_size=32, augment=False, shuffle=True):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.augment = augment
        self.shuffle = shuffle
        self.n_classes = len(np.unique(y))
        
        # Create class indices for balanced sampling
        self.class_indices = [np.where(y == i)[0] for i in range(self.n_classes)]
        self.min_class_size = min(len(indices) for indices in self.class_indices)
        
        if self.shuffle:
            self.on_epoch_end()
    
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
        batch_X = self.X[batch_indices]
        batch_y = to_categorical(self.y[batch_indices], self.n_classes)
        
        # Apply augmentation if enabled
        if self.augment:
            batch_X = np.array([
                AudioAugmentation.augment(x[:,:,0])[:,:,np.newaxis]
                for x in batch_X
            ])
        
        return batch_X, batch_y
    
    def on_epoch_end(self):
        if self.shuffle:
            for indices in self.class_indices:
                np.random.shuffle(indices)

def gather_data(data_dir):
    classes = sorted([d for d in os.listdir(data_dir) 
                     if os.path.isdir(os.path.join(data_dir,d))])
    
    X = []
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
                y_audio = load_audio_file(path)
                mel = extract_log_mel(y_audio)
                # Add slight noise during loading for robustness
                mel = mel + np.random.normal(0, 0.001, mel.shape)
                mel = np.clip(mel, 0, 1)
                X.append(mel)
                y.append(idx)
                files_used.append(path)
                n_processed += 1
            except Exception as e:
                print(f'Error processing {path}: {e}')
        
        print(f'Successfully processed {n_processed} files for class {cls}')
    
    X = np.array(X)
    X = np.expand_dims(X, -1)
    y = np.array(y)
    
    print('\nDataset summary:')
    print('Input shape:', X.shape)
    print('Label shape:', y.shape)
    print('\nClass distribution:')
    for cls, count in zip(classes, np.bincount(y)):
        print(f'{cls}: {count} samples')
    
    return X, y, classes, files_used

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', required=True)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--output', default='models/animal_cnn.h5')
    args = p.parse_args()

    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)

    X, y, classes, files_used = gather_data(args.data_dir)
    print('\nLoaded dataset with classes:', classes)

    if len(classes) < 2:
        raise SystemExit('Need >=2 classes in dataset')

    # Split preserving class distribution
    X_train, X_test, y_train, y_test, files_train, files_test = train_test_split(
        X, y, files_used, test_size=0.2, random_state=42, stratify=y
    )
    
    X_train, X_val, y_train, y_val, files_train, files_val = train_test_split(
        X_train, y_train, files_train, test_size=0.2, random_state=42, stratify=y_train
    )

    print('\nSplit sizes:')
    print('Train:', X_train.shape[0])
    print('Validation:', X_val.shape[0])
    print('Test:', X_test.shape[0])

    model = build_model(X_train.shape[1:], len(classes))
    model.summary()

    # Create output directory
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Data generators with strong augmentation
    train_gen = BalancedDataGenerator(X_train, y_train, args.batch_size, augment=True)
    val_gen = BalancedDataGenerator(X_val, y_val, args.batch_size, augment=False)

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

    # Train with balanced batches and augmentation
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.epochs,
        callbacks=callbacks
    )

    # Evaluate on test set
    print('\nEvaluating on test set...')
    y_test_cat = to_categorical(y_test, len(classes))
    test_loss, test_acc = model.evaluate(X_test, y_test_cat)
    print(f'Test accuracy: {test_acc:.4f}')

    # Get predictions and compute confusion matrix
    y_pred = model.predict(X_test)
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