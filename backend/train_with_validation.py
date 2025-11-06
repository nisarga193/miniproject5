"""
Enhanced training script with validation metrics, confusion matrix,
and better data splitting to ensure model learns to distinguish classes.
"""
import os
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from model_utils import load_audio_file, extract_log_mel

def build_model(input_shape, n_classes):
    """Improved CNN architecture with more regularization"""
    model = Sequential([
        # First conv block
        Conv2D(32, (3,3), activation='relu', padding='same', input_shape=input_shape),
        BatchNormalization(),
        Conv2D(32, (3,3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2,2)),
        Dropout(0.25),

        # Second conv block
        Conv2D(64, (3,3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(64, (3,3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2,2)),
        Dropout(0.25),

        # Third conv block
        Conv2D(128, (3,3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(128, (3,3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2,2)),
        Dropout(0.25),

        # Dense layers
        Flatten(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(n_classes, activation='softmax')
    ])

    model.compile(
        optimizer=Adam(1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def gather_data(data_dir):
    """Load and preprocess audio files, ensuring balanced class distribution"""
    classes = sorted([d for d in os.listdir(data_dir) 
                     if os.path.isdir(os.path.join(data_dir,d))])
    
    X = []
    y = []
    files_used = []  # track which files were successfully processed
    
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
                X.append(mel)
                y.append(idx)
                files_used.append(path)
                n_processed += 1
            except Exception as e:
                print(f'Error processing {path}: {e}')
        
        print(f'Successfully processed {n_processed} files for class {cls}')
    
    X = np.array(X)
    X = np.expand_dims(X, -1)  # add channel dim
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
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--output', default='models/animal_cnn.h5')
    args = p.parse_args()

    X, y, classes, files_used = gather_data(args.data_dir)
    print('\nLoaded dataset with classes:', classes)

    if len(classes) < 2:
        raise SystemExit('Need >=2 classes in dataset')

    # Split preserving class distribution
    X_train, X_test, y_train, y_test, files_train, files_test = train_test_split(
        X, y, files_used, test_size=0.2, random_state=42, stratify=y
    )
    
    # Further split training into train/validation
    X_train, X_val, y_train, y_val, files_train, files_val = train_test_split(
        X_train, y_train, files_train, test_size=0.2, random_state=42, stratify=y_train
    )

    # Convert to categorical
    y_train_cat = to_categorical(y_train, num_classes=len(classes))
    y_val_cat = to_categorical(y_val, num_classes=len(classes))
    y_test_cat = to_categorical(y_test, num_classes=len(classes))

    print('\nSplit sizes:')
    print('Train:', X_train.shape[0])
    print('Validation:', X_val.shape[0])
    print('Test:', X_test.shape[0])

    model = build_model(X_train.shape[1:], len(classes))
    model.summary()

    # Create output directory
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Callbacks
    cb_cp = ModelCheckpoint(
        args.output, monitor='val_accuracy',
        save_best_only=True, verbose=1
    )
    cb_es = EarlyStopping(
        monitor='val_accuracy', patience=10,
        restore_best_weights=True, verbose=1
    )
    cb_rl = ReduceLROnPlateau(
        monitor='val_loss', factor=0.5,
        patience=5, verbose=1
    )

    # Train
    history = model.fit(
        X_train, y_train_cat,
        validation_data=(X_val, y_val_cat),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=[cb_cp, cb_es, cb_rl]
    )

    # Evaluate on test set
    print('\nEvaluating on test set...')
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

    # Save some misclassified examples for analysis
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