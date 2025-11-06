import os
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from model_utils import load_audio_file, extract_log_mel


def build_model(input_shape, n_classes):
    model = Sequential()
    model.add(Conv2D(32, (3,3), activation='relu', padding='same', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3,3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(n_classes, activation='softmax'))

    model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def gather_data(data_dir):
    classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir,d))])
    X = []
    y = []
    for idx, cls in enumerate(classes):
        cls_dir = os.path.join(data_dir, cls)
        for fname in os.listdir(cls_dir):
            if not fname.lower().endswith(('.wav', '.mp3', '.flac', '.ogg')):
                continue
            path = os.path.join(cls_dir, fname)
            try:
                y_audio = load_audio_file(path)
                mel = extract_log_mel(y_audio)
                X.append(mel)
                y.append(idx)
            except Exception as e:
                print('skip', path, 'err', e)
    X = np.array(X)
    X = np.expand_dims(X, -1)
    y = np.array(y)
    return X, y, classes


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', required=True)
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--output', default='models/animal_cnn.h5')
    args = p.parse_args()

    X, y, classes = gather_data(args.data_dir)
    print('Data shapes', X.shape, y.shape, 'classes', len(classes))

    if len(classes) < 2:
        raise SystemExit('Need >=2 classes in dataset')

    y_cat = to_categorical(y, num_classes=len(classes))
    X_train, X_val, y_train, y_val = train_test_split(X, y_cat, test_size=0.2, random_state=42, stratify=y)

    model = build_model(X_train.shape[1:], len(classes))
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    cb_cp = ModelCheckpoint(args.output, monitor='val_accuracy', save_best_only=True, verbose=1)
    cb_rl = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)

    model.fit(X_train, y_train, validation_data=(X_val,y_val), epochs=args.epochs, batch_size=args.batch_size, callbacks=[cb_cp, cb_rl])

    # save classes mapping
    import json
    with open(args.output + '.classes.json', 'w') as f:
        json.dump(classes, f)
    print('Saved model and classes to', args.output)

if __name__ == '__main__':
    main()
