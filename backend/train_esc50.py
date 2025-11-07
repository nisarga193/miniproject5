import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import to_categorical
import json

def create_model(input_shape, num_classes):
    """Create a CNN model for audio classification."""
    model = Sequential([
        # First Convolutional Block
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        # Second Convolutional Block
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        # Third Convolutional Block
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        # Dense Layers
        Flatten(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    return model

def train_model():
    """Train the model on ESC-50 dataset."""
    # Load preprocessed data
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    X_train = np.load(os.path.join(data_dir, 'X_train.npy'))
    X_test = np.load(os.path.join(data_dir, 'X_test.npy'))
    y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
    y_test = np.load(os.path.join(data_dir, 'y_test.npy'))
    
    # Get unique classes
    classes = np.unique(y_train)
    num_classes = len(classes)
    
    # Save class mapping
    models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
    class_mapping = {i: cls for i, cls in enumerate(classes)}
    with open(os.path.join(models_dir, 'esc50_model.classes.json'), 'w') as f:
        json.dump(class_mapping, f)
    
    # Reshape data for CNN and convert labels to categorical
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
    
    y_train_cat = to_categorical(np.unique(y_train, return_inverse=True)[1])
    y_test_cat = to_categorical(np.unique(y_test, return_inverse=True)[1])
    
    # Create and compile model
    input_shape = (X_train.shape[1], X_train.shape[2], 1)
    model = create_model(input_shape, num_classes)
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Set up callbacks
    checkpoint = ModelCheckpoint(
        os.path.join(models_dir, 'esc50_model.h5'),
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        patience=10,
        restore_best_weights=True,
        mode='max'
    )
    
    # Train the model
    print("\nTraining model...")
    history = model.fit(
        X_train, y_train_cat,
        batch_size=32,
        epochs=100,
        validation_data=(X_test, y_test_cat),
        callbacks=[checkpoint, early_stopping],
        verbose=1
    )
    
    # Evaluate the model
    print("\nEvaluating model...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test_cat, verbose=0)
    print(f"Test accuracy: {test_accuracy*100:.2f}%")
    
    # Save final model
    final_model_path = os.path.join(models_dir, 'esc50_model_final.h5')
    model.save(final_model_path)
    print(f"Training completed! Model saved as '{final_model_path}'")

if __name__ == "__main__":
    train_model()