import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import os

def create_1d_cnn(input_length):
    """
    Defines a modern 1D CNN architecture for exoplanet transit detection.
    """
    model = Sequential([
        Input(shape=(input_length, 1)),
        Conv1D(filters=16, kernel_size=5, activation='relu'),
        MaxPooling1D(pool_size=2),
        Conv1D(filters=32, kernel_size=5, activation='relu'),
        MaxPooling1D(pool_size=2),
        Conv1D(filters=64, kernel_size=5, activation='relu'),
        MaxPooling1D(pool_size=2),
        Flatten(), 
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(3, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam', 
        loss='sparse_categorical_crossentropy', 
        metrics=['accuracy']
    )
    
    return model

def main():
    dataset_path = os.path.join('data', 'tess_ml_arrays', 'tess_dataset_exp4.npz')
    
    if not os.path.exists(dataset_path):
        print(f"Error: Could not find dataset at {dataset_path}")
        return

    print(f"Loading Exp 4 dataset from {dataset_path}...")
    data = np.load(dataset_path)
    X_raw = data['X']
    Y = data['y']
    
    num_samples = X_raw.shape[0]
    sequence_length = X_raw.shape[1]
    
    X_raw = np.nan_to_num(X_raw, nan=1.0, posinf=1.0, neginf=1.0)
    
    mean = np.mean(X_raw, axis=1, keepdims=True)
    std = np.std(X_raw, axis=1, keepdims=True)
    X_scaled = (X_raw - mean) / (std + 1e-8)
    X_scaled = np.nan_to_num(X_scaled, nan=0.0)
    
    X = X_scaled.reshape((num_samples, sequence_length, 1))
    
    print(f"Total Dataset Shape: {X.shape}")
    print(f"Total Labels Shape: {Y.shape}")
    
    print("\nSplitting data into 80% Training and 20% Validation...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, Y, test_size=0.2, random_state=42, stratify=Y
    )
    
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Validation samples: {X_val.shape[0]}")
    
    print("\nInitializing 1D CNN model...")
    model = create_1d_cnn(input_length=sequence_length)
    
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    
    print("\n--- Starting Deep Learning Training ---")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=30,
        batch_size=32,
        callbacks=[early_stop],
        verbose=1
    )
    
    print("\n--- Final Validation Performance ---")
    loss, accuracy = model.evaluate(X_val, y_val, verbose=0)
    print(f"Validation Accuracy: {accuracy * 100:.2f}%")
    print(f"Validation Loss: {loss:.4f}")
    
    save_dir = os.path.join('data', 'models')
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, 'exoplanet_cnn_exp4.keras')
    model.save(model_path)
    print(f"\n[SUCCESS] Exp 4 Model successfully saved to {model_path}")

if __name__ == '__main__':
    main()
