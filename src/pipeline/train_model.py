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
        # Explicit Input layer (This fixes the Keras 3 warning from our dummy test!)
        Input(shape=(input_length, 1)),
        
        # Block 1: Detect basic local features
        Conv1D(filters=16, kernel_size=5, activation='relu'),
        MaxPooling1D(pool_size=2),
        
        # Block 2: Detect larger structural patterns (the 'U' shape)
        Conv1D(filters=32, kernel_size=5, activation='relu'),
        MaxPooling1D(pool_size=2),
        
        # Block 3: High-level feature extraction
        Conv1D(filters=64, kernel_size=5, activation='relu'),
        MaxPooling1D(pool_size=2),
        
        Flatten(),
        
        Dense(64, activation='relu'),
        Dropout(0.3),  # Vital for preventing overfitting on our 1347 samples
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def main():
    # 1. Load the MASSIVE dataset
    dataset_path = os.path.join('data', 'tess_ml_arrays', 'tess_dataset_full.npz')
    
    if not os.path.exists(dataset_path):
        print(f"Error: Could not find dataset at {dataset_path}")
        return

    print(f"Loading full dataset from {dataset_path}...")
    data = np.load(dataset_path)
    X_raw = data['X']
    Y = data['y']
    
    # 2. Reshape and Normalize X for the 1D CNN
    num_samples = X_raw.shape[0]
    sequence_length = X_raw.shape[1]
    
    # CRITICAL FIX 2: Sanitize Data
    # Some sneaky NaNs or Infs (infinite values) must have survived the interpolation in build_dataset.py.
    # If even a single NaN enters the network, the gradients explode and loss becomes 'nan'.
    X_raw = np.nan_to_num(X_raw, nan=1.0, posinf=1.0, neginf=1.0)
    
    # CRITICAL FIX: Z-Score Normalization
    # The transit dips are tiny (e.g., a drop to 0.998). We need to standardize every single 
    # light curve to have a mean of 0 and a standard deviation of 1.
    mean = np.mean(X_raw, axis=1, keepdims=True)
    std = np.std(X_raw, axis=1, keepdims=True)
    X_scaled = (X_raw - mean) / (std + 1e-8)
    
    # One more safety net just in case standard deviation was completely 0
    X_scaled = np.nan_to_num(X_scaled, nan=0.0)
    
    X = X_scaled.reshape((num_samples, sequence_length, 1))
    
    print(f"Total Dataset Shape: {X.shape}")
    print(f"Total Labels Shape: {Y.shape}")
    
    # 3. Split into Training and Validation Sets (80% Train, 20% Test)
    # We use 'stratify=Y' to ensure the 50/50 ratio of Planets/Non-Planets is maintained in both sets.
    print("\nSplitting data into 80% Training and 20% Validation...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, Y, test_size=0.2, random_state=42, stratify=Y
    )
    
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Validation samples: {X_val.shape[0]}")
    
    # 4. Initialize the model
    print("\nInitializing 1D CNN model...")
    model = create_1d_cnn(input_length=sequence_length)
    
    # 5. Define Early Stopping 
    # This automatically stops training if the model stops learning, preventing it from just memorizing the data.
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    
    # 6. Real Training Loop
    print("\n--- Starting Deep Learning Training ---")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=30,          # Increased for real training
        batch_size=32,      # Standard batch size for 1300+ samples
        callbacks=[early_stop],
        verbose=1
    )
    
    # 7. Final Evaluation
    print("\n--- Final Validation Performance ---")
    loss, accuracy = model.evaluate(X_val, y_val, verbose=0)
    print(f"Validation Accuracy: {accuracy * 100:.2f}%")
    print(f"Validation Loss: {loss:.4f}")

if __name__ == '__main__':
    main()
