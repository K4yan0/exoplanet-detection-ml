import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
import os

def create_1d_cnn(input_length):
    """
    Defines a modern 1D CNN architecture for exoplanet transit detection.
    """
    model = Sequential([
        # Block 1: Detect basic local features (small drops in flux)
        Conv1D(filters=16, kernel_size=5, activation='relu', input_shape=(input_length, 1)),
        MaxPooling1D(pool_size=2),
        
        # Block 2: Detect larger structural patterns (the 'U' shape)
        Conv1D(filters=32, kernel_size=5, activation='relu'),
        MaxPooling1D(pool_size=2),
        
        # Block 3: High-level feature extraction
        Conv1D(filters=64, kernel_size=5, activation='relu'),
        MaxPooling1D(pool_size=2),
        
        # Flatten for the fully connected layers
        Flatten(),
        
        # Dense classification head
        Dense(64, activation='relu'),
        Dropout(0.3),  # Regularization to prevent overfitting when we scale up
        Dense(1, activation='sigmoid')  # Binary classification: 1 (Planet) or 0 (Non-Planet)
    ])
    
    # Compile the model using Adam and binary crossentropy
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def main():
    # 1. Load the dataset
    # We assume the script is executed from the project root directory
    dataset_path = os.path.join('data', 'tess_ml_arrays', 'tess_dataset_v1.npz')
    
    if not os.path.exists(dataset_path):
        print(f"Error: Could not find dataset at {dataset_path}")
        print("Please make sure you are running this script from the project root folder!")
        return

    print(f"Loading dataset from {dataset_path}...")
    data = np.load(dataset_path)
    X_raw = data['X']
    Y = data['y']
    
    print(f"Original X shape: {X_raw.shape}")
    print(f"Original Y shape: {Y.shape}")
    
    # 2. Reshape X for the 1D CNN
    # Keras Conv1D requires input shape: (batch_size, sequence_length, channels)
    # Since we have a single flux channel, channels = 1
    num_samples = X_raw.shape[0]
    sequence_length = X_raw.shape[1]
    
    X = X_raw.reshape((num_samples, sequence_length, 1))
    print(f"Reshaped X for CNN (batch, sequence, channels): {X.shape}")
    
    # 3. Initialize the model
    print("\nInitializing 1D CNN model...")
    model = create_1d_cnn(input_length=sequence_length)
    model.summary()
    
    # 4. Dummy Training Loop
    print("\n--- Starting Dummy Training (Validating Tensor Plumbing) ---")
    # We train for 5 epochs just to ensure the plumbing (shapes, dimensions, loss) works.
    history = model.fit(
        X, Y,
        epochs=5,
        batch_size=2,  # Tiny batch size because we only have 3 samples
        verbose=1
    )
    
    print("\n[SUCCESS] Tensor plumbing validated! The model compiles and trains perfectly.")

if __name__ == '__main__':
    main()
