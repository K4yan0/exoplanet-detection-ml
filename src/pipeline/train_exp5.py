import os
import json
import hashlib
import platform
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from datetime import datetime


# --- REPRODUCIBILITY CONSTANTS ---
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

DATASET_PATH = 'data/tess_ml_arrays/tess_dataset_ternary.npz'
MODEL_SAVE_PATH = 'data/models/exp5_reference_model.keras'
MANIFEST_PATH = 'docs/exp5_reproducibility_manifest.json'

def get_file_hash(filepath):
    """Calculate SHA-256 hash of a file."""
    hasher = hashlib.sha256()
    with open(filepath, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def build_model(input_shape):
    """Build the V1 Baseline 1D CNN with MC Dropout using Functional API."""
    inputs = tf.keras.Input(shape=input_shape)
    
    x = tf.keras.layers.Conv1D(filters=32, kernel_size=5, activation='relu')(inputs)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
    
    x = tf.keras.layers.Conv1D(filters=64, kernel_size=5, activation='relu')(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
    
    x = tf.keras.layers.Conv1D(filters=128, kernel_size=5, activation='relu')(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
    
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    
    # MC Dropout: passing training=True explicitly keeps dropout active during inference
    x = tf.keras.layers.Dropout(0.5)(x, training=True) 
    
    outputs = tf.keras.layers.Dense(3, activation='softmax')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def main():
    print("==================================================")
    print("TRAINING EXP 5 REFERENCE PIPELINE")
    print("==================================================")
    
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}. Please ensure you are in the project root.")
    
    dataset_hash = get_file_hash(DATASET_PATH)
    print(f"Dataset Hash: {dataset_hash}")
    
    # 1. Load Data
    data = np.load(DATASET_PATH)
    X = data['X']
    Y = data['y']
    print(f"Loaded dataset: {X.shape[0]} samples. Target shape: {Y.shape}")
    
    # 2. Apply Z-score Normalization (Strictly Post-Binning per EXP5_PIPELINE_V1 contract)
    print("Applying Z-score normalization...")
    
    # CRITICAL V1 Baseline Contract: Sanitize Data
    X = np.nan_to_num(X, nan=1.0, posinf=1.0, neginf=1.0)
    
    # Add a small epsilon to avoid division by zero
    epsilon = 1e-8
    mean = np.mean(X, axis=1, keepdims=True)
    std = np.std(X, axis=1, keepdims=True)
    X_norm = (X - mean) / (std + epsilon)
    
    # Keras expects (batch, steps, channels)
    X_norm = np.expand_dims(X_norm, axis=2)
    
    # 3. Strict Train/Validation Split
    X_train, X_val, y_train, y_val = train_test_split(X_norm, Y, test_size=0.2, random_state=RANDOM_SEED, stratify=Y)
    
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Validation samples: {X_val.shape[0]}")
    
    # 4. Build and Train Model
    model = build_model(input_shape=(2000, 1))
    
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    history = model.fit(
        X_train, y_train,
        epochs=30,
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=[early_stop]
    )
    
    # 5. Save the Reference Model
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    model.save(MODEL_SAVE_PATH)
    model_hash = get_file_hash(MODEL_SAVE_PATH)
    
    # 6. Generate Reproducibility Manifest
    try:
        import importlib.metadata
        def get_version(pkg):
            try: return importlib.metadata.version(pkg)
            except: return 'unknown'
    except ImportError:
        def get_version(pkg): return 'unknown'
    
    manifest = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "contract": "EXP5_PIPELINE_V1",
        "random_seed": RANDOM_SEED,
        "dataset": {
            "path": DATASET_PATH,
            "sha256_hash": dataset_hash,
            "total_samples": X.shape[0],
            "train_samples": X_train.shape[0],
            "val_samples": X_val.shape[0],
            "normalization": "z-score (per-sample)"
        },
        "model": {
            "path": MODEL_SAVE_PATH,
            "sha256_hash": model_hash,
            "architecture": "1D CNN",
            "loss": "sparse_categorical_crossentropy",
            "optimizer": "adam",
            "learning_rate": 0.001
        },
        "environment": {
            "os": platform.platform(),
            "python_version": platform.python_version(),
            "tensorflow_version": get_version('tensorflow'),
            "lightkurve_version": get_version('lightkurve'),
            "numpy_version": get_version('numpy'),
            "pandas_version": get_version('pandas'),
            "scikit-learn_version": get_version('scikit-learn')
        },
        "performance": {
            "val_accuracy": float(np.max(history.history['val_accuracy'])),
            "val_loss": float(np.min(history.history['val_loss']))
        }
    }
    
    with open(MANIFEST_PATH, 'w') as f:
        json.dump(manifest, f, indent=4)
        
    print(f"\nModel saved to: {MODEL_SAVE_PATH}")
    print(f"Reproducibility manifest saved to: {MANIFEST_PATH}")

if __name__ == '__main__':
    main()
