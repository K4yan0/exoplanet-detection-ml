import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import os

def create_1d_cnn(input_length):
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
    dataset_path = os.path.join('data', 'tess_ml_arrays', 'aligned_v1_exp4_dataset.npz')
    if not os.path.exists(dataset_path):
        print(f"Error: Could not find aligned dataset at {dataset_path}")
        print("Please run `python src/pipeline/build_exp4_dataset.py` first.")
        return

    print(f"Loading Aligned Dataset from {dataset_path}...")
    data = np.load(dataset_path)
    X_v1_raw = data['X_v1']
    X_exp4_raw = data['X_exp4']
    Y = data['y']
    
    num_samples = X_v1_raw.shape[0]
    sequence_length = X_v1_raw.shape[1]
    
    print(f"Total Aligned Samples: {num_samples}")
    
    # Preprocessing Function
    def preprocess(X_raw):
        X_raw = np.nan_to_num(X_raw, nan=1.0, posinf=1.0, neginf=1.0)
        mean = np.mean(X_raw, axis=1, keepdims=True)
        std = np.std(X_raw, axis=1, keepdims=True)
        X_scaled = (X_raw - mean) / (std + 1e-8)
        X_scaled = np.nan_to_num(X_scaled, nan=0.0)
        return X_scaled.reshape((num_samples, sequence_length, 1))

    X_v1 = preprocess(X_v1_raw)
    X_exp4 = preprocess(X_exp4_raw)
    
    # 3. Split BOTH using the exact same random_state
    # This guarantees identical validation cohorts
    print("\nSplitting into strictly isolated 80% Train / 20% Val cohorts...")
    indices = np.arange(num_samples)
    train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42, stratify=Y)
    
    # Slice arrays
    X_v1_train, X_v1_val = X_v1[train_idx], X_v1[val_idx]
    X_exp4_train, X_exp4_val = X_exp4[train_idx], X_exp4[val_idx]
    y_train, y_val = Y[train_idx], Y[val_idx]
    
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
    
    # --- TRAIN V1 BASELINE ---
    print("\n" + "="*50)
    print("TRAINING V1 BASELINE (NO CLIPPING)")
    print("="*50)
    model_v1 = create_1d_cnn(sequence_length)
    model_v1.fit(
        X_v1_train, y_train,
        validation_data=(X_v1_val, y_val),
        epochs=30, batch_size=32, callbacks=[early_stop], verbose=1
    )
    
    # --- TRAIN EXP 4 ---
    print("\n" + "="*50)
    print("TRAINING EXP 4 (OUTLIER REMOVAL)")
    print("="*50)
    model_exp4 = create_1d_cnn(sequence_length)
    model_exp4.fit(
        X_exp4_train, y_train,
        validation_data=(X_exp4_val, y_val),
        epochs=30, batch_size=32, callbacks=[early_stop], verbose=1
    )
    
    # --- FINAL COMPARISON ---
    print("\n" + "="*50)
    print("STRICT COHORT EVALUATION RESULTS")
    print("="*50)
    
    preds_v1 = np.argmax(model_v1.predict(X_v1_val, verbose=0), axis=1)
    preds_exp4 = np.argmax(model_exp4.predict(X_exp4_val, verbose=0), axis=1)
    
    print("\n--- V1 Baseline (No Clipping) Classification Report ---")
    print(classification_report(y_val, preds_v1, target_names=['Noise', 'Planet', 'EB']))
    
    print("\n--- Exp 4 (Outlier Removal) Classification Report ---")
    print(classification_report(y_val, preds_exp4, target_names=['Noise', 'Planet', 'EB']))
    
    # Save models
    save_dir = os.path.join('data', 'models')
    os.makedirs(save_dir, exist_ok=True)
    model_v1.save(os.path.join(save_dir, 'aligned_v1.keras'))
    model_exp4.save(os.path.join(save_dir, 'aligned_exp4.keras'))
    print("\nSaved both strictly aligned models to data/models/")

if __name__ == '__main__':
    main()
