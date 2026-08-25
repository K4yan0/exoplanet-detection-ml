import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# --- REPRODUCIBILITY CONSTANTS ---
RANDOM_SEED = 42
DATASET_PATH = 'data/tess_ml_arrays/tess_dataset_ternary.npz'
MODEL_ZSCORE_PATH = 'data/models/exp5_reference_model.keras'
MODEL_MAD_PATH = 'data/models/exp6_native_mad_model.keras'

def get_zscore_data(X):
    epsilon = 1e-8
    mean = np.mean(X, axis=1, keepdims=True)
    std = np.std(X, axis=1, keepdims=True)
    X_norm = (X - mean) / (std + epsilon)
    return np.expand_dims(X_norm, axis=2)

def get_mad_data(X):
    median = np.median(X, axis=1, keepdims=True)
    mad = np.median(np.abs(X - median), axis=1, keepdims=True)
    mad = np.where(mad == 0, 1e-8, mad)
    X_norm = (X - median) / (mad * 1.4826)
    return np.expand_dims(X_norm, axis=2)

def main():
    print("==================================================")
    print("EXP 6: THE 2x2 NORMALIZATION MATRIX")
    print("==================================================")
    
    # 1. Load Data
    data = np.load(DATASET_PATH)
    X_raw = data['X']
    Y = data['y']
    X_raw = np.nan_to_num(X_raw, nan=1.0, posinf=1.0, neginf=1.0)
    
    # We strictly evaluate on the validation set
    _, X_test_raw, _, y_test = train_test_split(X_raw, Y, test_size=0.2, random_state=RANDOM_SEED, stratify=Y)
    
    # 2. Normalize
    X_test_zscore = get_zscore_data(X_test_raw)
    X_test_mad = get_mad_data(X_test_raw)
    
    # 3. Load Models
    print("Loading Native Z-Score Model (Exp 5)...")
    model_z = tf.keras.models.load_model(MODEL_ZSCORE_PATH)
    
    print("Loading Native MAD Model (Exp 6)...")
    model_m = tf.keras.models.load_model(MODEL_MAD_PATH)
    
    # 4. Evaluate Matrix
    print("\nRunning inferences...")
    
    # Z-Model on Z-Data (Baseline)
    preds_zz = np.argmax(model_z.predict(X_test_zscore, verbose=0), axis=1)
    acc_zz = accuracy_score(y_test, preds_zz)
    
    # Z-Model on M-Data (Exp 2 - Representation Shift)
    preds_zm = np.argmax(model_z.predict(X_test_mad, verbose=0), axis=1)
    acc_zm = accuracy_score(y_test, preds_zm)
    
    # M-Model on M-Data (Exp 6 - Native MAD)
    preds_mm = np.argmax(model_m.predict(X_test_mad, verbose=0), axis=1)
    acc_mm = accuracy_score(y_test, preds_mm)
    
    # M-Model on Z-Data (Exp 6 - Cross Test)
    preds_mz = np.argmax(model_m.predict(X_test_zscore, verbose=0), axis=1)
    acc_mz = accuracy_score(y_test, preds_mz)
    
    print("\n" + "="*50)
    print("2x2 ACCURACY MATRIX")
    print("="*50)
    print(f"{'':25} | {'Z-Score Input':<15} | {'MAD Input':<15}")
    print("-" * 60)
    print(f"{'Native Z-Score CNN':25} | {acc_zz:.4f}          | {acc_zm:.4f}")
    print(f"{'Native MAD CNN':25} | {acc_mz:.4f}          | {acc_mm:.4f}")
    print("="*50)
    
    # F1 Scores
    f1_zz = f1_score(y_test, preds_zz, average='weighted')
    f1_zm = f1_score(y_test, preds_zm, average='weighted')
    f1_mm = f1_score(y_test, preds_mm, average='weighted')
    f1_mz = f1_score(y_test, preds_mz, average='weighted')
    
    print("\n2x2 WEIGHTED F1-SCORE MATRIX")
    print("-" * 60)
    print(f"{'Native Z-Score CNN':25} | {f1_zz:.4f}          | {f1_zm:.4f}")
    print(f"{'Native MAD CNN':25} | {f1_mz:.4f}          | {f1_mm:.4f}")
    print("="*50)

if __name__ == '__main__':
    main()
