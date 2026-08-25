import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, brier_score_loss, roc_auc_score

# --- REPRODUCIBILITY CONSTANTS ---
RANDOM_SEED = 42
DATASET_PATH = 'data/tess_ml_arrays/tess_dataset_ternary.npz'
MODEL_PATH = 'data/models/exp6_native_mad_model.keras'

def get_mad_data(X):
    median = np.median(X, axis=1, keepdims=True)
    mad = np.median(np.abs(X - median), axis=1, keepdims=True)
    mad = np.where(mad == 0, 1e-8, mad)
    X_norm = (X - median) / (mad * 1.4826)
    return np.expand_dims(X_norm, axis=2)

def calculate_ece(y_true, y_prob, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    y_prob_max = np.max(y_prob, axis=1)
    y_pred = np.argmax(y_prob, axis=1)
    
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_prob_max > bin_lower) & (y_prob_max <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(y_true[in_bin] == y_pred[in_bin])
            avg_confidence_in_bin = np.mean(y_prob_max[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
    return ece

def mc_dropout_predict(model, X, num_passes=50):
    predictions = []
    print(f"Running {num_passes} MC Dropout passes...")
    for _ in range(num_passes):
        predictions.append(model(X, training=True).numpy())
    
    predictions = np.array(predictions)
    mean_probs = np.mean(predictions, axis=0)
    variance_probs = np.var(predictions, axis=0)
    mean_uncertainty = np.mean(variance_probs, axis=1)
    
    return mean_probs, mean_uncertainty

def main():
    print("==================================================")
    print("EXP 6: NATIVE MAD CNN FULL EVALUATION")
    print("==================================================")
    
    # 1. Load Data
    data = np.load(DATASET_PATH)
    X = data['X']
    Y = data['y']
    X = np.nan_to_num(X, nan=1.0, posinf=1.0, neginf=1.0)
    
    # 2. Strict split first
    _, X_val, _, y_val = train_test_split(X, Y, test_size=0.2, random_state=RANDOM_SEED, stratify=Y)
    
    # 3. Apply MAD scaling
    X_val_norm = get_mad_data(X_val)
    
    # 4. Load Model
    model = tf.keras.models.load_model(MODEL_PATH)
    
    # 5. Execute MC Dropout
    mean_probs, uncertainty = mc_dropout_predict(model, X_val_norm, num_passes=50)
    y_pred = np.argmax(mean_probs, axis=1)
    
    print("\n--- GLOBAL METRICS ---")
    
    # Classification Report
    print("\nPer-Class Classification Report:")
    target_names = ['Noise (0)', 'Planet (1)', 'EB (2)']
    print(classification_report(y_val, y_pred, target_names=target_names))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_val, y_pred))
    
    y_val_onehot = tf.keras.utils.to_categorical(y_val, num_classes=3)
    roc_auc = roc_auc_score(y_val_onehot, mean_probs, multi_class='ovr', average='weighted')
    brier = np.mean(np.sum((mean_probs - y_val_onehot)**2, axis=1))
    ece = calculate_ece(y_val, mean_probs)
    
    print(f"\nROC-AUC (Weighted OVR): {roc_auc:.4f}")
    print(f"Multiclass Brier Score: {brier:.4f}")
    print(f"Expected Calibration Error (ECE): {ece:.4f}")
    
    print("\n--- MC DROPOUT UNCERTAINTY ANALYSIS ---")
    print(f"Mean Predictive Uncertainty (Variance): {np.mean(uncertainty):.5f}")

if __name__ == '__main__':
    main()
