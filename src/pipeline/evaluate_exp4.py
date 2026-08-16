import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_curve, auc, brier_score_loss
import os

def calculate_ece(y_true, y_prob, n_bins=10):
    confidences = np.max(y_prob, axis=1)
    predictions = np.argmax(y_prob, axis=1)
    accuracies = (predictions == y_true)
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
    return ece

def main():
    dataset_path = os.path.join('data', 'tess_ml_arrays', 'tess_dataset_exp4.npz')
    if not os.path.exists(dataset_path):
        print(f"Dataset not found: {dataset_path}")
        return
        
    data = np.load(dataset_path)
    X_raw = data['X']
    Y = data['y']
    
    X_raw = np.nan_to_num(X_raw, nan=1.0, posinf=1.0, neginf=1.0)
    mean = np.mean(X_raw, axis=1, keepdims=True)
    std = np.std(X_raw, axis=1, keepdims=True)
    X_scaled = (X_raw - mean) / (std + 1e-8)
    X_scaled = np.nan_to_num(X_scaled, nan=0.0)
    
    num_samples = X_scaled.shape[0]
    sequence_length = X_scaled.shape[1]
    X = X_scaled.reshape((num_samples, sequence_length, 1))
    
    _, X_val, _, y_val = train_test_split(
        X, Y, test_size=0.2, random_state=42, stratify=Y
    )
    
    model_path = os.path.join('data', 'models', 'exoplanet_cnn_exp4.keras')
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return
        
    model = load_model(model_path)
    y_pred_probs = model.predict(X_val, verbose=0)
    y_pred_classes = np.argmax(y_pred_probs, axis=1)
    
    print("\n--- Exp 4 CLASSIFICATION REPORT ---")
    print(classification_report(y_val, y_pred_classes, target_names=['Noise (0)', 'Planet (1)', 'EB (2)']))
    
    # ROC-AUC (Macro)
    y_val_onehot = tf.keras.utils.to_categorical(y_val, num_classes=3)
    fpr, tpr, _ = roc_curve(y_val_onehot.ravel(), y_pred_probs.ravel())
    roc_auc = auc(fpr, tpr)
    print(f"Macro ROC-AUC: {roc_auc:.4f}")
    
    # Brier Score (approximated for multiclass via flattened)
    brier = brier_score_loss(y_val_onehot.ravel(), y_pred_probs.ravel())
    print(f"Brier Score: {brier:.4f}")
    
    # ECE
    ece = calculate_ece(y_val, y_pred_probs)
    print(f"Expected Calibration Error (ECE): {ece:.4f}")
    
    # MC Uncertainty
    mc_uncertainty = np.mean(np.sum(-y_pred_probs * np.log(y_pred_probs + 1e-9), axis=1))
    print(f"MC Uncertainty (Entropy): {mc_uncertainty:.4f}")
    
    # Generate Artifact
    artifact_path = os.path.join('docs', 'V2_EXP4_001.md')
    with open(artifact_path, 'w') as f:
        f.write("# V2_EXP4_001: Outlier Removal\n\n")
        f.write("## Hypothesis\n")
        f.write("Removing outliers (stellar flares, instrumental spikes) before folding and binning will reduce noise and improve the CNN's ability to learn transit morphology.\n\n")
        f.write("## Metrics\n")
        f.write(f"- Macro ROC-AUC: {roc_auc:.4f}\n")
        f.write(f"- Brier Score: {brier:.4f}\n")
        f.write(f"- Expected Calibration Error (ECE): {ece:.4f}\n")
        f.write(f"- MC Uncertainty: {mc_uncertainty:.4f}\n\n")
        f.write("## Classification Report\n")
        f.write("```text\n")
        f.write(classification_report(y_val, y_pred_classes, target_names=['Noise (0)', 'Planet (1)', 'EB (2)']))
        f.write("\n```\n")
        
    print(f"\nArtifact saved to {artifact_path}")

if __name__ == '__main__':
    main()
