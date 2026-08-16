import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, brier_score_loss
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import label_binarize

def expected_calibration_error(y_true, y_prob, n_bins=10):
    ece = 0.0
    for k in range(y_prob.shape[1]):
        y_true_k = (y_true == k).astype(int)
        y_prob_k = y_prob[:, k]
        
        bins = np.linspace(0., 1., n_bins + 1)
        binids = np.searchsorted(bins[1:-1], y_prob_k)
        
        bin_sums = np.bincount(binids, weights=y_prob_k, minlength=len(bins))
        bin_true = np.bincount(binids, weights=y_true_k, minlength=len(bins))
        bin_total = np.bincount(binids, minlength=len(bins))
        
        nonzero = bin_total != 0
        prob_true = bin_true[nonzero] / bin_total[nonzero]
        prob_pred = bin_sums[nonzero] / bin_total[nonzero]
        
        ece += np.sum(np.abs(prob_true - prob_pred) * (bin_total[nonzero] / len(y_true)))
    return ece / y_prob.shape[1]

def main():
    print("Loading data...")
    dataset_path = os.path.join('data', 'tess_ml_arrays', 'tess_dataset_exp1.npz')
    data = np.load(dataset_path)
    X_raw = data['X']
    Y = data['y']
    
    # Contractual Baseline (V1) Preprocessing
    X_raw = np.nan_to_num(X_raw, nan=1.0, posinf=1.0, neginf=1.0)
    mean = np.mean(X_raw, axis=1, keepdims=True)
    std = np.std(X_raw, axis=1, keepdims=True)
    X_scaled = (X_raw - mean) / (std + 1e-8)
    X_scaled = np.nan_to_num(X_scaled, nan=0.0)
    
    X = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))
    
    print("Splitting test set...")
    _, X_val, _, y_val = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)
    
    print("Loading model...")
    model = load_model(os.path.join('data', 'models', 'exoplanet_cnn_v2_ternary.keras'))
    
    print("Predicting with MC Dropout (50 iterations)...")
    # MC Dropout
    n_iterations = 50
    predictions = []
    for _ in range(n_iterations):
        preds = model(X_val, training=True)
        predictions.append(preds.numpy())
        
    predictions = np.array(predictions) # (50, samples, 3)
    y_pred_probs = np.mean(predictions, axis=0) # (samples, 3)
    y_pred_uncert = np.std(predictions, axis=0) # (samples, 3)
    y_pred_classes = np.argmax(y_pred_probs, axis=1)
    
    print("Calculating metrics...")
    report = classification_report(y_val, y_pred_classes, output_dict=True)
    
    # ROC-AUC (OVR)
    y_val_bin = label_binarize(y_val, classes=[0, 1, 2])
    roc_auc = roc_auc_score(y_val_bin, y_pred_probs, multi_class='ovr')
    
    # ECE and Brier Score
    ece = expected_calibration_error(y_val, y_pred_probs)
    brier = np.mean([brier_score_loss(y_val_bin[:, k], y_pred_probs[:, k]) for k in range(3)])
    
    # Mean Epistemic Uncertainty
    mean_uncert = float(np.mean(y_pred_uncert))
    
    print("Plotting Confusion Matrix...")
    cm = confusion_matrix(y_val, y_pred_classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Noise', 'Planet', 'EB'],
                yticklabels=['Noise', 'Planet', 'EB'])
    plt.title('Ternary Confusion Matrix (Exp 1)')
    plt.ylabel('Actual True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    os.makedirs(os.path.join('docs', 'assets'), exist_ok=True)
    plt.savefig(os.path.join('docs', 'assets', 'v2_exp1_cm.png'))
    plt.close()
    
    print("Generating Report...")
    markdown_content = f"""# V2_EXP1_001

## Configuration
* **Model**: `exoplanet_cnn_v1.keras` (Frozen V1 Reference)
* **Dataset**: `tess_dataset_exp1.npz` (Test Split: 20%, Seed 42, Stratified)
* **Preprocessing Contract**: TERNARY_V1
  * Sectors: 1
  * Outlier Removal: OFF
  * SG Window: 401
  * Scaling: Z-Score (Mean/Std)
  * Clipping: OFF

## Global Metrics
* **ROC-AUC (OVR)**: {roc_auc:.4f}
* **Expected Calibration Error (ECE)**: {ece:.4f}
* **Brier Score**: {brier:.4f}
* **Mean MC Dropout Epistemic Uncertainty**: {mean_uncert:.4f}

## Per-Class Performance
| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **0 (Noise)** | {report['0']['precision']:.4f} | {report['0']['recall']:.4f} | {report['0']['f1-score']:.4f} | {report['0']['support']} |
| **1 (Planet)** | {report['1']['precision']:.4f} | {report['1']['recall']:.4f} | {report['1']['f1-score']:.4f} | {report['1']['support']} |
| **2 (EB)** | {report['2']['precision']:.4f} | {report['2']['recall']:.4f} | {report['2']['f1-score']:.4f} | {report['2']['support']} |
| **Accuracy** | - | - | **{report['accuracy']:.4f}** | {report['macro avg']['support']} |

## Artifacts
![Confusion Matrix](assets/v2_exp1_cm.png)

*(Note: Advanced dynamic metrics like SNR and Transit Depth will be mapped incrementally in future experimental scripts. This establishes the numerical floor for Exps 1-5).*
"""
    with open('docs/V2_EXP1_001.md', 'w') as f:
        f.write(markdown_content)
        
    print("Exp1 evaluation complete. Saved to docs/V2_EXP1_001.md")

if __name__ == '__main__':
    main()
