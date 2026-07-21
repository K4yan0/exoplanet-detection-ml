import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_and_preprocess_data():
    dataset_path = os.path.join('data', 'tess_ml_arrays', 'tess_dataset_full.npz')
    data = np.load(dataset_path)
    X_raw = data['X']
    Y = data['y']
    
    # EXACT same preprocessing as train_model.py to ensure the data looks identical
    X_raw = np.nan_to_num(X_raw, nan=1.0, posinf=1.0, neginf=1.0)
    mean = np.mean(X_raw, axis=1, keepdims=True)
    std = np.std(X_raw, axis=1, keepdims=True)
    X_scaled = (X_raw - mean) / (std + 1e-8)
    X_scaled = np.nan_to_num(X_scaled, nan=0.0)
    
    num_samples = X_scaled.shape[0]
    sequence_length = X_scaled.shape[1]
    X = X_scaled.reshape((num_samples, sequence_length, 1))
    
    # We use the EXACT same random seed (42) and stratify to isolate the exact 
    # same Validation set that the model was tested on!
    _, X_val, _, y_val = train_test_split(
        X, Y, test_size=0.2, random_state=42, stratify=Y
    )
    
    return X_val, y_val

def plot_confusion_matrix(y_true, y_pred_classes):
    cm = confusion_matrix(y_true, y_pred_classes)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negative (Noise)', 'Positive (Planet)'],
                yticklabels=['Negative (Noise)', 'Positive (Planet)'])
    plt.title('Confusion Matrix')
    plt.ylabel('Actual True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join('assets', 'confusion_matrix.png'))
    plt.close()

def plot_roc_curve(y_true, y_pred_probs):
    fpr, tpr, _ = roc_curve(y_true, y_pred_probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join('assets', 'roc_curve.png'))
    plt.close()

def analyze_worst_false_positives(X_val, y_val, y_pred_probs):
    # False Positive: True label is 0, but model predicted a high probability of 1
    # Find all actual negatives
    neg_indices = np.where(y_val == 0)[0]
    
    # Get probabilities for only the negative samples
    neg_probs = y_pred_probs[neg_indices].flatten()
    
    # Sort them by highest predicted probability
    worst_indices_relative = np.argsort(neg_probs)[::-1]
    
    print("\n--- Top 3 Worst False Positives ---")
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    
    for i in range(3):
        # We need the absolute index in X_val to grab the correct array
        rel_idx = worst_indices_relative[i]
        abs_idx = neg_indices[rel_idx]
        
        prob = neg_probs[rel_idx]
        
        print(f"FP #{i+1}: Probability of being a planet = {prob*100:.2f}%")
        
        # Plot the 1D array
        ax = axes[i]
        ax.plot(X_val[abs_idx].flatten(), color='red', alpha=0.7)
        ax.set_title(f"False Positive #{i+1} (Model Confidence: {prob*100:.1f}%)")
        ax.set_ylabel('Normalized Flux (Z-Score)')
        if i == 2:
            ax.set_xlabel('Phase Bins (0 to 2000)')
            
    plt.tight_layout()
    plt.savefig(os.path.join('assets', 'worst_false_positives.png'))
    plt.close()

def main():
    print("Loading Validation Data...")
    X_val, y_val = load_and_preprocess_data()
    
    # The new path for the saved model
    model_path = os.path.join('data', 'models', 'exoplanet_cnn_v1.keras')
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}.")
        print("Please rerun train_model.py first to save the model!")
        return
        
    print(f"Loading Model from {model_path}...")
    model = load_model(model_path)
    
    print("Running Predictions...")
    y_pred_probs = model.predict(X_val)
    y_pred_classes = (y_pred_probs > 0.5).astype(int).flatten()
    
    print("\nGenerating Confusion Matrix -> assets/confusion_matrix.png")
    plot_confusion_matrix(y_val, y_pred_classes)
    
    print("Generating ROC Curve -> assets/roc_curve.png")
    plot_roc_curve(y_val, y_pred_probs)
    
    print("Hunting for the worst False Positives -> assets/worst_false_positives.png")
    analyze_worst_false_positives(X_val, y_val, y_pred_probs)
    
    print("\n[SUCCESS] Error Analysis complete. Check the 'assets/' folder for the visualizations!")

if __name__ == '__main__':
    main()
