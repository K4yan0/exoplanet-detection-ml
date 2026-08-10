import os
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from sklearn.model_selection import train_test_split
from scipy.optimize import minimize
from sklearn.metrics import log_loss

# 1. Expected Calibration Error (ECE) Calculation
def calculate_ece(y_true, y_prob, n_bins=10):
    """
    Calculates the Expected Calibration Error.
    y_true: Array of true class indices (Shape: N,)
    y_prob: Array of predicted probabilities for each class (Shape: N, num_classes)
    """
    # For multi-class, we usually calculate ECE based on the predicted class (the argmax)
    confidences = np.max(y_prob, axis=1)
    predictions = np.argmax(y_prob, axis=1)
    accuracies = (predictions == y_true)
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    bin_stats = []
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Determine which samples fall into this confidence bin
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            bin_stats.append({
                'bin_center': (bin_lower + bin_upper) / 2,
                'accuracy': accuracy_in_bin,
                'confidence': avg_confidence_in_bin,
                'count': np.sum(in_bin)
            })
            
    return ece, bin_stats

def plot_reliability_diagram(bin_stats_uncalibrated, bin_stats_calibrated, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for idx, (stats, title) in enumerate([(bin_stats_uncalibrated, "Uncalibrated"), (bin_stats_calibrated, "Calibrated (Temperature Scaling)")]):
        ax = axes[idx]
        confidences = [b['bin_center'] for b in stats]
        accuracies = [b['accuracy'] for b in stats]
        
        ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration')
        ax.bar(confidences, accuracies, width=0.1, align='center', alpha=0.7, color='#ff3366' if idx==0 else '#00ffcc', edgecolor='black')
        ax.plot(confidences, accuracies, marker='o', color='white')
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel('Mean Predicted Confidence')
        ax.set_ylabel('True Accuracy')
        ax.set_title(title)
        ax.grid(True, alpha=0.2)
        ax.set_facecolor('#1e293b')
        ax.legend()
        
    fig.patch.set_facecolor('#0f172a')
    for ax in axes:
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')
        for spine in ax.spines.values():
            spine.set_edgecolor('gray')
            
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close()

def main():
    # Load the Validation dataset exactly as we did in training
    dataset_path = os.path.join('data', 'tess_ml_arrays', 'tess_dataset_ternary.npz')
    data = np.load(dataset_path)
    X_raw = data['X']
    Y = data['y']
    
    # Sanitize & Normalize
    X_raw = np.nan_to_num(X_raw, nan=1.0, posinf=1.0, neginf=1.0)
    mean = np.mean(X_raw, axis=1, keepdims=True)
    std = np.std(X_raw, axis=1, keepdims=True)
    X_scaled = (X_raw - mean) / (std + 1e-8)
    X_scaled = np.nan_to_num(X_scaled, nan=0.0)
    X = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))
    
    _, X_val, _, y_val = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)
    
    # Load Model
    model_path = os.path.join('data', 'models', 'exoplanet_cnn_v2_ternary.keras')
    model = load_model(model_path)
    
    # We need the RAW LOGITS (the numbers before the softmax activation).
    # Since the last layer combines Dense+Softmax, we iterate manually:
    x_tensor = tf.convert_to_tensor(X_val, dtype=tf.float32)
    for layer in model.layers[:-1]:
        x_tensor = layer(x_tensor, training=False)
        
    last_layer = model.layers[-1]
    logits = tf.matmul(x_tensor, last_layer.kernel)
    if last_layer.bias is not None:
        logits = logits + last_layer.bias
        
    logits = logits.numpy()
    
    # Calculate Uncalibrated Probabilities
    def softmax(z):
        e_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return e_z / np.sum(e_z, axis=1, keepdims=True)
        
    y_prob_uncalibrated = softmax(logits)
    
    ece_uncalibrated, stats_uncalibrated = calculate_ece(y_val, y_prob_uncalibrated)
    print(f"Uncalibrated ECE: {ece_uncalibrated * 100:.2f}%")
    
    # Optimize Temperature (T)
    # We want to find a single scalar T that minimizes the Negative Log Likelihood
    def nll_objective(T, logits, labels):
        T = T[0]
        if T <= 0:
            return 1e9
        calibrated_probs = softmax(logits / T)
        # Add epsilon to prevent log(0)
        calibrated_probs = np.clip(calibrated_probs, 1e-9, 1 - 1e-9)
        # Sparse categorical crossentropy
        nll = -np.mean(np.log(calibrated_probs[np.arange(len(labels)), labels]))
        return nll

    print("Optimizing Temperature...")
    res = minimize(nll_objective, x0=[1.0], args=(logits, y_val), bounds=[(0.1, 10.0)])
    optimal_T = res.x[0]
    print(f"Optimal Temperature (T): {optimal_T:.4f}")
    
    # Calculate Calibrated Probabilities
    y_prob_calibrated = softmax(logits / optimal_T)
    ece_calibrated, stats_calibrated = calculate_ece(y_val, y_prob_calibrated)
    print(f"Calibrated ECE: {ece_calibrated * 100:.2f}%")
    
    # Save Temperature to file so the Flask app can use it
    calibration_data = {
        'temperature': float(optimal_T),
        'uncalibrated_ece': float(ece_uncalibrated),
        'calibrated_ece': float(ece_calibrated)
    }
    
    with open(os.path.join('data', 'models', 'calibration_params.json'), 'w') as f:
        json.dump(calibration_data, f, indent=4)
        
    # Draw Reliability Diagram
    os.makedirs('assets', exist_ok=True)
    plot_path = os.path.join('assets', 'reliability_diagram_ternary.png')
    plot_reliability_diagram(stats_uncalibrated, stats_calibrated, plot_path)
    print(f"Reliability Diagram saved to {plot_path}")

if __name__ == "__main__":
    main()
