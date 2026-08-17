import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import sys
from scipy.signal import find_peaks, peak_widths

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.core.xai import compute_gradcam

def load_and_scale(npz_path):
    data = np.load(npz_path)
    X_raw = data['X']
    Y = data['y']
    X_raw = np.nan_to_num(X_raw, nan=1.0, posinf=1.0, neginf=1.0)
    mean = np.mean(X_raw, axis=1, keepdims=True)
    std = np.std(X_raw, axis=1, keepdims=True)
    X_scaled = (X_raw - mean) / (std + 1e-8)
    X_scaled = np.nan_to_num(X_scaled, nan=0.0)
    X = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))
    return X, Y

def calculate_morphology(flux):
    flux = flux.flatten()
    # Depth is minimum value (since it's Z-scored, it's negative)
    depth = np.min(flux)
    
    # Baseline is mean of the out-of-transit regions (e.g. edges)
    baseline = np.mean(np.concatenate([flux[:500], flux[1500:]]))
    
    # Estimate width by finding the main dip (invert flux)
    inverted = -flux
    peaks, _ = find_peaks(inverted, height=np.max(inverted)*0.5)
    if len(peaks) > 0:
        main_peak = peaks[np.argmax(inverted[peaks])]
        widths = peak_widths(inverted, [main_peak], rel_height=0.5)
        duration = widths[0][0]
    else:
        duration = 0
        
    return depth, baseline, duration

def main():
    print("Loading V1 and Exp1 datasets...")
    X_v1, Y_v1 = load_and_scale('data/tess_ml_arrays/tess_dataset_ternary.npz')
    X_exp1, Y_exp1 = load_and_scale('data/tess_ml_arrays/tess_dataset_exp1.npz')
    
    _, X_val_v1, _, y_val = train_test_split(X_v1, Y_v1, test_size=0.2, random_state=42, stratify=Y_v1)
    _, X_val_exp1, _, _ = train_test_split(X_exp1, Y_exp1, test_size=0.2, random_state=42, stratify=Y_exp1)
    
    print("Loading model...")
    model = load_model('data/models/exoplanet_cnn_v2_ternary.keras')
    
    conv_layers = [l.name for l in model.layers if 'conv1d' in l.name]
    conv1_name = conv_layers[0] if len(conv_layers) >= 3 else model.layers[0].name
    
    eb_indices = np.where(y_val == 2)[0]
    
    shifts = []
    
    print("Calculating shifts for EBs...")
    for idx in eb_indices:
        x_v1 = X_val_v1[idx]
        x_exp1 = X_val_exp1[idx]
        
        hm_v1 = compute_gradcam(model, x_v1, conv1_name, target_class=2)
        hm_exp1 = compute_gradcam(model, x_exp1, conv1_name, target_class=2)
        
        # Ensure identical [0, 1] scaling check
        assert np.max(hm_v1) <= 1.0 and np.max(hm_exp1) <= 1.0
        
        mse = np.mean((np.array(hm_v1) - np.array(hm_exp1))**2)
        shifts.append((idx, mse, hm_v1, hm_exp1))
        
    # Sort by highest MSE
    shifts.sort(key=lambda x: x[1], reverse=True)
    
    print("\nTop 3 EBs with highest attribution shift:")
    top_3 = shifts[:3]
    
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle("Morphological Drivers of Attribution Shift (Top 3 Disrupted EBs)", fontsize=16)
    
    for i, (idx, mse, hm_v1, hm_exp1) in enumerate(top_3):
        x_v1 = X_val_v1[idx]
        x_exp1 = X_val_exp1[idx]
        
        d1, b1, w1 = calculate_morphology(x_v1)
        d2, b2, w2 = calculate_morphology(x_exp1)
        
        print(f"\nEB Index: {idx} | Shift MSE: {mse:.4f}")
        print(f"  V1 Depth: {d1:.2f} | Exp1 Depth: {d2:.2f} | Delta: {d2-d1:.2f}")
        print(f"  V1 Baseline: {b1:.2f} | Exp1 Baseline: {b2:.2f} | Delta: {b2-b1:.2f}")
        print(f"  V1 Width: {w1:.1f} | Exp1 Width: {w2:.1f} | Delta: {w2-w1:.1f}")
        
        phases = np.linspace(-0.5, 0.5, 2000)
        
        # Plot signals
        axes[i, 0].plot(phases, x_v1.flatten(), label='V1 (SG101)', alpha=0.7)
        axes[i, 0].plot(phases, x_exp1.flatten(), label='Exp1 (SG401)', alpha=0.7)
        axes[i, 0].set_title(f"Processed Signal (EB Index {idx})\nMorph Δ - Depth: {d2-d1:.2f} | Base: {b2-b1:.2f} | Width: {w2-w1:.1f}")
        axes[i, 0].legend()
        
        # Plot attribution
        axes[i, 1].plot(phases, hm_v1, label='V1 Conv1', color='blue', alpha=0.8)
        axes[i, 1].plot(phases, hm_exp1, label='Exp1 Conv1', color='red', alpha=0.8)
        axes[i, 1].fill_between(phases, hm_v1, hm_exp1, color='gray', alpha=0.2)
        axes[i, 1].set_title(f"Conv1 Attribution Shift (MSE: {mse:.4f})")
        axes[i, 1].legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    os.makedirs('docs/images', exist_ok=True)
    plt.savefig('docs/images/exp1a_morphology_validation.png')
    print("\nSaved morphology validation plot to docs/images/exp1a_morphology_validation.png")

if __name__ == '__main__':
    main()
