import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.core.xai import compute_gradcam

def load_zscore_and_mad(npz_path):
    data = np.load(npz_path)
    X_raw = data['X']
    Y = data['y']
    X_raw = np.nan_to_num(X_raw, nan=1.0, posinf=1.0, neginf=1.0)
    
    # Z-Score
    mean = np.mean(X_raw, axis=1, keepdims=True)
    std = np.std(X_raw, axis=1, keepdims=True)
    X_z = (X_raw - mean) / (std + 1e-8)
    X_z = np.nan_to_num(X_z, nan=0.0).reshape((X_raw.shape[0], 2000, 1))
    
    # MAD
    median = np.median(X_raw, axis=1, keepdims=True)
    mad = np.median(np.abs(X_raw - median), axis=1, keepdims=True)
    X_m = (X_raw - median) / (mad + 1e-8)
    X_m = np.nan_to_num(X_m, nan=0.0).reshape((X_raw.shape[0], 2000, 1))
    
    return X_z, X_m, Y

def main():
    print("Loading data...")
    X_z, X_m, Y = load_zscore_and_mad('data/tess_ml_arrays/tess_dataset_ternary.npz')
    
    _, X_val_z, _, y_val = train_test_split(X_z, Y, test_size=0.2, random_state=42, stratify=Y)
    _, X_val_m, _, _ = train_test_split(X_m, Y, test_size=0.2, random_state=42, stratify=Y)
    
    print("Loading model...")
    model = load_model('data/models/exoplanet_cnn_v2_ternary.keras')
    
    conv_layers = [l.name for l in model.layers if 'conv1d' in l.name]
    conv1_name = conv_layers[0] if len(conv_layers) >= 3 else model.layers[0].name
    
    # Select one of each class
    idx_noise = np.where(y_val == 0)[0][0]
    idx_planet = np.where(y_val == 1)[0][0]
    idx_eb = np.where(y_val == 2)[0][0]
    
    cases = [
        ("Noise", idx_noise, 0),
        ("Planet", idx_planet, 1),
        ("Eclipsing Binary", idx_eb, 2)
    ]
    
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle("Exp 2A Sanity Check: Z-Score vs MAD Attribution (Conv1)", fontsize=16)
    
    for i, (name, idx, target_class) in enumerate(cases):
        x_z = X_val_z[idx]
        x_m = X_val_m[idx]
        
        hm_z = compute_gradcam(model, x_z, conv1_name, target_class=target_class)
        hm_m = compute_gradcam(model, x_m, conv1_name, target_class=target_class)
        
        phases = np.linspace(-0.5, 0.5, 2000)
        
        # Plot Z-Score
        axes[i, 0].plot(phases, x_z.flatten(), color='blue', alpha=0.7)
        axes[i, 0].scatter(phases, x_z.flatten(), c=hm_z, cmap='Purples', s=10)
        axes[i, 0].set_title(f"Z-Score {name} (Idx {idx})")
        axes[i, 0].set_ylim(min(np.min(x_z), np.min(x_m))-0.5, max(np.max(x_z), np.max(x_m))+0.5)
        
        # Plot MAD
        axes[i, 1].plot(phases, x_m.flatten(), color='red', alpha=0.7)
        axes[i, 1].scatter(phases, x_m.flatten(), c=hm_m, cmap='Oranges', s=10)
        axes[i, 1].set_title(f"MAD {name} (Idx {idx})")
        axes[i, 1].set_ylim(axes[i,0].get_ylim())

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    os.makedirs('docs/images', exist_ok=True)
    plt.savefig('docs/images/exp2a_sanity_check.png')
    print("Saved to docs/images/exp2a_sanity_check.png")

if __name__ == '__main__':
    main()
