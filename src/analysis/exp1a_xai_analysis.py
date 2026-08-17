import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import sys

# Add src to path to import xai
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.core.xai import compute_gradcam, compute_integrated_gradients, compute_shap

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

def plot_case(name, x_v1, x_exp1, prob_v1, prob_exp1, heatmaps_v1, heatmaps_exp1, true_class):
    fig, axes = plt.subplots(5, 2, figsize=(16, 15), gridspec_kw={'height_ratios': [2, 1, 1, 1, 1]})
    fig.suptitle(f"XAI Attribution Analysis: {name}\nTrue Class: {true_class}", fontsize=16)
    
    phases = np.linspace(-0.5, 0.5, 2000)
    
    classes = ['Noise', 'Planet', 'EB']
    
    def format_probs(probs):
        return " | ".join([f"{c}: {p:.2f}" for c, p in zip(classes, probs)])
        
    v1_title = f"V1 (SG101)\nProbs: {format_probs(prob_v1)}"
    exp1_title = f"Exp1 (SG401)\nProbs: {format_probs(prob_exp1)}"
    
    # Raw processed signal
    axes[0, 0].plot(phases, x_v1.flatten(), color='blue')
    axes[0, 0].set_title(v1_title)
    axes[0, 0].set_ylim(min(np.min(x_v1), np.min(x_exp1)) - 0.5, max(np.max(x_v1), np.max(x_exp1)) + 0.5)
    
    axes[0, 1].plot(phases, x_exp1.flatten(), color='red')
    axes[0, 1].set_title(exp1_title)
    axes[0, 1].set_ylim(axes[0,0].get_ylim())
    
    # Heatmaps
    hm_keys = ['Grad-CAM Conv1', 'Grad-CAM Conv3', 'Integrated Gradients', 'SHAP']
    colors = ['Purples', 'Oranges', 'Greens', 'Reds']
    
    for i, hm_key in enumerate(hm_keys):
        row = i + 1
        # V1
        axes[row, 0].plot(phases, x_v1.flatten(), color='gray', alpha=0.3)
        axes[row, 0].scatter(phases, x_v1.flatten(), c=heatmaps_v1[hm_key], cmap=colors[i], s=5)
        axes[row, 0].set_ylabel(hm_key)
        axes[row, 0].set_ylim(axes[0,0].get_ylim())
        
        # Exp1
        axes[row, 1].plot(phases, x_exp1.flatten(), color='gray', alpha=0.3)
        axes[row, 1].scatter(phases, x_exp1.flatten(), c=heatmaps_exp1[hm_key], cmap=colors[i], s=5)
        axes[row, 1].set_ylim(axes[0,0].get_ylim())

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    os.makedirs('docs/images', exist_ok=True)
    safe_name = name.replace(" ", "_").replace("/", "_")
    plt.savefig(f"docs/images/exp1a_xai_{safe_name}.png")
    plt.close()

def main():
    print("Loading V1 and Exp1 datasets...")
    X_v1, Y_v1 = load_and_scale('data/tess_ml_arrays/tess_dataset_ternary.npz')
    X_exp1, Y_exp1 = load_and_scale('data/tess_ml_arrays/tess_dataset_exp1.npz')
    
    _, X_val_v1, _, y_val = train_test_split(X_v1, Y_v1, test_size=0.2, random_state=42, stratify=Y_v1)
    _, X_val_exp1, _, _ = train_test_split(X_exp1, Y_exp1, test_size=0.2, random_state=42, stratify=Y_exp1)
    
    print("Loading model...")
    model = load_model('data/models/exoplanet_cnn_v2_ternary.keras')
    
    # Layer names
    conv1_name = None
    conv3_name = None
    conv_layers = [l.name for l in model.layers if 'conv1d' in l.name]
    if len(conv_layers) >= 3:
        conv1_name = conv_layers[0]
        conv3_name = conv_layers[2]
    else:
        conv1_name = model.layers[0].name
        conv3_name = model.layers[2].name
        
    print(f"Using layers {conv1_name} and {conv3_name} for Grad-CAM")
    
    print("Predicting on validation sets...")
    prob_v1 = model.predict(X_val_v1, verbose=0)
    prob_exp1 = model.predict(X_val_exp1, verbose=0)
    
    pred_v1 = np.argmax(prob_v1, axis=1)
    pred_exp1 = np.argmax(prob_exp1, axis=1)
    
    # Find indices for the 4 cases
    case1_idx = np.where((y_val == 1) & (pred_v1 == 1) & (pred_exp1 == 1))[0]
    case2_idx = np.where((y_val == 1) & (pred_v1 != 1) & (pred_exp1 == 1))[0]
    case3_idx = np.where((y_val == 2) & (pred_v1 == 2) & (pred_exp1 != 2))[0]
    case4_idx = np.where((y_val == 0) & (pred_v1 == 0) & (pred_exp1 != 0))[0]
    
    cases = {
        'Case 1_ Planet Agreed': case1_idx[0] if len(case1_idx) > 0 else None,
        'Case 2_ Planet Corrected': case2_idx[0] if len(case2_idx) > 0 else None,
        'Case 3_ EB Lost': case3_idx[0] if len(case3_idx) > 0 else None,
        'Case 4_ Noise Confused': case4_idx[0] if len(case4_idx) > 0 else None,
    }
    
    for name, idx in cases.items():
        if idx is None:
            print(f"Skipping {name} - no matching cases found.")
            continue
            
        print(f"Generating XAI for {name} (Index {idx})...")
        x_v1 = X_val_v1[idx]
        x_exp1 = X_val_exp1[idx]
        target_class = y_val[idx]
        
        heatmaps_v1 = {}
        heatmaps_exp1 = {}
        
        print("  -> Computing Grad-CAM Conv1...")
        heatmaps_v1['Grad-CAM Conv1'] = compute_gradcam(model, x_v1, conv1_name, target_class)
        heatmaps_exp1['Grad-CAM Conv1'] = compute_gradcam(model, x_exp1, conv1_name, target_class)
        
        print("  -> Computing Grad-CAM Conv3...")
        heatmaps_v1['Grad-CAM Conv3'] = compute_gradcam(model, x_v1, conv3_name, target_class)
        heatmaps_exp1['Grad-CAM Conv3'] = compute_gradcam(model, x_exp1, conv3_name, target_class)
        
        print("  -> Computing Integrated Gradients...")
        heatmaps_v1['Integrated Gradients'] = compute_integrated_gradients(model, x_v1, target_class=target_class)
        heatmaps_exp1['Integrated Gradients'] = compute_integrated_gradients(model, x_exp1, target_class=target_class)
        
        print("  -> Computing SHAP...")
        heatmaps_v1['SHAP'] = compute_shap(model, x_v1, target_class=target_class)
        heatmaps_exp1['SHAP'] = compute_shap(model, x_exp1, target_class=target_class)
        
        plot_case(name, x_v1, x_exp1, prob_v1[idx], prob_exp1[idx], heatmaps_v1, heatmaps_exp1, target_class)

    print("--- Quantitative Attribution Shift ---")
    print("Calculating Grad-CAM Conv1 shift across all EBs...")
    eb_indices = np.where(y_val == 2)[0]
    
    total_shift = 0.0
    misclassified_shift = 0.0
    correct_shift = 0.0
    
    misclassified_count = 0
    correct_count = 0
    
    for idx in eb_indices:
        x_v1 = X_val_v1[idx]
        x_exp1 = X_val_exp1[idx]
        
        hm_v1 = compute_gradcam(model, x_v1, conv1_name, target_class=2)
        hm_exp1 = compute_gradcam(model, x_exp1, conv1_name, target_class=2)
        
        mse = np.mean((np.array(hm_v1) - np.array(hm_exp1))**2)
        
        if pred_v1[idx] == 2 and pred_exp1[idx] != 2:
            misclassified_shift += mse
            misclassified_count += 1
        elif pred_v1[idx] == 2 and pred_exp1[idx] == 2:
            correct_shift += mse
            correct_count += 1
            
    if misclassified_count > 0:
        print(f"Average Conv1 MSE for EBs LOST by SG401: {misclassified_shift/misclassified_count:.4f}")
    if correct_count > 0:
        print(f"Average Conv1 MSE for EBs KEPT by SG401: {correct_shift/correct_count:.4f}")

    # Planet shifts
    planet_indices = np.where(y_val == 1)[0]
    planet_gained_shift = 0.0
    planet_gained_count = 0
    planet_kept_shift = 0.0
    planet_kept_count = 0
    
    for idx in planet_indices:
        x_v1 = X_val_v1[idx]
        x_exp1 = X_val_exp1[idx]
        
        hm_v1 = compute_gradcam(model, x_v1, conv1_name, target_class=1)
        hm_exp1 = compute_gradcam(model, x_exp1, conv1_name, target_class=1)
        
        mse = np.mean((np.array(hm_v1) - np.array(hm_exp1))**2)
        
        if pred_v1[idx] != 1 and pred_exp1[idx] == 1:
            planet_gained_shift += mse
            planet_gained_count += 1
        elif pred_v1[idx] == 1 and pred_exp1[idx] == 1:
            planet_kept_shift += mse
            planet_kept_count += 1
            
    if planet_gained_count > 0:
        print(f"Average Conv1 MSE for Planets GAINED by SG401: {planet_gained_shift/planet_gained_count:.4f}")
    if planet_kept_count > 0:
        print(f"Average Conv1 MSE for Planets KEPT by SG401: {planet_kept_shift/planet_kept_count:.4f}")
        
    print("Done.")

if __name__ == '__main__':
    main()
