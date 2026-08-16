import numpy as np
import tensorflow as tf
import lightkurve as lk
import pandas as pd
import json
import argparse
import sys
import os

MODEL_PATH = 'data/models/exp5_reference_model.keras'
TARGETS_PATH = 'data/tess_positive_targets.json'
EB_TARGETS_PATH = 'data/tess_eb_targets.json'

def generate_sample(flattened_lc, period, epoch, num_bins=2000):
    folded_lc = flattened_lc.fold(period=period, epoch_time=epoch)
    binned_lc = folded_lc.bin(bins=num_bins)
    
    flux = binned_lc.flux.value
    
    if np.isnan(flux).any():
        flux = pd.Series(flux).interpolate(limit_direction='both').values
        
    if len(flux) != num_bins or np.isnan(flux).any():
        return None
        
    return flux

def get_target_params(tic_id):
    tic_id = tic_id.replace("TIC TIC ", "TIC ")
    
    with open(TARGETS_PATH, 'r') as f:
        pos = json.load(f)
    if tic_id in pos:
        return pos[tic_id]['period'], pos[tic_id]['epoch'], "Planet"
        
    with open(EB_TARGETS_PATH, 'r') as f:
        ebs = json.load(f)
    if tic_id in ebs:
        return ebs[tic_id]['period'], ebs[tic_id]['epoch'], "EB"
        
    print(f"Warning: {tic_id} not found in target lists. Defaulting to Noise label.")
    return 1.345, 1792.0, "Noise"

def mc_dropout_predict(model, X, num_passes=50):
    predictions = []
    for _ in range(num_passes):
        predictions.append(model(X, training=True).numpy())
    predictions = np.array(predictions)
    mean_probs = np.mean(predictions, axis=0)
    variance_probs = np.var(predictions, axis=0)
    mean_uncertainty = np.mean(variance_probs, axis=1)
    return mean_probs, mean_uncertainty

def run_dashboard(tic_id):
    print("==================================================")
    print("EXP 5 TRANSPARENCY DASHBOARD")
    print("==================================================")
    
    period, epoch, true_label = get_target_params(tic_id)
    
    print(f"Target: {tic_id}")
    print(f"True Label: {true_label} (Period: {period}, Epoch: {epoch})")
    print("Preprocessing: EXP5_PIPELINE_V1")
    
    # 1. Download
    search_result = lk.search_lightcurve(tic_id, author='SPOC', exptime=120)
    if len(search_result) == 0:
        print("Data not found.")
        return
    
    lc = search_result[0].download()
    sector = search_result[0].mission[0] if search_result[0].mission else "Unknown"
    print(f"Sector: {sector}")
    
    # 2. SG101
    print("SG window: 101")
    flattened_lc = lc.flatten(window_length=101)
    
    # 3. Binning
    print("Bins: 2000")
    flux_raw = generate_sample(flattened_lc, period, epoch)
    if flux_raw is None:
        print("Failed to bin properly.")
        return
        
    # 4. Z-Score Normalization Calculation
    flux_sanitized = np.nan_to_num(flux_raw, nan=1.0, posinf=1.0, neginf=1.0)
    mu = np.mean(flux_sanitized)
    sigma = np.std(flux_sanitized)
    epsilon = 1e-8
    
    flux_z = (flux_sanitized - mu) / (sigma + epsilon)
    
    print(f"mean: {mu:.5f}")
    print(f"std: {sigma:.5f}")
    
    print("\n--- Calculation Verification ---")
    print(f"z[0] = (x[0] - mean) / std")
    print(f"     = ({flux_sanitized[0]:.5f} - {mu:.5f}) / {sigma:.5f}")
    print(f"     = {flux_z[0]:.5f}")
    
    # 5. Prediction
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found at {MODEL_PATH}. Train it first.")
        return
        
    model = tf.keras.models.load_model(MODEL_PATH)
    X = np.expand_dims(flux_z, axis=(0, 2)) # shape (1, 2000, 1)
    
    mean_probs, uncertainty = mc_dropout_predict(model, X, num_passes=100)
    
    p_noise, p_planet, p_eb = mean_probs[0]
    pred_idx = np.argmax(mean_probs[0])
    labels = ["Noise", "Planet", "EB"]
    
    print("\n--- Model Inference ---")
    print(f"Predicted: {labels[pred_idx]}")
    print(f"P(Noise):  {p_noise:.4f}")
    print(f"P(Planet): {p_planet:.4f}")
    print(f"P(EB):     {p_eb:.4f}")
    
    unc_val = uncertainty[0]
    print(f"\nMC Uncertainty (Variance): {unc_val:.5f}")
    if unc_val < 0.005:
        print("Status: High Confidence / Low Uncertainty -> Strong Candidate")
    elif unc_val >= 0.005 and p_planet > 0.5:
        print("Status: High Confidence / High Uncertainty -> Requires Caution")
    else:
        print("Status: Standard Output")

if __name__ == '__main__':
    if len(sys.argv) > 1:
        run_dashboard(sys.argv[1])
    else:
        # Example Target
        run_dashboard("TIC 375654303")
