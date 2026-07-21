import sys
import os
import numpy as np
import pandas as pd
import lightkurve as lk
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

def process_and_predict(star_id):
    print(f"\n--- Phase 3: Automated Exoplanet Inference ---")
    print(f"Target: {star_id}")
    
    # 1. Fetch Data
    print("1. Searching NASA MAST for SPOC light curves...")
    search_result = lk.search_lightcurve(star_id, mission='TESS', author='SPOC')
    
    if len(search_result) == 0:
        print(f"Error: No SPOC data found for {star_id}.")
        return
        
    print("   Downloading data...")
    lc = search_result[0].download()
    if lc is None:
        print(f"Error: Download failed.")
        return
        
    # 2. Flatten
    print("2. Flattening light curve (removing stellar variations)...")
    flattened_lc = lc.flatten(window_length=101)
    
    # 3. Find the Period and Epoch using BLS (Box-fitting Least Squares)
    # This is a critical astronomical step: We don't know the orbital period of a random star.
    # We use BLS to mathematically search for the strongest periodic dip, and use that to fold!
    print("3. Running BLS algorithm to find candidate transit period...")
    # By omitting the manual period grid, lightkurve automatically generates a highly 
    # optimized, ultra-fine frequency grid to avoid missing the true period by a fraction of a day.
    periodogram = flattened_lc.to_periodogram(method='bls')
    best_period = periodogram.period_at_max_power
    best_epoch = periodogram.transit_time_at_max_power
    
    print(f"   Candidate Period: {best_period.value:.4f} days")
    print(f"   Candidate Epoch:  {best_epoch.value:.4f}")
    
    # 4. Fold and Bin
    print("4. Folding and binning to exactly 2000 points...")
    folded_lc = flattened_lc.fold(period=best_period, epoch_time=best_epoch)
    binned_lc = folded_lc.bin(bins=2000)
    
    flux = binned_lc.flux.value
    
    if np.isnan(flux).any():
        flux = pd.Series(flux).interpolate(limit_direction='both').values
        
    if len(flux) != 2000 or np.isnan(flux).any():
        print("Error: Could not extract a clean 2000-point array.")
        return
        
    # 5. Normalize EXACTLY like the training data
    print("5. Normalizing (Z-score & Clipping)...")
    X_raw = np.array([flux])
    X_raw = np.nan_to_num(X_raw, nan=1.0, posinf=1.0, neginf=1.0)
    
    mean = np.mean(X_raw, axis=1, keepdims=True)
    std = np.std(X_raw, axis=1, keepdims=True)
    X_scaled = (X_raw - mean) / (std + 1e-8)
    X_scaled = np.nan_to_num(X_scaled, nan=0.0)
    # REMOVED np.clip to prevent the Clever Hans effect
    
    X = X_scaled.reshape((1, 2000, 1))
    
    # 6. Predict
    print("6. Loading Neural Network...")
    model_path = os.path.join('data', 'models', 'exoplanet_cnn_v1.keras')
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}.")
        return
        
    model = load_model(model_path)
    
    print("7. Running Prediction...")
    prediction = model.predict(X, verbose=0)[0][0]
    
    print("\n" + "="*40)
    if prediction > 0.5:
        print(f"🌟 PLANET DETECTED! 🌟")
        print(f"Confidence: {prediction * 100:.2f}%")
    else:
        print(f"❌ NO PLANET DETECTED ❌")
        print(f"Confidence (Noise): {(1 - prediction) * 100:.2f}%")
    print("="*40 + "\n")
    
    # 7. Plotting the result
    plt.figure(figsize=(10, 5))
    plt.plot(X_scaled[0].flatten(), color='blue' if prediction > 0.5 else 'red')
    plt.title(f"{star_id} - Predicted {'Planet' if prediction > 0.5 else 'Noise'} ({prediction*100:.1f}%)")
    plt.xlabel('Phase Bins (0 to 2000)')
    plt.ylabel('Normalized Flux (Z-Score)')
    plt.tight_layout()
    
    plot_path = os.path.join('assets', f'{star_id.replace(" ", "_")}_inference.png')
    plt.savefig(plot_path)
    print(f"Visualized candidate curve saved to: {plot_path}")

if __name__ == '__main__':
    # You can pass a star ID as a command line argument, otherwise it defaults to Pi Mensae
    star_input = sys.argv[1] if len(sys.argv) > 1 else "TIC 261136679"
    if not star_input.startswith("TIC"):
        star_input = f"TIC {star_input}"
        
    # Fix the double TIC bug just in case the user types "TIC TIC 123"
    star_input = star_input.replace("TIC TIC ", "TIC ")
        
    process_and_predict(star_input)
