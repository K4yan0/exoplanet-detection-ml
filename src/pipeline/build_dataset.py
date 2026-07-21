import lightkurve as lk
import numpy as np
import pandas as pd
import os
import json

def generate_sample(flattened_lc, period, epoch, num_bins=2000):
    """
    Folds and bins an already flattened light curve.
    Returns a 1D numpy array of length `num_bins`.
    """
    folded_lc = flattened_lc.fold(period=period, epoch_time=epoch)
    binned_lc = folded_lc.bin(bins=num_bins)
    
    flux = binned_lc.flux.value
    
    # Interpolate any NaNs
    if np.isnan(flux).any():
        flux = pd.Series(flux).interpolate(limit_direction='both').values
        
    # Check if interpolation left any NaNs (e.g., if entire array was NaN)
    if np.isnan(flux).any():
        return None
        
    return flux

def main():
    # 1. Load the JSON targets
    json_path = os.path.join('data', 'tess_positive_targets.json')
    if not os.path.exists(json_path):
        print(f"Error: Could not find {json_path}")
        print("Make sure you are running from the project root!")
        return
        
    with open(json_path, 'r') as f:
        targets = json.load(f)
        
    print(f"Loaded {len(targets)} targets from {json_path}")
    
    X_list = []
    y_list = []
    
    print("\nStarting Mass Dataset Generation...")
    
    # 2. Loop through all stars
    for count, (star_id, params) in enumerate(targets.items(), 1):
        # Fix the "TIC TIC" formatting bug from the fetch script
        clean_star_id = star_id.replace("TIC TIC ", "TIC ")
        
        print(f"[{count}/{len(targets)}] Processing {clean_star_id}...")
        
        try:
            # Step A: Fetch once
            search_result = lk.search_lightcurve(clean_star_id, mission='TESS', author='SPOC')
            if len(search_result) == 0:
                print(f"  [!] No SPOC data found. Skipping.")
                continue
                
            # Download the first light curve found
            lc = search_result[0].download()
            if lc is None:
                continue
            
            # Step B: Flatten once (This is computationally heavy, so doing it once per star is brilliant)
            flattened_lc = lc.flatten(window_length=101)
            
            true_period = params['period']
            true_epoch = params['epoch']
            
            # Step C: Generate POSITIVE sample (Label 1)
            flux_pos = generate_sample(flattened_lc, true_period, true_epoch)
            if flux_pos is not None:
                X_list.append(flux_pos)
                y_list.append(1)
                
            # Step D: Generate NEGATIVE sample (Label 0)
            # Mangle the period and shift the epoch
            mangled_period = true_period * 1.345
            shifted_epoch = true_epoch + (true_period * 0.5)
            
            flux_neg = generate_sample(flattened_lc, mangled_period, shifted_epoch)
            if flux_neg is not None:
                X_list.append(flux_neg)
                y_list.append(0)
                
            print(f"  [SUCCESS] Added 1 Positive and 1 Negative sample.")
            
        except Exception as e:
            print(f"  [ERROR] Failed to process {star_id}: {e}")

    # Convert to final Numpy matrices
    X = np.array(X_list)
    Y = np.array(y_list)
    
    print("\n--- Final Dataset Summary ---")
    print(f"X Matrix Shape: {X.shape}")
    print(f"Y Vector Shape: {Y.shape}")
    
    # Save the dataset
    save_dir = os.path.join('data', 'tess_ml_arrays')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'tess_dataset_full.npz')
    
    np.savez(save_path, X=X, y=Y)
    print(f"\nMass Dataset successfully saved to: {save_path}")

if __name__ == '__main__':
    main()
