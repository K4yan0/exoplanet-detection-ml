import lightkurve as lk
import numpy as np
import pandas as pd
import os
import json
import warnings
warnings.filterwarnings('ignore')

def generate_sample(flattened_lc, period, epoch, num_bins=2000):
    """Folds and bins an already flattened light curve."""
    folded_lc = flattened_lc.fold(period=period, epoch_time=epoch)
    binned_lc = folded_lc.bin(bins=num_bins)
    
    flux = binned_lc.flux.value
    
    if np.isnan(flux).any():
        flux = pd.Series(flux).interpolate(limit_direction='both').values
        
    if len(flux) != num_bins:
        return None
        
    if np.isnan(flux).any():
        return None
        
    return flux

def main():
    json_path_pos = os.path.join('data', 'tess_positive_targets.json')
    json_path_eb = os.path.join('data', 'tess_eb_targets.json')
        
    with open(json_path_pos, 'r') as f:
        targets_pos = json.load(f)
    with open(json_path_eb, 'r') as f:
        targets_eb = json.load(f)
        
    print(f"Loaded {len(targets_pos)} Planets and {len(targets_eb)} EBs")
    
    # We will balance the dataset: roughly 300 of each
    # Noise will be generated from Planets (mangled period/epoch)
    
    X_list = []
    y_list = []
    
    # Process Planets & Generate Noise
    print("\n--- Processing PLANETS & NOISE (Label 1 & Label 0) ---")
    pos_count = 0
    max_samples = 300
    
    for star_id, params in targets_pos.items():
        if pos_count >= max_samples:
            break
            
        clean_star_id = star_id.replace("TIC TIC ", "TIC ")
        print(f"Planet [{pos_count+1}/{max_samples}] Processing {clean_star_id}...")
        
        try:
            search_result = lk.search_lightcurve(clean_star_id, mission='TESS', author='SPOC')
            if len(search_result) == 0:
                continue
                
            lc = search_result[0].download()
            if lc is None:
                continue
            
            flattened_lc = lc.flatten(window_length=101)
            
            true_period = params['period']
            true_epoch = params['epoch']
            
            # Label 1: Planet
            flux_pos = generate_sample(flattened_lc, true_period, true_epoch)
            if flux_pos is not None:
                X_list.append(flux_pos)
                y_list.append(1)
                
                # Label 0: Noise
                mangled_period = true_period * 1.345
                shifted_epoch = true_epoch + (true_period * 0.5)
                flux_neg = generate_sample(flattened_lc, mangled_period, shifted_epoch)
                
                if flux_neg is not None:
                    X_list.append(flux_neg)
                    y_list.append(0)
                    
                pos_count += 1
        except Exception as e:
            pass

    # Process Eclipsing Binaries
    print("\n--- Processing ECLIPSING BINARIES (Label 2) ---")
    eb_count = 0
    
    for star_id, params in targets_eb.items():
        if eb_count >= max_samples:
            break
            
        clean_star_id = star_id
        print(f"EB [{eb_count+1}/{max_samples}] Processing {clean_star_id}...")
        
        try:
            search_result = lk.search_lightcurve(clean_star_id, mission='TESS', author='SPOC')
            if len(search_result) == 0:
                continue
                
            lc = search_result[0].download()
            if lc is None:
                continue
            
            flattened_lc = lc.flatten(window_length=101)
            
            true_period = params['period']
            true_epoch = params['epoch']
            
            # Label 2: Eclipsing Binary
            flux_eb = generate_sample(flattened_lc, true_period, true_epoch)
            if flux_eb is not None:
                X_list.append(flux_eb)
                y_list.append(2)
                eb_count += 1
        except Exception as e:
            pass

    # Finalize
    X = np.array(X_list)
    Y = np.array(y_list)
    
    print("\n--- Final Dataset Summary ---")
    print(f"X Matrix Shape: {X.shape}")
    print(f"Y Vector Shape: {Y.shape}")
    print(f"Label 0 (Noise): {np.sum(Y==0)}")
    print(f"Label 1 (Planet): {np.sum(Y==1)}")
    print(f"Label 2 (EB): {np.sum(Y==2)}")
    
    save_dir = os.path.join('data', 'tess_ml_arrays')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'tess_dataset_ternary.npz')
    
    np.savez(save_path, X=X, y=Y)
    print(f"\nTernary Dataset successfully saved to: {save_path}")

if __name__ == '__main__':
    main()
