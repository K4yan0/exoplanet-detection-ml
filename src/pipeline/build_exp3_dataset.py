import lightkurve as lk
import numpy as np
import pandas as pd
import os
import json
import glob
import warnings
warnings.filterwarnings('ignore')

CACHE_DIR = os.path.expanduser('~/.cache/lightkurve/mastDownload/TESS')

def get_stitched_lc(tic_str, max_sectors=5):
    tic_num = tic_str.replace("TIC ", "").strip()
    zero_padded = str(tic_num).zfill(16)
    pattern = os.path.join(CACHE_DIR, f'*{zero_padded}*', '*lc.fits')
    matches = glob.glob(pattern)
    
    matches.sort()
    matches = matches[:max_sectors]
    
    if len(matches) == 0:
        return None
        
    lcs = [lk.read(m) for m in matches]
    if len(lcs) == 1:
        return lcs[0]
        
    lc_collection = lk.LightCurveCollection(lcs)
    return lc_collection.stitch()

def generate_sample(flattened_lc, period, epoch, num_bins=2000):
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
    manifest_path = os.path.join('data', 'multisector_manifest.json')
    
    with open(json_path_pos, 'r') as f: targets_pos = json.load(f)
    with open(json_path_eb, 'r') as f: targets_eb = json.load(f)
    with open(manifest_path, 'r') as f: manifest = json.load(f)
    
    # We only process targets that are in our rigorous cohort manifest
    valid_tics = set(manifest.keys())
    
    X_list = []
    y_list = []
    
    print("\n--- Building Exp 3 (Multi-Sector) Dataset ---")
    print("V1 Baseline Contract: SG101 + No Outlier Clipping")
    
    # Process Planets & Noise
    for star_id, params in targets_pos.items():
        if star_id not in valid_tics: continue
        clean_star_id = star_id.replace("TIC TIC ", "TIC ")
            
        print(f"Processing Planet & Noise for {clean_star_id}...")
        try:
            lc = get_stitched_lc(clean_star_id, max_sectors=5)
            if lc is None: continue
            
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
        except Exception as e:
            print(f"Failed processing {clean_star_id}: {e}")

    # Process Eclipsing Binaries
    for star_id, params in targets_eb.items():
        if star_id not in valid_tics: continue
            
        print(f"Processing EB for {star_id}...")
        try:
            lc = get_stitched_lc(star_id, max_sectors=5)
            if lc is None: continue
            
            flattened_lc = lc.flatten(window_length=101)
            
            true_period = params['period']
            true_epoch = params['epoch']
            
            # Label 2: Eclipsing Binary
            flux_eb = generate_sample(flattened_lc, true_period, true_epoch)
            if flux_eb is not None:
                X_list.append(flux_eb)
                y_list.append(2)
        except Exception as e:
            print(f"Failed processing {star_id}: {e}")

    X = np.array(X_list)
    Y = np.array(y_list)
    
    print("\n--- Exp 3 Dataset Summary ---")
    print(f"X Matrix Shape: {X.shape}")
    print(f"Y Vector Shape: {Y.shape}")
    print(f"Label 0 (Noise): {np.sum(Y==0)}")
    print(f"Label 1 (Planet): {np.sum(Y==1)}")
    print(f"Label 2 (EB): {np.sum(Y==2)}")
    
    save_dir = os.path.join('data', 'tess_ml_arrays')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'tess_dataset_exp3.npz')
    np.savez(save_path, X=X, y=Y)
    print(f"\nExp 3 Dataset successfully saved to: {save_path}")

if __name__ == '__main__':
    main()
