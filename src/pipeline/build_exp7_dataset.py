import lightkurve as lk
import numpy as np
import pandas as pd
import os
import json
import glob
import warnings
warnings.filterwarnings('ignore')

CACHE_DIR = os.path.expanduser('~/.cache/lightkurve/mastDownload/TESS')

def get_sector_lcs(tic_str, max_sectors=5):
    """Retrieves up to max_sectors LightCurves for a target."""
    tic_num = tic_str.replace("TIC ", "").strip()
    zero_padded = str(tic_num).zfill(16)
    pattern = os.path.join(CACHE_DIR, f'*{zero_padded}*', '*lc.fits')
    matches = glob.glob(pattern)
    
    matches.sort()
    matches = matches[:max_sectors]
    
    if len(matches) < max_sectors:
        return None # Require exactly max_sectors
        
    lcs = []
    for m in matches:
        try:
            lc = lk.read(m)
            # Remove NaN fluxes early
            lc = lc[~np.isnan(lc.flux.value)]
            
            # EXP 7 CONTRACT: Local median normalization per sector BEFORE stitching
            median_flux = np.nanmedian(lc.flux.value)
            if median_flux > 0:
                lc.flux = lc.flux / median_flux
                lc.flux_err = lc.flux_err / median_flux
                lcs.append(lc)
        except:
            pass
            
    if len(lcs) < max_sectors:
        return None
        
    return lcs

def generate_sample(lc_collection, period, epoch, num_bins=2000):
    """Stitches, folds, bins, and standardizes the LC."""
    try:
        stitched_lc = lc_collection.stitch()
        
        # EXP 7 CONTRACT: Flatten SG101
        flattened_lc = stitched_lc.flatten(window_length=101)
        
        # EXP 7 CONTRACT: Joint Phase-Folding
        folded_lc = flattened_lc.fold(period=period, epoch_time=epoch)
        
        # EXP 7 CONTRACT: Binning to 2000
        binned_lc = folded_lc.bin(bins=num_bins)
        
        flux = binned_lc.flux.value
        
        if np.isnan(flux).any():
            flux = pd.Series(flux).interpolate(limit_direction='both').values
            
        if len(flux) != num_bins or np.isnan(flux).any():
            return None
            
        # EXP 7 CONTRACT: Final Z-Score
        mean = np.mean(flux)
        std = np.std(flux)
        if std == 0:
            return None
            
        z_flux = (flux - mean) / std
        return z_flux
    except Exception as e:
        return None

def process_target_with_noise(item):
    star_id, params, is_eb = item
    clean_star_id = star_id.replace("TIC TIC ", "TIC ")
    
    # Load exactly 5 sectors
    lcs_5_obj = get_sector_lcs(clean_star_id, max_sectors=5)
    if lcs_5_obj is None: return None
    
    lcs_1_obj = lk.LightCurveCollection([lcs_5_obj[0]])
    lcs_5_obj = lk.LightCurveCollection(lcs_5_obj)
    
    true_period = params['period']
    true_epoch = params['epoch']
    
    # 1. Positive Label (Planet/EB)
    flux_1sec_pos = generate_sample(lcs_1_obj, true_period, true_epoch)
    flux_5sec_pos = generate_sample(lcs_5_obj, true_period, true_epoch)
    
    if flux_1sec_pos is None or flux_5sec_pos is None: return None
        
    pos_label = 2 if is_eb else 1
    pos_sample = (flux_1sec_pos, flux_5sec_pos, pos_label)
    
    # 2. Negative Label (Noise/Empty)
    # Shift period and epoch to look at empty space in the same light curves
    mangled_period = true_period * 1.345
    shifted_epoch = true_epoch + (true_period * 0.5)
    
    flux_1sec_neg = generate_sample(lcs_1_obj, mangled_period, shifted_epoch)
    flux_5sec_neg = generate_sample(lcs_5_obj, mangled_period, shifted_epoch)
    
    if flux_1sec_neg is None or flux_5sec_neg is None: return None
    
    neg_sample = (flux_1sec_neg, flux_5sec_neg, 0)
    
    return (pos_sample, neg_sample)

def main():
    json_path_pos = os.path.join('data', 'tess_positive_targets.json')
    json_path_eb = os.path.join('data', 'tess_eb_targets.json')
    manifest_path = os.path.join('data', 'multisector_manifest.json')
    
    with open(json_path_pos, 'r') as f: targets_pos = json.load(f)
    with open(json_path_eb, 'r') as f: targets_eb = json.load(f)
    with open(manifest_path, 'r') as f: manifest = json.load(f)
    
    # Pre-filter for targets that claim to have 5+ sectors in MAST
    potential_tics = {k: v for k, v in manifest.items() if v.get('total_available_in_mast', 0) >= 5}
    
    print("\n--- Building Exp 7 (Strict Controlled Multi-Sector) Datasets ---")
    print(f"Potential 5-sector targets from manifest: {len(potential_tics)}")
    
    import concurrent.futures
    import multiprocessing
    
    planet_items = [(star_id, params, False) for star_id, params in targets_pos.items() if star_id in potential_tics]
    eb_items = [(star_id, params, True) for star_id, params in targets_eb.items() if star_id in potential_tics]
    all_items = planet_items + eb_items
    
    print(f"Total targets to process: {len(all_items)}")
    
    X_list_1sec = []
    X_list_5sec = []
    y_list_1sec = []
    tics_list = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        for i, res in enumerate(executor.map(process_target_with_noise, all_items)):
            if res is not None:
                pos_s, neg_s = res
                
                # STRICT NaN check before adding to cohort
                if (np.isnan(pos_s[0]).any() or np.isnan(pos_s[1]).any() or 
                    np.isnan(neg_s[0]).any() or np.isnan(neg_s[1]).any()):
                    print(f"Processed target {i+1}/{len(all_items)}: FAILED (Contains NaNs)")
                    continue
                
                tic_id = all_items[i][0]
                
                # Positive
                X_list_1sec.append(pos_s[0])
                X_list_5sec.append(pos_s[1])
                y_list_1sec.append(pos_s[2])
                tics_list.append(f"{tic_id}_Positive")
                
                # Negative
                X_list_1sec.append(neg_s[0])
                X_list_5sec.append(neg_s[1])
                y_list_1sec.append(neg_s[2])
                tics_list.append(f"{tic_id}_Noise")
                
                print(f"Processed target {i+1}/{len(all_items)}: SUCCESS (Clean)")
            else:
                print(f"Processed target {i+1}/{len(all_items)}: FAILED (Generation Error)")
                
    X_1sec = np.array(X_list_1sec)
    X_5sec = np.array(X_list_5sec)
    Y = np.array(y_list_1sec)
    TICS = np.array(tics_list)
    
    # Expand dims for CNN (N, 2000, 1)
    X_1sec = np.expand_dims(X_1sec, axis=-1)
    X_5sec = np.expand_dims(X_5sec, axis=-1)
    
    print("\n--- Exp 7 Dataset Summary (Strict Paired Cohort) ---")
    print(f"X_1sec Matrix Shape: {X_1sec.shape}")
    print(f"X_5sec Matrix Shape: {X_5sec.shape}")
    print(f"Y Vector Shape: {Y.shape}")
    print(f"TICS Vector Shape: {TICS.shape}")
    print(f"Label 0 (Noise): {np.sum(Y==0)}")
    print(f"Label 1 (Planet): {np.sum(Y==1)}")
    print(f"Label 2 (EB): {np.sum(Y==2)}")
    
    save_dir = os.path.join('data', 'tess_ml_arrays')
    os.makedirs(save_dir, exist_ok=True)
    
    save_path_1 = os.path.join(save_dir, 'tess_dataset_exp7_1sec.npz')
    np.savez(save_path_1, X=X_1sec, y=Y, tics=TICS)
    
    save_path_5 = os.path.join(save_dir, 'tess_dataset_exp7_5sec.npz')
    np.savez(save_path_5, X=X_5sec, y=Y, tics=TICS)
    
    print(f"\nExp 7 Datasets successfully saved to:")
    print(f" - {save_path_1}")
    print(f" - {save_path_5}")

if __name__ == '__main__':
    main()
