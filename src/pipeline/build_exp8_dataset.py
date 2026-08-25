import lightkurve as lk
import numpy as np
import pandas as pd
import os
import json
import glob
import warnings
import concurrent.futures

warnings.filterwarnings('ignore')
CACHE_DIR = os.path.expanduser('~/.cache/lightkurve/mastDownload/TESS')

def get_sector_lcs(tic_str, max_sectors=5):
    tic_num = tic_str.replace("TIC ", "").strip()
    zero_padded = str(tic_num).zfill(16)
    pattern = os.path.join(CACHE_DIR, f'*{zero_padded}*', '*lc.fits')
    matches = glob.glob(pattern)
    matches.sort()
    matches = matches[:max_sectors]
    if len(matches) < max_sectors:
        return None
        
    lcs = []
    for m in matches:
        try:
            lc = lk.read(m)
            lc = lc[~np.isnan(lc.flux.value)]
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

def generate_sample_single_sector(lc, period, epoch, num_bins=2000):
    try:
        flattened_lc = lc.flatten(window_length=101)
        folded_lc = flattened_lc.fold(period=period, epoch_time=epoch)
        binned_lc = folded_lc.bin(bins=num_bins)
        
        flux = binned_lc.flux.value
        
        if np.isnan(flux).any():
            flux = pd.Series(flux).interpolate(limit_direction='both').values
            
        if len(flux) != num_bins or np.isnan(flux).any() or np.isinf(flux).any():
            return None
            
        mean = np.mean(flux)
        std = np.std(flux)
        if std == 0:
            return None
            
        z_flux = (flux - mean) / std
        
        if np.isnan(z_flux).any():
            return None
            
        return z_flux
    except Exception as e:
        return None

def process_target_with_noise(item):
    star_id, params, is_eb = item
    clean_star_id = star_id.replace("TIC TIC ", "TIC ")
    
    lcs_5_list = get_sector_lcs(clean_star_id, max_sectors=5)
    if lcs_5_list is None: return None
    
    true_period = params['period']
    true_epoch = params['epoch']
    
    # 1. Positive Label (Planet/EB)
    flux_5_ind_pos = []
    for lc in lcs_5_list:
        sample = generate_sample_single_sector(lc, true_period, true_epoch)
        if sample is None: return None
        flux_5_ind_pos.append(sample)
    flux_5_ind_pos = np.array(flux_5_ind_pos) # (5, 2000)
    
    pos_label = 2 if is_eb else 1
    pos_sample = (flux_5_ind_pos, pos_label)
    
    # 2. Negative Label (Noise/Empty)
    mangled_period = true_period * 1.345
    shifted_epoch = true_epoch + (true_period * 0.5)
    
    flux_5_ind_neg = []
    for lc in lcs_5_list:
        sample = generate_sample_single_sector(lc, mangled_period, shifted_epoch)
        if sample is None: return None
        flux_5_ind_neg.append(sample)
    flux_5_ind_neg = np.array(flux_5_ind_neg) # (5, 2000)
    
    neg_sample = (flux_5_ind_neg, 0)
    
    return (pos_sample, neg_sample)

def main():
    print("\n--- Building Exp 8 (Independent Sectors) Dataset ---")
    
    # Load targets that survived Exp 7 to ensure a perfectly paired cohort
    path_exp7 = 'data/tess_ml_arrays/tess_dataset_exp7_5sec.npz'
    if not os.path.exists(path_exp7):
        print("Exp 7 dataset not found. Please run build_exp7_dataset.py first.")
        return
        
    d7 = np.load(path_exp7)
    exp7_tics = d7['tics']
    
    # Extract clean target names (strip _Positive, _Noise, and double TIC)
    target_names = set([t.replace("_Positive", "").replace("_Noise", "").replace("TIC TIC ", "TIC ") for t in exp7_tics])
    
    json_path_pos = os.path.join('data', 'tess_positive_targets.json')
    json_path_eb = os.path.join('data', 'tess_eb_targets.json')
    
    with open(json_path_pos, 'r') as f: targets_pos = json.load(f)
    with open(json_path_eb, 'r') as f: targets_eb = json.load(f)
    
    # Fix the keys to match target_names logic
    all_items = []
    for k, v in targets_pos.items():
        clean_k = k.replace("TIC TIC ", "TIC ")
        if clean_k in target_names:
            all_items.append((k, v, False))
            
    for k, v in targets_eb.items():
        clean_k = k.replace("TIC TIC ", "TIC ")
        if clean_k in target_names:
            all_items.append((k, v, True))
    
    print(f"Total targets to process (Exp 7 cohort subset): {len(all_items)}")
    
    X_list = []
    y_list = []
    tics_list = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        for i, res in enumerate(executor.map(process_target_with_noise, all_items)):
            if res is not None:
                pos_s, neg_s = res
                tic_id = all_items[i][0]
                
                # Double-check for NaNs just in case
                if np.isnan(pos_s[0]).any() or np.isnan(neg_s[0]).any():
                    print(f"Processed target {i+1}/{len(all_items)}: FAILED (Contains NaNs)")
                    continue
                
                # Positive
                X_list.append(pos_s[0])
                y_list.append(pos_s[1])
                tics_list.append(f"{tic_id}_Positive")
                
                # Negative
                X_list.append(neg_s[0])
                y_list.append(neg_s[1])
                tics_list.append(f"{tic_id}_Noise")
                
                print(f"Processed target {i+1}/{len(all_items)}: SUCCESS")
            else:
                print(f"Processed target {i+1}/{len(all_items)}: FAILED (Generation Error)")
                
    X = np.array(X_list)
    Y = np.array(y_list)
    TICS = np.array(tics_list)
    
    X = np.expand_dims(X, axis=-1)
    
    print("\n--- Exp 8 Dataset Summary ---")
    print(f"X Matrix Shape: {X.shape}")
    print(f"Y Vector Shape: {Y.shape}")
    print(f"TICS Vector Shape: {TICS.shape}")
    print(f"Label 0 (Noise): {np.sum(Y==0)}")
    print(f"Label 1 (Planet): {np.sum(Y==1)}")
    print(f"Label 2 (EB): {np.sum(Y==2)}")
    
    save_dir = os.path.join('data', 'tess_ml_arrays')
    os.makedirs(save_dir, exist_ok=True)
    
    save_path = os.path.join(save_dir, 'tess_dataset_exp8.npz')
    np.savez(save_path, X=X, y=Y, tics=TICS)
    
    print(f"\nExp 8 Dataset successfully saved to: {save_path}")

if __name__ == '__main__':
    main()
