import os
import json
import numpy as np
import lightkurve as lk
import warnings
import glob
import time

warnings.filterwarnings('ignore')

CACHE_DIR = os.path.expanduser('~/.cache/lightkurve/mastDownload/TESS')

def find_local_fits(tic_id_str):
    tic_num = tic_id_str.replace("TIC", "").strip()
    zero_padded = str(tic_num).zfill(16)
    pattern = os.path.join(CACHE_DIR, f'*{zero_padded}*', '*lc.fits')
    matches = glob.glob(pattern)
    if len(matches) > 0:
        return matches[0]
    return None

def generate_sample(flattened_lc, period, epoch, num_bins=2000):
    folded_lc = flattened_lc.fold(period=period, epoch_time=epoch)
    binned_lc = folded_lc.bin(bins=num_bins)
    flux = binned_lc.flux.value
    if np.isnan(flux).any() or len(flux) != num_bins:
        return None
    mean_flux = np.mean(flux)
    std_flux = np.std(flux)
    if std_flux == 0:
        return None
    normalized_flux = (flux - mean_flux) / std_flux
    return normalized_flux.reshape(-1, 1)

def main():
    print("Loading targets...")
    with open('data/tess_positive_targets.json', 'r') as f:
        targets_pos = json.load(f)
    with open('data/tess_eb_targets.json', 'r') as f:
        targets_eb = json.load(f)

    X_list = []
    y_list = []
    max_samples = 300
    
    preprocessing_times = []
    
    # Planets & Noise
    pos_count = 0
    for star_id, params in targets_pos.items():
        if pos_count >= max_samples:
            break
        clean_star_id = star_id.replace("TIC TIC ", "TIC ")
        
        fits_path = find_local_fits(clean_star_id)
        if not fits_path:
            continue
            
        print(f"Planet [{pos_count+1}/{max_samples}] Processing {clean_star_id} from {fits_path}")
        
        try:
            lc = lk.read(fits_path)
            
            t0 = time.time()
            flattened_lc = lc.flatten(window_length=401)
            t1 = time.time()
            preprocessing_times.append({'target': clean_star_id, 'class': 'Planet/Noise', 'time': t1 - t0})
            
            true_period = params['period']
            true_epoch = params['epoch']
            
            flux_pos = generate_sample(flattened_lc, true_period, true_epoch)
            if flux_pos is not None:
                X_list.append(flux_pos)
                y_list.append(1)
                
                mangled_period = true_period * 1.345
                shifted_epoch = true_epoch + (true_period * 0.5)
                flux_neg = generate_sample(flattened_lc, mangled_period, shifted_epoch)
                
                if flux_neg is not None:
                    X_list.append(flux_neg)
                    y_list.append(0)
                pos_count += 1
        except Exception as e:
            print(f"  -> Failed {clean_star_id} due to {type(e).__name__}: {e}")

    # Eclipsing Binaries
    print("\n--- Processing ECLIPSING BINARIES (Label 2) ---")
    eb_count = 0
    for star_id, params in targets_eb.items():
        if eb_count >= max_samples:
            break
        clean_star_id = star_id
        
        fits_path = find_local_fits(clean_star_id)
        if not fits_path:
            continue
            
        print(f"EB [{eb_count+1}/{max_samples}] Processing {clean_star_id} from {fits_path}")
        
        try:
            lc = lk.read(fits_path)
            
            t0 = time.time()
            flattened_lc = lc.flatten(window_length=401)
            t1 = time.time()
            preprocessing_times.append({'target': clean_star_id, 'class': 'EB', 'time': t1 - t0})
            
            true_period = params['period']
            true_epoch = params['epoch']
            
            flux_eb = generate_sample(flattened_lc, true_period, true_epoch)
            if flux_eb is not None:
                X_list.append(flux_eb)
                y_list.append(2)
                eb_count += 1
        except Exception as e:
            print(f"  -> Failed {clean_star_id} due to {type(e).__name__}: {e}")

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
    save_path = os.path.join(save_dir, 'tess_dataset_exp1.npz')
    
    np.savez(save_path, X=X, y=Y)
    print(f"\nTernary Dataset successfully saved to: {save_path}")
    
    # Save preprocessing times
    with open('data/exp1_preprocessing_times.json', 'w') as f:
        json.dump(preprocessing_times, f, indent=4)
    print("Saved preprocessing times to data/exp1_preprocessing_times.json")

if __name__ == '__main__':
    main()
