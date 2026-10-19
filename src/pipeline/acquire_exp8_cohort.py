import os
import json
import time
import glob
import warnings
import numpy as np
import pandas as pd
import lightkurve as lk
from lightkurve.utils import LightkurveError

warnings.filterwarnings('ignore')
CACHE_DIR = os.path.expanduser('~/.cache/lightkurve/mastDownload/TESS')
os.makedirs(CACHE_DIR, exist_ok=True)

# 10 Known Difficult Validation Targets
DIFFICULT_TARGET_TICS = [
    '259377017', '36724087', '287328202', '345143460', '234994474',
    '150030205', '262530407', '181804752', '307809773', '254113311'
]

def get_cached_sectors_count(tic_id_str):
    tic_num = tic_id_str.replace("TIC ", "").replace("TIC TIC ", "").strip()
    zero_padded = str(tic_num).zfill(16)
    pattern = os.path.join(CACHE_DIR, f'*{zero_padded}*', '*lc.fits')
    return len(glob.glob(pattern))

def get_cached_lcs(tic_id_str, max_sectors=5):
    tic_num = tic_id_str.replace("TIC ", "").replace("TIC TIC ", "").strip()
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
        except Exception:
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
        z_flux = np.nan_to_num(z_flux, nan=0.0, posinf=0.0, neginf=0.0)
        return z_flux
    except Exception:
        return None

def process_target(star_id, params, is_eb):
    lcs = get_cached_lcs(star_id, max_sectors=5)
    if lcs is None:
        return None
        
    true_period = params['period']
    true_epoch = params['epoch']
    
    # 1. Positive sample
    pos_sectors = []
    for lc in lcs:
        s = generate_sample_single_sector(lc, true_period, true_epoch)
        if s is None:
            return None
        pos_sectors.append(s)
    pos_sectors = np.array(pos_sectors) # (5, 2000)
    pos_label = 2 if is_eb else 1
    
    # 2. Negative sample (off-period noise)
    mangled_period = true_period * 1.345
    shifted_epoch = true_epoch + (true_period * 0.5)
    neg_sectors = []
    for lc in lcs:
        s = generate_sample_single_sector(lc, mangled_period, shifted_epoch)
        if s is None:
            return None
        neg_sectors.append(s)
    neg_sectors = np.array(neg_sectors) # (5, 2000)
    neg_label = 0
    
    return (pos_sectors, pos_label), (neg_sectors, neg_label)

def execute_download_with_backoff(search_item, max_retries=3):
    for attempt in range(max_retries):
        try:
            search_item.download()
            return True
        except Exception:
            time.sleep(2 ** attempt)
    return False

def acquire_and_build(target_planets=62, target_ebs=62):
    print("=" * 60)
    print(f"ACQUIRING EXP8 COHORT: Goal = {target_planets} Planets, {target_ebs} EBs (Total = {target_planets+target_ebs} targets)")
    print("=" * 60)
    
    pos_targets_path = 'data/tess_positive_targets.json'
    eb_targets_path = 'data/tess_eb_targets.json'
    
    if not os.path.exists(pos_targets_path) or not os.path.exists(eb_targets_path):
        print(f"\n[ERROR] Target catalog JSONs not found in 'data/'.")
        print("Please ensure 'tess_positive_targets.json' and 'tess_eb_targets.json' are placed in 'data/'.")
        print("If you need to fetch them from NASA archive, run:")
        print("  python src/pipeline/fetch_targets.py")
        print("  python src/pipeline/fetch_eb_targets.py\n")
        return None
        
    with open(pos_targets_path, 'r') as f:
        targets_pos = json.load(f)
    with open(eb_targets_path, 'r') as f:
        targets_eb = json.load(f)
        
    # Sort planets so that the 10 difficult validation targets are processed FIRST
    priority_planets = []
    other_planets = []
    for tic, params in targets_pos.items():
        if any(diff_id in tic for diff_id in DIFFICULT_TARGET_TICS):
            priority_planets.append((tic, params, False))
        else:
            other_planets.append((tic, params, False))
            
    print(f"Priority Validation Planets: {len(priority_planets)} found.")
    planet_queue = priority_planets + other_planets
    eb_queue = [(tic, params, True) for tic, params in targets_eb.items()]
    
    acquired_targets = []
    
    def download_cohort(queue, max_needed, kind_label):
        count = 0
        for tic, params, is_eb in queue:
            if count >= max_needed:
                break
                
            clean_tic = tic.replace("TIC ", "").replace("TIC TIC ", "").strip()
            
            # Check if already cached
            cached = get_cached_sectors_count(tic)
            if cached >= 5:
                # Test if it passes local QC
                res = process_target(tic, params, is_eb)
                if res is not None:
                    count += 1
                    acquired_targets.append((tic, params, is_eb))
                    print(f"[{kind_label}] {tic} already cached and valid ({count}/{max_needed})")
                    continue
                    
            # Search MAST
            try:
                search = lk.search_lightcurve(f"TIC {clean_tic}", mission='TESS', author='SPOC')
            except Exception as e:
                time.sleep(2)
                continue
                
            if len(search) < 5:
                continue
                
            print(f"[{kind_label}] Downloading 5 sectors for {tic} (available: {len(search)})...")
            download_ok = True
            for i in range(5):
                ok = execute_download_with_backoff(search[i])
                if not ok:
                    download_ok = False
                    break
                    
            if download_ok:
                # Verify local QC immediately
                res = process_target(tic, params, is_eb)
                if res is not None:
                    count += 1
                    acquired_targets.append((tic, params, is_eb))
                    print(f"  --> Success! {tic} added ({count}/{max_needed})")
                else:
                    print(f"  --> Sector QC failed (NaNs/flux anomaly) for {tic}")
            time.sleep(0.5)
        return count
        
    print("\n--- PHASE 1: ACQUIRING PLANETS ---")
    p_count = download_cohort(planet_queue, target_planets, "PLANET")
    print(f"Total Valid Planets Acquired: {p_count}")
    
    print("\n--- PHASE 2: ACQUIRING ECLIPSING BINARIES ---")
    eb_count = download_cohort(eb_queue, target_ebs, "EB")
    print(f"Total Valid EBs Acquired: {eb_count}")
    
    print(f"\nTotal Physical Targets Acquired: {len(acquired_targets)}")
    
    # Build Exp 8 NPZ
    print("\n--- PHASE 3: COMPILING tess_dataset_exp8.npz ---")
    X_list = []
    y_list = []
    tics_list = []
    
    for tic, params, is_eb in acquired_targets:
        res = process_target(tic, params, is_eb)
        if res is not None:
            pos_s, neg_s = res
            
            # Positive
            X_list.append(pos_s[0])
            y_list.append(pos_s[1])
            tics_list.append(f"{tic}_Positive")
            
            # Negative
            X_list.append(neg_s[0])
            y_list.append(neg_s[1])
            tics_list.append(f"{tic}_Noise")
            
    X = np.array(X_list)
    Y = np.array(y_list)
    TICS = np.array(tics_list)
    
    X = np.expand_dims(X, axis=-1)
    
    print("\n" + "=" * 50)
    print("DATASET ASSEMBLY COMPLETE")
    print("=" * 50)
    print(f"X Matrix Shape: {X.shape}")
    print(f"Y Vector Shape: {Y.shape}")
    print(f"TICS Vector Shape: {TICS.shape}")
    print(f"Label 0 (Noise):   {np.sum(Y == 0)}")
    print(f"Label 1 (Planet):  {np.sum(Y == 1)}")
    print(f"Label 2 (EB):      {np.sum(Y == 2)}")
    
    save_dir = os.path.join('data', 'tess_ml_arrays')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'tess_dataset_exp8.npz')
    np.savez(save_path, X=X, y=Y, tics=TICS)
    print(f"\n[SAVED] Exp 8 Dataset written to: {save_path}")
    
    # Verify the 10 difficult validation planets
    diff_found = [t for t in DIFFICULT_TARGET_TICS if any(t in s for s in tics_list)]
    print(f"Difficult Validation Planets in Dataset: {len(diff_found)}/10: {diff_found}")
    return save_path

if __name__ == '__main__':
    acquire_and_build(target_planets=62, target_ebs=62)
