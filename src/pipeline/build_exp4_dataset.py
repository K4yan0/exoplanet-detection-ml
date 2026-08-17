import lightkurve as lk
import numpy as np
import pandas as pd
import os
import json
import warnings
import time
import requests
import socket

# Force a 30-second timeout on ALL sockets (fixes astropy urllib hangs)
socket.setdefaulttimeout(30.0)

# Force a 30-second timeout on ALL requests (fixes MAST search hangs)
original_request = requests.Session.request
def timeout_request(self, method, url, **kwargs):
    if 'timeout' not in kwargs or kwargs['timeout'] is None:
        kwargs['timeout'] = 30
    return original_request(self, method, url, **kwargs)
requests.Session.request = timeout_request

warnings.filterwarnings('ignore')

def safe_search_and_download(clean_star_id):
    """Robust wrapper for MAST searches to prevent hanging. Retries indefinitely if MAST is down."""
    attempt = 1
    while True:
        try:
            search_result = lk.search_lightcurve(clean_star_id, mission='TESS', author='SPOC')
            if len(search_result) == 0:
                return None
            lc = search_result[0].download()
            return lc
        except Exception as e:
            print(f"    [Error/Timeout on MAST API for {clean_star_id}: {e}, retrying... (Attempt {attempt})]")
            time.sleep(5)
            attempt += 1

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

def remove_outliers_nondestructive(lc, sigma_upper=3.0, sigma_lower=10.0):
    """
    Identifies outliers and interpolates over them instead of dropping rows.
    This prevents length-reduction issues during subsequent binning.
    """
    clean_lc = lc.copy()
    flux = clean_lc.flux.value
    
    # Calculate robust median and std
    median = np.nanmedian(flux)
    std = np.nanstd(flux)
    
    # Create mask for outliers
    outlier_mask = (flux > median + sigma_upper * std) | (flux < median - sigma_lower * std)
    
    if outlier_mask.any():
        # Replace outliers with NaN
        flux[outlier_mask] = np.nan
        # Interpolate NaNs
        flux = pd.Series(flux).interpolate(limit_direction='both').values
        
    clean_lc.flux = flux
    return clean_lc

def main():
    json_path_pos = os.path.join('data', 'tess_positive_targets.json')
    json_path_eb = os.path.join('data', 'tess_eb_targets.json')
        
    with open(json_path_pos, 'r') as f:
        targets_pos = json.load(f)
    with open(json_path_eb, 'r') as f:
        targets_eb = json.load(f)
        
    print(f"Loaded {len(targets_pos)} Planets and {len(targets_eb)} EBs")
    
    X_v1_list = []
    X_exp4_list = []
    y_list = []
    
    print("\n--- Processing PLANETS & NOISE (Label 1 & Label 0) ---")
    pos_count = 0
    max_samples = 300
    
    for star_id, params in targets_pos.items():
        if pos_count >= max_samples:
            break
            
        clean_star_id = star_id.replace("TIC TIC ", "TIC ")
        print(f"Attempting {clean_star_id}... ({pos_count}/300 successes)")
        
        try:
            lc = safe_search_and_download(clean_star_id)
            if lc is None:
                continue
            
            flattened_lc = lc.flatten(window_length=101)
            true_period = params['period']
            true_epoch = params['epoch']
            
            # Label 1: Planet
            flux_pos_v1 = generate_sample(flattened_lc, true_period, true_epoch)
            if flux_pos_v1 is not None:
                clean_lc = remove_outliers_nondestructive(flattened_lc, sigma_upper=3.0, sigma_lower=10.0)
                flux_pos_exp4 = generate_sample(clean_lc, true_period, true_epoch)
                
                if flux_pos_exp4 is not None:
                    X_v1_list.append(flux_pos_v1)
                    X_exp4_list.append(flux_pos_exp4)
                    y_list.append(1)
                    
                    # Label 0: Noise
                    mangled_period = true_period * 1.345
                    shifted_epoch = true_epoch + (true_period * 0.5)
                    
                    flux_neg_v1 = generate_sample(flattened_lc, mangled_period, shifted_epoch)
                    if flux_neg_v1 is not None:
                        flux_neg_exp4 = generate_sample(clean_lc, mangled_period, shifted_epoch)
                        if flux_neg_exp4 is not None:
                            X_v1_list.append(flux_neg_v1)
                            X_exp4_list.append(flux_neg_exp4)
                            y_list.append(0)
                            
                    pos_count += 1
                    print(f"  -> SUCCESS! Total: {pos_count}/300")
        except Exception as e:
            pass

    print("\n--- Processing ECLIPSING BINARIES (Label 2) ---")
    eb_count = 0
    
    for star_id, params in targets_eb.items():
        if eb_count >= max_samples:
            break
            
        clean_star_id = star_id
        print(f"Attempting EB {clean_star_id}... ({eb_count}/300 successes)")
        
        try:
            lc = safe_search_and_download(clean_star_id)
            if lc is None:
                continue
            
            flattened_lc = lc.flatten(window_length=101)
            true_period = params['period']
            true_epoch = params['epoch']
            
            flux_eb_v1 = generate_sample(flattened_lc, true_period, true_epoch)
            if flux_eb_v1 is not None:
                clean_lc = remove_outliers_nondestructive(flattened_lc, sigma_upper=3.0, sigma_lower=10.0)
                flux_eb_exp4 = generate_sample(clean_lc, true_period, true_epoch)
                
                if flux_eb_exp4 is not None:
                    X_v1_list.append(flux_eb_v1)
                    X_exp4_list.append(flux_eb_exp4)
                    y_list.append(2)
                    eb_count += 1
                    print(f"  -> SUCCESS! Total: {eb_count}/300")
        except Exception as e:
            pass

    X_v1 = np.array(X_v1_list)
    X_exp4 = np.array(X_exp4_list)
    Y = np.array(y_list)
    
    print("\n--- Aligned Dataset Summary ---")
    print(f"X_v1 Shape: {X_v1.shape}")
    print(f"X_exp4 Shape: {X_exp4.shape}")
    print(f"Y Vector Shape: {Y.shape}")
    
    save_dir = os.path.join('data', 'tess_ml_arrays')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'aligned_v1_exp4_dataset.npz')
    
    np.savez(save_path, X_v1=X_v1, X_exp4=X_exp4, y=Y)
    print(f"\nAligned Dataset successfully saved to: {save_path}")

if __name__ == '__main__':
    main()
