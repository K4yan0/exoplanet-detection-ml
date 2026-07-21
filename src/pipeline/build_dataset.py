import lightkurve as lk
import numpy as np
import pandas as pd
import os

def process_star(star_id, period, epoch, num_bins=2000):
    """
    Fetches, flattens, folds, and bins a light curve for a given star.
    Returns a 1D numpy array of length `num_bins`.
    """
    print(f"  -> Fetching SPOC data for {star_id}...")
    # Get the SPOC pipeline light curve, download the first one found
    search_result = lk.search_lightcurve(star_id, mission='TESS', author='SPOC')
    
    if len(search_result) == 0:
        print(f"  [!] No SPOC data found for {star_id}. Skipping.")
        return None
        
    lc = search_result[0].download()
    
    print(f"  -> Flattening...")
    flattened_lc = lc.flatten(window_length=101)
    
    print(f"  -> Folding (Period: {period}, Epoch: {epoch})...")
    folded_lc = flattened_lc.fold(period=period, epoch_time=epoch)
    
    print(f"  -> Binning to {num_bins} points...")
    binned_lc = folded_lc.bin(bins=num_bins)
    
    flux = binned_lc.flux.value
    
    # Interpolate any NaNs
    if np.isnan(flux).any():
        flux = pd.Series(flux).interpolate(limit_direction='both').values
        
    return flux

def main():
    # 1. Define our small test dictionary of targets
    # Format: { 'Star Name': {'label': 1 or 0, 'period': float, 'epoch': float} }
    targets = {
        # POSITIVES (Real Exoplanets)
        'Pi Mensae':   {'label': 1, 'period': 6.267852, 'epoch': 1325.504},
        'WASP-126':    {'label': 1, 'period': 3.2888,   'epoch': 1354.214}, # Approximate epoch for WASP-126b
        
        # NEGATIVES (Non-planets / Random stars folded on arbitrary periods to simulate noise)
        'Tau Ceti':    {'label': 0, 'period': 8.4321,   'epoch': 1330.0},
        'Sirius':      {'label': 0, 'period': 4.1234,   'epoch': 1328.0}
    }
    
    X_list = []
    y_list = []
    
    print("Starting Dataset Generation...")
    
    # 2. Loop through the dictionary
    for star_id, params in targets.items():
        print(f"\nProcessing {star_id} (Label: {params['label']})")
        try:
            flux_array = process_star(
                star_id=star_id, 
                period=params['period'], 
                epoch=params['epoch']
            )
            
            if flux_array is not None:
                # 3. Append to our lists
                X_list.append(flux_array)
                y_list.append(params['label'])
                print(f"  [SUCCESS] Added {star_id} to dataset.")
        except Exception as e:
            print(f"  [ERROR] Failed to process {star_id}: {e}")

    # Convert lists to final Numpy matrices
    X = np.array(X_list)
    Y = np.array(y_list)
    
    print("\n--- Dataset Summary ---")
    print(f"X Matrix Shape: {X.shape}")
    print(f"Y Vector Shape: {Y.shape}")
    
    # 4. Save the dataset
    # We save to data/tess_ml_arrays/ assuming the script is run from the project root
    save_dir = os.path.join('data', 'tess_ml_arrays')
    os.makedirs(save_dir, exist_ok=True)
    
    save_path = os.path.join(save_dir, 'tess_dataset_v1.npz')
    np.savez(save_path, X=X, y=Y)
    print(f"\nDataset successfully saved to: {save_path}")

if __name__ == '__main__':
    main()
