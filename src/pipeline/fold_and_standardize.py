import lightkurve as lk
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def main():
    print("1. Downloading Pi Mensae light curve...")
    # Get the SPOC pipeline light curve
    search_result = lk.search_lightcurve('Pi Mensae', mission='TESS', author='SPOC')
    lc = search_result[0].download()
    
    print("2. Flattening the light curve...")
    flattened_lc = lc.flatten(window_length=101)
    
    print("3. Folding the light curve (Step 3)...")
    # Parameters for Pi Mensae c as provided
    period = 6.267852
    epoch_time = 1325.504
    
    folded_lc = flattened_lc.fold(period=period, epoch_time=epoch_time)
    
    print("4. Standardizing (Binning) for 1D CNN (Step 4 Prep)...")
    # Bin the folded light curve into a fixed number of bins
    # This guarantees the input to the 1D CNN is always exactly 'num_bins' long.
    num_bins = 2000
    binned_lc = folded_lc.bin(bins=num_bins)
    
    # Extract exactly what the CNN needs: a 1D array of flux values
    phase = binned_lc.time.value
    flux = binned_lc.flux.value
    
    # In real data, binning can sometimes result in empty bins (NaNs). 
    # We interpolate them to ensure the CNN receives clean, complete data without gaps.
    if np.isnan(flux).any():
        nan_count = np.isnan(flux).sum()
        print(f"  Found {nan_count} empty bins (NaNs). Interpolating...")
        flux = pd.Series(flux).interpolate(limit_direction='both').values
    else:
        print("  No empty bins found. Data is perfectly dense.")
        
    print(f"  Final 1D Array Shape ready for CNN: {flux.shape}")
    
    # Plotting to verify our work
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot 1: The scatter of the folded light curve
    folded_lc.scatter(ax=ax1, alpha=0.5, label='Folded Data (Raw Points)', color='gray')
    ax1.set_title(f'Step 3: Phase-Folded Light Curve (Period: {period} d, Epoch: {epoch_time})')
    
    # Plot 2: The standardized binned data ready for the CNN
    ax2.plot(phase, flux, drawstyle='steps-mid', color='blue', linewidth=2, label=f'Standardized Input ({num_bins} points)')
    ax2.set_ylabel('Normalized Flux')
    ax2.set_xlabel('Phase (Time mapped from -0.5 to 0.5 of the orbital period)')
    ax2.set_title(f'Step 4 Prep: Binned 1D Array for CNN Input')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('pi_mensae_folded.png')
    print("Saved plot to pi_mensae_folded.png. Displaying...")
    plt.show()

if __name__ == '__main__':
    main()
