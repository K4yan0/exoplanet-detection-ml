import lightkurve as lk
import matplotlib.pyplot as plt

def main():
    print("Searching MAST for Pi Mensae (TESS mission)...")
    # 1. Search MAST for a well-known TESS star with a confirmed exoplanet
    # We use author='SPOC' to get the processed Science Processing Operations Center light curves
    search_result = lk.search_lightcurve('Pi Mensae', mission='TESS', author='SPOC')
    print(f"Found {len(search_result)} light curves. Downloading the first one...")
    
    # 2. Download its light curve
    lc = search_result[0].download()
    
    print("Plotting the raw and flattened light curves...")
    # 3. Plot the raw curve, apply flatten(), and plot the flattened result
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Raw light curve
    lc.plot(ax=ax1, ylabel='Raw Flux')
    ax1.set_title('Raw Light Curve of Pi Mensae (TESS)')
    
    # Flattened light curve
    # The flatten() method removes long-term trends using a Savitzky-Golay filter
    flattened_lc = lc.flatten(window_length=101)
    flattened_lc.plot(ax=ax2, ylabel='Normalized Flux')
    ax2.set_title('Flattened Light Curve')
    
    plt.tight_layout()
    plt.savefig('pi_mensae_lightcurve.png')
    print("Saved plot to pi_mensae_lightcurve.png. Showing plot...")
    plt.show()

if __name__ == '__main__':
    main()
