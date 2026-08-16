import matplotlib.pyplot as plt
import lightkurve as lk
import numpy as np
import glob
import os
import json
import time

def main():
    # Load targets to get epoch and period
    with open('data/tess_positive_targets.json', 'r') as f:
        targets_pos = json.load(f)
    
    # Pick a representative target (e.g., TIC 25078924)
    target_tic = "25078924"
    params = targets_pos.get(f"TIC {target_tic}") or targets_pos.get(f"TIC TIC {target_tic}")
    if not params:
        # Fallback to first
        target_tic = list(targets_pos.keys())[0].replace("TIC ", "").replace("TIC ", "")
        params = list(targets_pos.values())[0]

    period = params['period']
    epoch = params['epoch']

    # Find local FITS
    cache_dir = os.path.expanduser('~/.cache/lightkurve/mastDownload/TESS')
    pattern = os.path.join(cache_dir, f'*{str(target_tic).zfill(16)}*', '*lc.fits')
    matches = glob.glob(pattern)
    if not matches:
        print(f"Could not find FITS for {target_tic}")
        return
        
    lc = lk.read(matches[0])
    
    # Process V1 (SG101)
    t0 = time.time()
    lc_v1 = lc.flatten(window_length=101)
    t_v1 = time.time() - t0
    folded_v1 = lc_v1.fold(period=period, epoch_time=epoch).bin(bins=2000)
    flux_v1 = folded_v1.flux.value
    flux_v1 = (flux_v1 - np.mean(flux_v1)) / np.std(flux_v1)
    
    # Process Exp1 (SG401)
    t0 = time.time()
    lc_exp1 = lc.flatten(window_length=401)
    t_exp1 = time.time() - t0
    folded_exp1 = lc_exp1.fold(period=period, epoch_time=epoch).bin(bins=2000)
    flux_exp1 = folded_exp1.flux.value
    flux_exp1 = (flux_exp1 - np.mean(flux_exp1)) / np.std(flux_exp1)

    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(12, 12))
    
    # 1. Full flattened light curves
    axes[0].plot(lc_v1.time.value, lc_v1.flux.value, label='V1 (SG101)', alpha=0.7)
    axes[0].plot(lc_exp1.time.value, lc_exp1.flux.value, label='Exp1 (SG401)', alpha=0.7)
    axes[0].set_title(f"Full Flattened Light Curve - TIC {target_tic}")
    axes[0].legend()
    
    # 2. Folded transit morphology (Zoomed in on phase 0)
    phases = np.linspace(-0.5, 0.5, 2000)
    axes[1].plot(phases, flux_v1, label=f'V1 (SG101) [{t_v1:.2f}s]', alpha=0.8)
    axes[1].plot(phases, flux_exp1, label=f'Exp1 (SG401) [{t_exp1:.2f}s]', alpha=0.8)
    axes[1].set_xlim(-0.1, 0.1) # zoom on transit
    axes[1].set_title("Transit Morphology Comparison (Folded & Normalized)")
    axes[1].legend()
    
    # 3. Difference (Residuals)
    axes[2].plot(phases, flux_exp1 - flux_v1, color='red', label='Difference (SG401 - SG101)')
    axes[2].set_xlim(-0.1, 0.1)
    axes[2].set_title("Residuals around Transit")
    axes[2].legend()
    
    plt.tight_layout()
    os.makedirs('docs/images', exist_ok=True)
    out_path = f'docs/images/morphology_comparison_{target_tic}.png'
    plt.savefig(out_path)
    print(f"Saved morphology comparison to {out_path}")
    print(f"V1 Processing Time: {t_v1:.2f}s")
    print(f"Exp1 Processing Time: {t_exp1:.2f}s")

if __name__ == '__main__':
    main()
