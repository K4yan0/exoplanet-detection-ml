import os
import json
import numpy as np
import lightkurve as lk
import pandas as pd
import glob
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.pipeline.build_exp3_dataset import get_stitched_lc, generate_sample

CACHE_DIR = os.path.expanduser('~/.cache/lightkurve/mastDownload/TESS')

def main():
    print("Reconciling V1 vs Exp 3 Cohort...")
    
    # Get original 591 V1 targets from the manifest
    manifest_path = os.path.join('data', 'multisector_manifest.json')
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
        
    valid_tics = set(manifest.keys())
    
    json_path_pos = os.path.join('data', 'tess_positive_targets.json')
    json_path_eb = os.path.join('data', 'tess_eb_targets.json')
    with open(json_path_pos, 'r') as f: targets_pos = json.load(f)
    with open(json_path_eb, 'r') as f: targets_eb = json.load(f)

    # Let's run the exact generation logic and catch exactly where it fails
    missing_samples = []
    
    # 1. Planets & Noise
    for star_id, params in targets_pos.items():
        if star_id not in valid_tics: continue
        clean_star_id = star_id.replace("TIC TIC ", "TIC ")
        
        lc = get_stitched_lc(clean_star_id, max_sectors=5)
        if lc is None:
            missing_samples.append({'tic': clean_star_id, 'class': 'Planet/Noise', 'reason': 'get_stitched_lc returned None'})
            continue
            
        try:
            flattened_lc = lc.flatten(window_length=101)
        except Exception as e:
            missing_samples.append({'tic': clean_star_id, 'class': 'Planet/Noise', 'reason': f'flatten failed: {e}'})
            continue
            
        true_period = params['period']
        true_epoch = params['epoch']
        
        flux_pos = generate_sample(flattened_lc, true_period, true_epoch)
        if flux_pos is None:
            missing_samples.append({'tic': clean_star_id, 'class': 'Planet', 'reason': 'generate_sample failed (likely NaN or length)'})
            
        mangled_period = true_period * 1.345
        shifted_epoch = true_epoch + (true_period * 0.5)
        flux_neg = generate_sample(flattened_lc, mangled_period, shifted_epoch)
        if flux_neg is None:
            missing_samples.append({'tic': clean_star_id, 'class': 'Noise', 'reason': 'generate_sample failed (likely NaN or length)'})

    # 2. EBs
    for star_id, params in targets_eb.items():
        if star_id not in valid_tics: continue
        
        lc = get_stitched_lc(star_id, max_sectors=5)
        if lc is None:
            missing_samples.append({'tic': star_id, 'class': 'EB', 'reason': 'get_stitched_lc returned None'})
            continue
            
        try:
            flattened_lc = lc.flatten(window_length=101)
        except Exception as e:
            missing_samples.append({'tic': star_id, 'class': 'EB', 'reason': f'flatten failed: {e}'})
            continue
            
        true_period = params['period']
        true_epoch = params['epoch']
        
        flux_eb = generate_sample(flattened_lc, true_period, true_epoch)
        if flux_eb is None:
            missing_samples.append({'tic': star_id, 'class': 'EB', 'reason': 'generate_sample failed (likely NaN or length)'})
            
    print(f"Total Missing Samples Identified: {len(missing_samples)}")
    df = pd.DataFrame(missing_samples)
    print(df['class'].value_counts())
    print(df['reason'].value_counts())
    
    # Sector distribution actually evaluated
    print("\n--- Actual Sector Distribution Evaluated ---")
    sector_counts = []
    for star_id in valid_tics:
        clean_star_id = star_id.replace("TIC TIC ", "TIC ")
        tic_num = clean_star_id.replace("TIC ", "").strip()
        zero_padded = str(tic_num).zfill(16)
        pattern = os.path.join(CACHE_DIR, f'*{zero_padded}*', '*lc.fits')
        matches = glob.glob(pattern)
        sector_counts.append(min(5, len(matches)))
        
    s_dist = pd.Series(sector_counts).value_counts().sort_index()
    for k, v in s_dist.items():
        print(f"{k} Sector(s): {v}")
        
    # Are the two failed MAST targets among the missing?
    failed_mast = ['TIC 0422327572', 'TIC 0022457438']
    for t in failed_mast:
        found = False
        for m in missing_samples:
            if t in m['tic']:
                print(f"MAST Failed target {t} is among missing: {m['reason']}")
                found = True
        if not found:
            print(f"MAST Failed target {t} was SUCCESSFULLY processed.")

if __name__ == '__main__':
    main()
