import os
import json
import time
import glob
import lightkurve as lk
from lightkurve.utils import LightkurveError
import numpy as np

CACHE_DIR = os.path.expanduser('~/.cache/lightkurve/mastDownload/TESS')
MANIFEST_PATH = 'data/multisector_manifest.json'

def get_cached_sectors_count(tic_id_str):
    tic_num = tic_id_str.replace("TIC ", "").replace("TIC ", "").strip()
    zero_padded = str(tic_num).zfill(16)
    pattern = os.path.join(CACHE_DIR, f'*{zero_padded}*', '*lc.fits')
    matches = glob.glob(pattern)
    return len(matches)

def build_target_list():
    """Identifies the exact 591 physical targets (300 Planets, 291 EBs) used in V1."""
    with open('data/tess_positive_targets.json', 'r') as f:
        targets_pos = json.load(f)
    with open('data/tess_eb_targets.json', 'r') as f:
        targets_eb = json.load(f)
        
    cohort = []
    
    # Planets
    pos_count = 0
    for star_id in targets_pos.keys():
        if pos_count >= 300: break
        if get_cached_sectors_count(star_id) > 0:
            cohort.append(star_id)
            pos_count += 1
            
    # EBs
    eb_count = 0
    for star_id in targets_eb.keys():
        if eb_count >= 291: break
        if get_cached_sectors_count(star_id) > 0:
            cohort.append(star_id)
            eb_count += 1
            
    return cohort

def execute_download_with_backoff(search_item, max_retries=3):
    for attempt in range(max_retries):
        try:
            # We don't need to check cache manually because lightkurve checks its own cache
            search_item.download()
            return True, attempt + 1
        except Exception as e:
            time.sleep((2 ** attempt)) # Exponential backoff: 1s, 2s, 4s
    return False, max_retries

def main():
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH, 'r') as f:
            manifest = json.load(f)
    else:
        print("Identifying exact V1 target cohort...")
        cohort = build_target_list()
        print(f"Cohort identified: {len(cohort)} physical targets.")
        
        manifest = {}
        for tic in cohort:
            manifest[tic] = {
                'required_sectors': 5,
                'sectors_already_cached': get_cached_sectors_count(tic),
                'sectors_missing': 0, # Will be computed
                'download_status': 'PENDING',
                'attempt_count': 0,
                'total_available_in_mast': 0
            }
            
    print("Beginning acquisition phase...")
    targets_completed = 0
    targets_failed = 0
    
    for tic, info in manifest.items():
        if info['download_status'] == 'COMPLETE':
            targets_completed += 1
            continue
            
        print(f"Processing {tic}...")
        clean_tic = tic.replace("TIC ", "").replace("TIC ", "").strip()
        
        # 1. Search MAST
        try:
            search = lk.search_lightcurve(f"TIC {clean_tic}", mission='TESS', author='SPOC')
            info['total_available_in_mast'] = len(search)
        except Exception as e:
            print(f"  MAST Search failed for {tic}: {e}")
            time.sleep(5) # Cooldown
            continue
            
        # 2. Determine how many we actually need (up to 5)
        to_download = min(5, len(search))
        info['required_sectors'] = to_download
        
        success = True
        for i in range(to_download):
            search_item = search[i]
            # Download individually to prevent one failure from dropping the others
            ok, attempts = execute_download_with_backoff(search_item)
            info['attempt_count'] += attempts
            if not ok:
                success = False
                print(f"  Failed to download sector at index {i} for {tic}")
                
        # 3. Update cached count
        info['sectors_already_cached'] = get_cached_sectors_count(tic)
        info['sectors_missing'] = max(0, to_download - info['sectors_already_cached'])
        
        if success and info['sectors_missing'] == 0:
            info['download_status'] = 'COMPLETE'
            targets_completed += 1
        else:
            info['download_status'] = 'FAILED'
            targets_failed += 1
            
        # Save incrementally
        with open(MANIFEST_PATH, 'w') as f:
            json.dump(manifest, f, indent=4)
            
        # Polite delay to respect MAST
        time.sleep(2)
        
    print("\n--- ACQUISITION MANIFEST SUMMARY ---")
    print(f"Total Cohort: {len(manifest)}")
    print(f"COMPLETE: {targets_completed}")
    print(f"FAILED/INCOMPLETE: {targets_failed}")
    
    # Interesting observational data (sector availability)
    availabilities = [info['total_available_in_mast'] for info in manifest.values()]
    print("\nSector Availability Distribution:")
    for i in range(1, 6):
        count = sum(1 for a in availabilities if a == i)
        print(f"  Exactly {i} sector(s): {count} targets")
    print(f"  >5 sectors: {sum(1 for a in availabilities if a > 5)} targets")
    
    if targets_failed == 0:
        print("\nAll required multi-sector data successfully acquired. Ready to build Exp 3.")
    else:
        print(f"\nWARNING: {targets_failed} targets failed to download completely. Cohort is incomplete.")

if __name__ == '__main__':
    main()
