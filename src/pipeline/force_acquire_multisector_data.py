import os
import json
import time
import glob
import lightkurve as lk
from lightkurve.utils import LightkurveError

CACHE_DIR = os.path.expanduser('~/.cache/lightkurve/mastDownload/TESS')

def get_cached_sectors_count(tic_id_str):
    tic_num = tic_id_str.replace("TIC ", "").replace("TIC ", "").strip()
    zero_padded = str(tic_num).zfill(16)
    pattern = os.path.join(CACHE_DIR, f'*{zero_padded}*', '*lc.fits')
    return len(glob.glob(pattern))

def execute_download_with_backoff(search_item, max_retries=3):
    for attempt in range(max_retries):
        try:
            search_item.download()
            return True, attempt + 1
        except Exception as e:
            time.sleep((2 ** attempt))
    return False, max_retries

def main():
    print("Force Acquiring Multisector Data for Exp 7...")
    with open('data/tess_positive_targets.json', 'r') as f:
        targets_pos = json.load(f)
    with open('data/tess_eb_targets.json', 'r') as f:
        targets_eb = json.load(f)
        
    # Build combined list of targets we care about
    cohort = []
    for tic in targets_pos.keys(): cohort.append((tic, 'planet'))
    for tic in targets_eb.keys(): cohort.append((tic, 'eb'))
    
    # We want EXACTLY 100 Planets and 100 EBs that have 5 sectors to save download time.
    planets_found = 0
    ebs_found = 0
    
    manifest = {}
    
    for tic, t_type in cohort:
        if t_type == 'planet' and planets_found >= 100: continue
        if t_type == 'eb' and ebs_found >= 100: continue
            
        clean_tic = tic.replace("TIC ", "").replace("TIC ", "").strip()
        
        try:
            search = lk.search_lightcurve(f"TIC {clean_tic}", mission='TESS', author='SPOC')
            total_avail = len(search)
        except Exception as e:
            time.sleep(2)
            continue
            
        if total_avail >= 5:
            # We found a target with 5+ sectors! Download the first 5.
            print(f"Target {tic} ({t_type}) has {total_avail} sectors. Downloading 5...")
            success = True
            for i in range(5):
                search_item = search[i]
                ok, _ = execute_download_with_backoff(search_item)
                if not ok:
                    success = False
                    break
            
            if success:
                manifest[tic] = {
                    'total_available_in_mast': total_avail,
                    'type': t_type
                }
                if t_type == 'planet': planets_found += 1
                if t_type == 'eb': ebs_found += 1
                
                print(f"--> Success! Planets: {planets_found}/100, EBs: {ebs_found}/100")
                
                # Save incrementally
                with open('data/multisector_manifest.json', 'w') as f:
                    json.dump(manifest, f, indent=4)
                    
        time.sleep(1) # Be nice to MAST
        
    print("\nAcquisition Complete.")
    print(f"Total targets with 5 sectors acquired: {len(manifest)}")

if __name__ == '__main__':
    main()
