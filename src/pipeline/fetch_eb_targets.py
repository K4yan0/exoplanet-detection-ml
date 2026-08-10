import json
import os
import numpy as np
from astroquery.vizier import Vizier

def fetch_eb_targets():
    print("Querying VizieR for the Villanova TESS Eclipsing Binary Catalog (J/ApJS/258/16)...")
    
    # Configure VizieR query
    v = Vizier(columns=['TIC', 'Per', 'BJD0', 'Morph'])
    # Let's get 300 EBs so we have a good sample size for training
    v.ROW_LIMIT = 300 
    
    catalogs = v.get_catalogs('J/ApJS/258/16')
    if len(catalogs) == 0:
        print("Failed to fetch catalog from VizieR.")
        return
        
    table = catalogs[0]
    print(f"Found {len(table)} Eclipsing Binaries!")
    
    targets = {}
    
    for row in table:
        # Format TIC correctly
        tic_id = str(row['TIC']).zfill(10) # Ensure 10-digit TIC
        target_id = f"TIC {tic_id}"
        
        period = row['Per']
        epoch_bjd = row['BJD0']
        
        # We must convert BJD (Barycentric Julian Date) to BTJD (Barycentric TESS Julian Date)
        # NASA/VizieR uses BJD. TESS/lightkurve uses BTJD = BJD - 2457000.0
        epoch_btjd = float(epoch_bjd) - 2457000.0
        
        # Only add valid floats
        if not np.ma.is_masked(period) and not np.ma.is_masked(epoch_bjd):
            targets[target_id] = {
                'label': 2, # 2 = Eclipsing Binary
                'period': float(period),
                'epoch': float(epoch_btjd),
                'planet_name': "Eclipsing Binary"
            }
                
    save_dir = os.path.join('data')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'tess_eb_targets.json')
    
    with open(save_path, 'w') as f:
        json.dump(targets, f, indent=4)
        
    print(f"\nSuccessfully saved {len(targets)} unique EB target stars to {save_path}")

if __name__ == "__main__":
    fetch_eb_targets()
