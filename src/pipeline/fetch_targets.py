from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive
import json
import os

def fetch_tess_targets():
    print("Querying NASA Exoplanet Archive (PSCompPars table) via astroquery...")
    
    # Query the Planetary Systems Composite Parameters (pscomppars) table.
    # We filter by discovery facility = TESS and ensure period/epoch data is not null.
    # This automatically downloads the data into an Astropy Table object.
    table = NasaExoplanetArchive.query_criteria(
        table="pscomppars",
        where="disc_facility like '%TESS%' and pl_orbper is not null and pl_tranmid is not null"
    )
    
    print(f"Found {len(table)} confirmed TESS planets with period and epoch data!")
    
    targets = {}
    
    # Loop through the table and build our dictionary
    for row in table:
        star_name = row['hostname']
        tic_id = row['tic_id']
        
        # Astroquery returns Astropy Quantities (which have units attached like "days").
        # We need to extract just the raw float values to do normal math and save to JSON.
        period = row['pl_orbper']
        if hasattr(period, 'value'): 
            period = period.value
            
        epoch_bjd = row['pl_tranmid']
        if hasattr(epoch_bjd, 'value'): 
            epoch_bjd = epoch_bjd.value
            
        planet_name = row['pl_name']
        
        # CRITICAL: Convert BJD (Barycentric Julian Date) to BTJD (Barycentric TESS Julian Date)
        # NASA archives usually store transit midpoints in BJD.
        # lightkurve and TESS standard pipelines use BTJD.
        # Formula: BTJD = BJD - 2457000.0
        epoch_btjd = float(epoch_bjd) - 2457000.0
        
        # lightkurve prefers TIC IDs when searching for TESS data (e.g., 'TIC 12345678')
        # If a TIC ID is missing for some reason, we fall back to the host star name.
        # It's common to format it exactly as a string "TIC <ID>"
        if tic_id and not np.ma.is_masked(tic_id): # Check if astropy masked array value is valid
            target_id = f"TIC {tic_id}"
        else:
            target_id = star_name
            
        # We only need one transit parameter set per star for our single-target test pipeline.
        # If a star has multiple planets, this will just grab the first one.
        if target_id not in targets:
            targets[target_id] = {
                'label': 1,
                'period': float(period),
                'epoch': float(epoch_btjd),
                'planet_name': str(planet_name)
            }
            
    # Define save path (we put it in the data folder)
    save_dir = os.path.join('data')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'tess_positive_targets.json')
    
    # Save the dictionary as a JSON file
    with open(save_path, 'w') as f:
        json.dump(targets, f, indent=4)
        
    print(f"\nSuccessfully saved {len(targets)} unique TESS target stars to {save_path}")

if __name__ == "__main__":
    # We import numpy here just for the masked array check above
    import numpy as np
    fetch_tess_targets()
