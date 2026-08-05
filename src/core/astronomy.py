import numpy as np
import pandas as pd
import lightkurve as lk

def get_folded_lightcurve(star_id):
    """Fetches TESS data, runs BLS, and returns folded light curve metrics."""
    # Fetch up to 5 sectors to prevent massive API timeouts on the frontend,
    # but still give us ~135 days of baseline data.
    search_result = lk.search_lightcurve(star_id, mission='TESS', author='SPOC')
    if len(search_result) == 0:
        return {'success': False, 'error': f'No SPOC data found for {star_id}. Try a different star!'}
        
    lc_collection = search_result[:5].download_all()
    if lc_collection is None or len(lc_collection) == 0:
        return {'success': False, 'error': 'Download failed from NASA MAST.'}
        
    lc = lc_collection.stitch()
    flattened_lc = lc.flatten(window_length=101)
    
    # Dynamically calculate the maximum searchable period based on the observation baseline
    time_span = float(lc.time[-1].value - lc.time[0].value)
    max_period = max(10.0, min(time_span / 2, 100.0))
    
    periodogram = flattened_lc.to_periodogram(method='bls', period=np.linspace(1, max_period, 100000))
    best_period = periodogram.period_at_max_power
    best_epoch = periodogram.transit_time_at_max_power
    
    duration_hours = float(periodogram.duration_at_max_power.value) * 24
    depth_ppt = float(periodogram.depth_at_max_power) * 1000 # Parts per thousand
    
    folded_lc = flattened_lc.fold(period=best_period, epoch_time=best_epoch)
    binned_lc = folded_lc.bin(bins=2000)
    
    flux = binned_lc.flux.value
    if np.isnan(flux).any():
        flux = pd.Series(flux).interpolate(limit_direction='both').values
        
    if len(flux) != 2000 or np.isnan(flux).any():
        return {'success': False, 'error': 'Could not extract a clean 2000-point array.'}
        
    return {
        'success': True,
        'flux': flux,
        'period': float(best_period.value),
        'duration_hours': duration_hours,
        'depth_ppt': depth_ppt
    }
