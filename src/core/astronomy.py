import numpy as np
import pandas as pd
import lightkurve as lk

def get_folded_lightcurve(star_id):
    """Fetches TESS data, runs BLS, and returns folded light curve metrics."""
    search_result = lk.search_lightcurve(star_id, mission='TESS', author='SPOC')
    if len(search_result) == 0:
        return {'success': False, 'error': f'No SPOC data found for {star_id}. Try a different star!'}
        
    lc = search_result[0].download()
    if lc is None:
        return {'success': False, 'error': 'Download failed from NASA MAST.'}
        
    flattened_lc = lc.flatten(window_length=101)
    periodogram = flattened_lc.to_periodogram(method='bls', period=np.linspace(1, 20, 100000))
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
