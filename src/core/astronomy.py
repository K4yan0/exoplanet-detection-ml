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
    
    # Use minimum and maximum period kwargs to let lightkurve calculate the optimal grid
    # frequency_factor controls grid density. 5 is a safe, fast balance for multi-sector data.
    periodogram = flattened_lc.to_periodogram(
        method='bls', 
        minimum_period=1.0, 
        maximum_period=max_period, 
        frequency_factor=5
    )
    best_period = periodogram.period_at_max_power
    best_epoch = periodogram.transit_time_at_max_power
    
    duration_hours = float(periodogram.duration_at_max_power.value) * 24
    depth_ppt = float(periodogram.depth_at_max_power) * 1000 # Parts per thousand
    
    folded_lc = flattened_lc.fold(period=best_period, epoch_time=best_epoch)
    binned_lc = folded_lc.bin(bins=2000)
    
    flux = binned_lc.flux.value
    
    # Fill any NaNs created by binning
    if np.isnan(flux).any():
        flux = pd.Series(flux).interpolate(limit_direction='both').values
        
    # If lightkurve dropped empty bins, force it to exactly 2000 points via interpolation
    if len(flux) != 2000:
        from scipy.interpolate import interp1d
        x_old = np.linspace(0, 1, len(flux))
        x_new = np.linspace(0, 1, 2000)
        flux = interp1d(x_old, flux, kind='linear', fill_value='extrapolate')(x_new)
        
    return {
        'success': True,
        'flux': flux,
        'period': float(best_period.value),
        'duration_hours': duration_hours,
        'depth_ppt': depth_ppt
    }
