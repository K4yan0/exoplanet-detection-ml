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
        
    # Drop corrupted sectors with negative background flux to prevent inversion during stitch()
    clean_lcs = [l for l in lc_collection if np.nanmedian(l.flux.value) > 0]
    if len(clean_lcs) == 0:
        return {'success': False, 'error': 'All downloaded sectors had corrupted negative flux.'}
        
    lc = lk.LightCurveCollection(clean_lcs).stitch()
    # Clean the data by removing outliers before flattening
    lc = lc.remove_nans().remove_outliers(sigma_upper=4, sigma_lower=4)
    
    # CRITIQUE 3: Detrending (Stellar Variability)
    # We use a Savitzky-Golay filter (flatten), but we MUST use a wide window (401 points = ~13 hours)
    # If we use the default 101, it will erase 4-hour transits entirely!
    flattened_lc = lc.flatten(window_length=401)
    
    # Dynamically calculate the maximum searchable period based on the observation baseline
    time_span = float(lc.time[-1].value - lc.time[0].value)
    max_period = max(10.0, min(time_span / 2, 60.0)) # Cap at 60 days to ensure safety
    
    # We EXPLICITLY pass a period grid (20,000 points) and a duration grid (5 points).
    # Total combinations = 100,000. We also pass a massive frequency_factor (500) to 
    # bypass a bug in lightkurve that accidentally evaluates autopower limits even 
    # when an explicit period array is provided!
    periodogram = flattened_lc.to_periodogram(
        method='bls', 
        period=np.linspace(1, max_period, 20000),
        duration=np.linspace(0.02, 0.2, 5),
        frequency_factor=500
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
