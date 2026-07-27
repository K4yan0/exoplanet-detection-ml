import numpy as np
import random

def inject_transit_signal(
        flux_series,
        min_depth=0.05,
        max_depth=0.3,
        min_duration=15,
        max_duration=60,
        ingress_duration=5,
        noise_level=0.0 # <--- NEW PARAMETER
):
    """
    Injects a synthetic trapezoidal transit signal into a flux series
    and adds random noise to make it realistic.

    Args:
        flux_series (np.array): The 1D normalized flux data.
        min_depth (float): Minimum depth (light blockage) of the transit.
        max_depth (float): Maximum depth.
        min_duration (int): Minimum duration (in time steps) of the transit.
        max_duration (int): Maximum duration.
        ingress_duration (int): Duration of the "ramp" down/up.
        noise_level (float): Std deviation of the Gaussian noise to add.

    Returns:
        np.array: The flux series with an injected, noisy transit.
    """
    series_len = len(flux_series)

    # --- 1. Randomize parameters ---
    depth = random.uniform(min_depth, max_depth)
    duration = random.randint(min_duration, max_duration)
    actual_ingress_len = min(ingress_duration, duration // 2)

    # --- 2. Find a random start time ---
    try:
        start_time = random.randint(0, series_len - duration - 1)
    except ValueError:
        print("Warning: Duration may be too long for series. Skipping augmentation.")
        return flux_series

    end_time = start_time + duration

    # --- 3. Create the trapezoid dip ---
    dip_signal = np.full(duration, depth)
    dip_signal[:actual_ingress_len] = np.linspace(0, depth, actual_ingress_len)
    dip_signal[-actual_ingress_len:] = np.linspace(depth, 0, actual_ingress_len)

    # --- 4. Create the full signal to subtract ---
    signal_to_subtract = np.zeros(series_len)
    signal_to_subtract[start_time:end_time] = dip_signal

    # --- 5. Inject the signal ---
    new_flux_series = flux_series - signal_to_subtract

    # --- 6. ADD NOISE (THE CRITICAL NEW STEP) ---
    # Add Gaussian noise to the *entire* series to simulate
    # the noisy reality of astronomical data.
    if noise_level > 0:
        noise = np.random.normal(loc=0.0, scale=noise_level, size=series_len)
        new_flux_series = new_flux_series + noise

    return new_flux_series