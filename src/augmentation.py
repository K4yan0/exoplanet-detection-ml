import numpy as np
import random

def inject_transit_signal(
        flux_series,
        min_depth=0.05,
        max_depth=0.3,
        min_duration=15,
        max_duration=60,
        ingress_duration=5
):
    """
    Injects a synthetic trapezoidal transit signal into a flux series.

    Args:
        flux_series (np.array): The 1D normalized flux data.
        min_depth (float): Minimum depth (light blockage) of the transit.
        max_depth (float): Maximum depth.
        min_duration (int): Minimum duration (in time steps) of the transit.
        max_duration (int): Maximum duration.
        ingress_duration (int): Duration of the "ramp" down/up (slope of the trapezoid).

    Returns:
        np.array: The flux series with an injected transit.
    """
    series_len = len(flux_series)

    # --- 1. Randomize parameters ---
    # We randomize to make the model more robust
    depth = random.uniform(min_depth, max_depth)
    duration = random.randint(min_duration, max_duration)

    # Ensure ingress isn't longer than half the duration
    actual_ingress_len = min(ingress_duration, duration // 2)

    # --- 2. Find a random start time ---
    # Ensure the full transit fits within the series
    try:
        start_time = random.randint(0, series_len - duration - 1)
    except ValueError:
        # Handle case where duration is longer than series (shouldn't happen with our data)
        print("Warning: Duration may be too long for series. Skipping augmentation.")
        return flux_series

    end_time = start_time + duration

    # --- 3. Create the trapezoid dip ---
    # Start with the full depth (the bottom of the dip)
    dip_signal = np.full(duration, depth)

    # Create the ingress (ramp down from 0 to depth)
    dip_signal[:actual_ingress_len] = np.linspace(0, depth, actual_ingress_len)

    # Create the egress (ramp up from depth to 0)
    dip_signal[-actual_ingress_len:] = np.linspace(depth, 0, actual_ingress_len)

    # --- 4. Create the full signal to subtract ---
    # This is an array of zeros with the dip "pasted" in
    signal_to_subtract = np.zeros(series_len)
    signal_to_subtract[start_time:end_time] = dip_signal

    # --- 5. Inject the signal ---
    new_flux_series = flux_series - signal_to_subtract

    return new_flux_series