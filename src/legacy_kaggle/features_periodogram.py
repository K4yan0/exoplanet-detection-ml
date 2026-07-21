import numpy as np
from astropy.timeseries import LombScargle

def generate_periodogram_features(flux_row, min_period=0.1, max_period=100):
    times = np.arange(len(flux_row))
    flux = flux_row - np.mean(flux_row) # Normaliser

    try:
        frequency, power = LombScargle(times, flux).autopower(
            minimum_frequency=1/max_period,
            maximum_frequency=1/min_period,
            samples_per_peak=10 # Haute résolution
        )

        # 2. Nos "Features" (Nos mesures)
        max_power = np.max(power)
        mean_power = np.mean(power)
        std_power = np.std(power)

        # C'est la feature la plus importante :
        # "À quel point le pic est-il plus haut que le bruit de fond ?"
        if std_power > 0:
            peak_significance = (max_power - mean_power) / std_power
        else:
            peak_significance = 0.0

        return [max_power, mean_power, std_power, peak_significance]

    except Exception as e:
        return [0.0, 0.0, 0.0, 0.0] # Échec