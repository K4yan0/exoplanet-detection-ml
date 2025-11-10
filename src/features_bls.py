# Dans : src/features_bls.py

import numpy as np
from astropy.timeseries import BoxLeastSquares
# PAS d'import de src.preprocess

def generate_bls_features(flux_row):
    """
    Calcule les caractéristiques d'un transit potentiel en utilisant
    l'algorithme BLS. v28 - Correction de la normalisation.
    """

    times = np.arange(len(flux_row))

    # 1. Nettoyage ROBUSTE des données
    flux_cleaned = np.nan_to_num(flux_row, nan=0.0, posinf=0.0, neginf=0.0)

    # 2. DÉTRENDAGE CORRECT
    # L'erreur était de DIVISER par la médiane (qui est ~0).
    # Il faut juste SOUSTRAIRE la médiane pour centrer les données.
    flux_detrended = flux_cleaned - np.median(flux_cleaned)

    # 3. Durées de transit
    transit_durations = np.linspace(1, 25, 10).astype(int)

    try:
        # 4. Exécuter BLS sur les données propres et détrendées
        bls = BoxLeastSquares(times, flux_detrended)
        results = bls.autopower(transit_durations, frequency_factor=5.0)

        # 5. Extraire les "Features"
        best_index = np.argmax(results.power)

        peak_snr = results.power_snr[best_index]
        depth = results.depth[best_index]
        duration = results.duration[best_index]
        period = results.period[best_index]
        max_power = results.power[best_index]

        return [peak_snr, depth, duration, period, max_power]

    except Exception as e:
        # print(f"Avertissement : Echec de BLS : {e}")
        return [0.0, 0.0, 0.0, 0.0, 0.0]