import numpy as np
from astropy.timeseries import BoxLeastSquares
from src.preprocess import normalize_flux

def generate_bls_features(flux_row):
    """
    Calcule les caractéristiques d'un transit potentiel en utilisant
    l'algorithme BoxLeastSquares (BLS) de la NASA.

    Retourne des features statistiques sur le meilleur "creux" trouvé.
    """
    try:
        flux_normalized = normalize_flux(flux_row)
    except Exception as e:
        print(f"Avertissement : Échec de normalize_flux : {e}")
        return [0.0, 0.0, 0.0, 0.0]

    times = np.arange(len(flux_normalized))

    # Définir les durées de transit à tester (en "bacs" d'index)
    transit_durations = np.linspace(1, 25, 10).astype(int)

    try:
        # 3. Exécuter BLS sur les données PROPRES
        bls = BoxLeastSquares(times, flux_normalized) # <-- Utiliser flux_normalized

        results = bls.autopower(transit_durations, frequency_factor=5.0)

        # 4. Extraire les "Features" du meilleur résultat
        best_index = np.argmax(results.power)

        peak_snr = results.power_snr[best_index]
        depth = results.depth[best_index]
        duration = results.duration[best_index]
        period = results.period[best_index]

        # On ajoute la "puissance" brute, au cas où
        max_power = results.power[best_index]

        return [peak_snr, depth, duration, period, max_power]

    except Exception as e:
        # En cas d'échec de BLS
        return [0.0, 0.0, 0.0, 0.0, 0.0]