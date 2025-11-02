import numpy as np
from astropy.timeseries import BoxLeastSquares

def generate_bls_features(flux_row):
    """
    Calcule les caractéristiques d'un transit potentiel en utilisant
    l'algorithme BoxLeastSquares (BLS) de la NASA.

    Retourne des features statistiques sur le meilleur "creux" trouvé.
    """
    times = np.arange(len(flux_row))
    flux = flux_row

    # Normaliser le flux pour que la "profondeur" soit significative
    flux_norm = flux - np.mean(flux)

    # Définir les durées de transit à tester (en "bacs" d'index)
    # Ex: de 1 bac (très court) à 100 bacs
    transit_durations = np.linspace(1, 100, 10).astype(int)

    try:
        # 1. Exécuter l'algorithme BLS
        # Il teste toutes les périodes, durées, et heures de départ
        bls = BoxLeastSquares(times, flux_norm)

        # 'autopower' trouve le meilleur modèle de "boîte" (creux)
        results = bls.autopower(transit_durations, frequency_factor=5.0)

        # 2. Extraire les "Features" du meilleur résultat
        # 'results' contient les propriétés du *meilleur* creux trouvé

        # L'index du meilleur pic de puissance
        best_index = np.argmax(results.power)

        # La "signifiance" statistique du creux (très important !)
        # C'est l'équivalent de notre 'peak_significance'
        peak_snr = results.power_snr[best_index]

        # La profondeur du creux
        depth = results.depth[best_index]

        # La durée du creux (en bacs)
        duration = results.duration[best_index]

        # La période du creux (en bacs/jours)
        period = results.period[best_index]

        return [peak_snr, depth, duration, period]

    except Exception as e:
        # En cas d'échec (ex: données invalides)
        return [0.0, 0.0, 0.0, 0.0]