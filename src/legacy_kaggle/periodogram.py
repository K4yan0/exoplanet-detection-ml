import numpy as np
from astropy.timeseries import LombScargle

def generate_periodogram(flux_row, n_bins=1000, min_period=0.1, max_period=100):
    # 1. Préparer les données
    times = np.arange(len(flux_row))
    flux = flux_row

    # Normaliser le flux est crucial pour Lomb-Scargle
    flux_norm = flux - np.mean(flux)

    try:
        # 2. Définir les fréquences à tester
        # Nous avons besoin d'une GRILLE FIXE pour que tous les outputs
        # aient la même taille (n_bins)
        min_freq = 1 / max_period
        max_freq = 1 / min_period

        # Grille linéaire de fréquences
        frequency_bins = np.linspace(min_freq, max_freq, n_bins)

        # 3. Calculer la puissance pour CHAQUE fréquence
        # C'est ce qui remplace .autopower()
        power = LombScargle(times, flux_norm).power(frequency_bins)

        # 4. Normaliser le périodogramme (la sortie)
        # Centrer-réduire pour le CNN
        mean = np.mean(power)
        std = np.std(power)

        if std == 0:
            # Si le périodogramme est plat (pas de signal)
            return np.zeros(n_bins)

        normalized_power = (power - mean) / std
        return normalized_power

    except Exception as e:
        # En cas d'échec (ex: données invalides), retourner des zéros
        print(f"Avertissement : Echec de Lomb-Scargle. {e}")
        return np.zeros(n_bins)