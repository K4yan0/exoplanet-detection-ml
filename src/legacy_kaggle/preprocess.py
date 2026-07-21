from scipy.signal import savgol_filter

# 2. Définir la fonction (avec le VRAI code)
def normalize_flux(flux_row):
    """
    Normalise une seule courbe de lumière (une ligne) en utilisant
    un filtre Savitzky-Golay pour trouver la tendance.
    """
    # Paramètres du filtre (fenêtre de 101 points, polynôme d'ordre 3)
    window_length = 101
    polyorder = 3

    # 1. Trouver la tendance (C'EST LE CODE QUI MANQUAIT)
    trend = savgol_filter(flux_row, window_length, polyorder, mode='mirror')

    # 2. Normaliser en divisant le flux par la tendance
    # (flux / tendance) - 1  -> centre le flux autour de 0
    # On ajoute 1e-9 pour éviter la division par zéro
    return (flux_row / (trend + 1e-9)) - 1