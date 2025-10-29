# Dans : src/phase_folding.py

import numpy as np
from astropy.timeseries import LombScargle
from scipy.stats import binned_statistic

# --- Étape 1 : Trouver la meilleure période (Lomb-Scargle) ---

def find_best_period(times, flux):
    """
    Estime la période la plus probable en utilisant Lomb-Scargle.
    """
    flux_norm = flux - np.mean(flux)
    min_period = 0.1
    max_period = 100

    frequency, power = LombScargle(times, flux_norm).autopower(
        minimum_frequency=1/max_period,
        maximum_frequency=1/min_period,
        samples_per_peak=10
    )

    best_frequency = frequency[np.argmax(power)]
    best_period = 1 / best_frequency

    return best_period

# --- Étape 2 : Plier la phase ---

def fold_phase(times, flux, period, t0=0):
    """
    Replie une série temporelle sur une période donnée.
    """
    phases = ((times - t0) / period) % 1
    return phases, flux

# --- Étape 3 : Binner (Agréger) les données pliées ---

def bin_folded_data(phases, flux, n_bins=500):
    """
    Moyenne les données repliées dans un nombre fixe de 'bacs'.
    """
    bin_means, bin_edges, binnumber = binned_statistic(
        phases,
        flux,
        statistic='mean',
        bins=n_bins,
        range=(0, 1)
    )

    if np.any(np.isnan(bin_means)):
        global_mean = np.nanmean(bin_means)
        bin_means = np.nan_to_num(bin_means, nan=global_mean)

    return bin_means

# --- Étape 4 : Le Pipeline de Pré-traitement Complet ---

def create_folded_lightcurve(flux_row, n_bins=500):
    """
    Pipeline complet : 3197 points bruts -> 500 points binnés.
    """
    times = np.arange(len(flux_row))
    flux = flux_row

    try:
        period = find_best_period(times, flux)
        phases, flux_folded = fold_phase(times, flux, period)
        binned_flux = bin_folded_data(phases, flux_folded, n_bins=n_bins)
    except Exception as e:
        binned_flux = np.zeros(n_bins)

    mean = np.mean(binned_flux)
    std = np.std(binned_flux)

    if std == 0:
        return np.zeros(n_bins)

    normalized_flux = (binned_flux - mean) / std
    return normalized_flux