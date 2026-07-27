import pandas as pd
# Vous pourriez ajouter numpy, scipy.stats, etc. ici si besoin.

def create_baseline_features(normalized_flux_df):
    """
    Prend un DataFrame de flux normalisés (N lignes, 3197 colonnes)
    et retourne un DataFrame de caractéristiques (N lignes, K colonnes)
    pour le modèle Random Forest.
    """

    # Crée un DataFrame vide pour stocker les nouvelles features
    features_df = pd.DataFrame(index=normalized_flux_df.index)

    print("Début du feature engineering...")

    # --- Caractéristiques statistiques de base ---
    # Ces features décrivent la distribution globale du flux
    features_df['mean'] = normalized_flux_df.mean(axis=1)
    features_df['std'] = normalized_flux_df.std(axis=1)
    features_df['min'] = normalized_flux_df.min(axis=1)
    features_df['max'] = normalized_flux_df.max(axis=1)
    features_df['median'] = normalized_flux_df.median(axis=1)
    features_df['skew'] = normalized_flux_df.skew(axis=1) # Asymétrie (tendance des creux vs pics)
    features_df['kurt'] = normalized_flux_df.kurt(axis=1) # Aplatissement (à quel point les creux sont "piqués")

    # --- Caractéristiques spécifiques au "transit" ---
    # On essaie de quantifier la "profondeur du creux"

    # Moyenne des 10 points de flux les plus bas
    # C'est un bon indicateur de la profondeur d'un transit potentiel
    features_df['mean_of_10_lowest'] = normalized_flux_df.apply(
        lambda row: row.nsmallest(10).mean(), axis=1
    )

    # Moyenne des 10 points de flux les plus hauts
    features_df['mean_of_10_highest'] = normalized_flux_df.apply(
        lambda row: row.nlargest(10).mean(), axis=1
    )

    # Différence entre les extrêmes (une mesure simple de l'amplitude)
    features_df['range'] = features_df['max'] - features_df['min']

    # (Feature plus avancée si vous voulez)
    # Autocorrélation pour trouver la périodicité (plus complexe,
    # mais c'est la feature la plus importante ! On peut la garder pour plus tard)
    # ...

    print(f"Création de {features_df.shape[1]} features terminée.")

    # Nettoyer les NaN/Inf qui pourraient résulter des calculs
    features_df = features_df.fillna(0).replace([float('inf'), float('-inf')], 0)

    return features_df