# 🪐 Détection d'Exoplanètes : Baseline (Random Forest) vs. Deep Learning (CNN 1D)

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

Ce projet compare deux approches de Machine Learning pour la détection d'exoplanètes à partir des courbes de lumière (flux lumineux) des télescopes spatiaux TESS et Kepler.

## 1. L'Hypothèse Scientifique 🎯

Un "transit" d'exoplanète (le passage de la planète devant son étoile) crée une signature morphologique identifiable : un "creux" périodique et de courte durée dans la courbe de lumière de l'étoile.

Ce projet vise à entraîner des modèles de Machine Learning à reconnaître cette signature pour la différencier du simple bruit de fond ou d'autres variations stellaires.

## 2. La Tâche de Machine Learning 🤖

* **Tâche** : Classification Binaire Supervisée.
* **Input (X)** : Une série temporelle (courbe de lumière, ex: `~3000+` colonnes `FLUX`).
* **Output (Y)** : Prédiction à 2 classes : `0` (Pas de Planète) ou `1` (Planète Détectée).

## 3. Méthodologie (Double Modèle) 🧠

Pour évaluer la meilleure approche, nous implémentons et comparons deux modèles distincts :

### Partie A : Le Modèle de Baseline (Random Forest)

* **Objectif** : Établir un score de référence.
* **Modèle** : `RandomForestClassifier`.
* **Approche** : Basée sur l'**ingénierie de caractéristiques (feature engineering)**. Nous traduisons manuellement la série temporelle brute de 3000+ points en un petit ensemble de caractéristiques descriptives (ex: profondeur moyenne des creux, écart-type du flux, périodicité via autocorrélation).

### Partie B : Le Modèle Avancé (Deep Learning)

* **Objectif** : Battre la baseline en utilisant l'apprentissage de caractéristiques.
* **Modèle** : Réseau de Neurones Convolutif 1D (CNN 1D).
* **Approche** : **Aucun feature engineering**. La courbe de lumière brute (vecteur de `~3000+` points) est fournie directement en entrée. Le CNN agit comme un détecteur de "forme" (shape) automatique, apprenant à identifier la morphologie du transit par lui-même.

## 4. Les Données ✅

* **Source** : Télescope spatial TESS (ou Kepler).
* **Dataset Suggéré** : [Exoplanet Hunting in TESS Light Curves (Kaggle)](https://www.kaggle.com/datasets/keplersmachines/kepler-labelled-time-series-data)
* **Format** : `train.csv` et `test.csv` contenant un `LABEL` (Y) et `~3000+` colonnes `FLUX` (X).

**Note importante** : Conformément aux bonnes pratiques, les fichiers de données brutes (`.csv`) sont listés dans le `.gitignore` et ne doivent pas être "pushés" sur ce dépôt. Pour exécuter le projet, veuillez télécharger les données depuis la source Kaggle et les placer dans le dossier `data/raw/`.

## 5. Installation

Pour configurer votre environnement local et exécuter ce projet :

1.  **Clonez le dépôt :**
    ```bash
    git clone [https://github.com/VOTRE_NOM/exoplanet-detection-ml.git](https://github.com/VOTRE_NOM/exoplanet-detection-ml.git)
    cd exoplanet-detection-ml
    ```

2.  **Créez un environnement virtuel (recommandé) :**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Sur Windows: .\venv\Scripts\activate
    ```

3.  **Installez les dépendances :**
    ```bash
    pip install -r requirements.txt
    ```

## 6. Utilisation et Workflow

Ce projet utilise un workflow Git structuré :

* `main` : Branche de production, protégée. Les merges ne se font que depuis `develop`.
* `develop` : Branche d'intégration principale (branche par défaut).
* `feature/...` : Branches de travail pour les nouvelles fonctionnalités.

Tout nouveau code doit être ajouté via une **Pull Request** d'une branche `feature/` vers `develop`.

### Reproduire les résultats

Les notebooks sont numérotés et doivent être exécutés dans l'ordre :

1.  **`notebooks/01_EDA_and_Preprocessing.ipynb`** :
    * Chargement des données.
    * Analyse exploratoire (EDA).
    * Nettoyage, imputation des NaN et normalisation (detrending) des courbes de lumière.

2.  **`notebooks/02_RF_Baseline_Model.ipynb`** :
    * Génération des caractéristiques (feature engineering) via `src/features.py`.
    * Entraînement et évaluation du `RandomForestClassifier`.

3.  **`notebooks/03_CNN_1D_Model.ipynb`** :
    * Préparation des données brutes (mise en forme pour Keras/TensorFlow).
    * Construction, entraînement et évaluation du modèle CNN 1D.

4.  **`notebooks/04_Model_Comparison.ipynb`** :
    * Comparaison finale des deux modèles.
    * Génération des matrices de confusion, calcul des F1-Scores et des courbes/scores PR-AUC.

## 7. Structure du Dépôt

## 8. Résultats Attendus (Point 6)

L'évaluation finale comparera les deux modèles. Le dataset étant très déséquilibré (beaucoup plus de non-planètes que de planètes), l'Accuracy n'est pas une bonne métrique.

Nous nous concentrerons sur :
* **PR-AUC** (Aire sous la courbe Précision-Rappel) : La métrique principale pour ce type de problème.
* **F1-Score** : L'équilibre entre Précision et Rappel.
* **Matrice de Confusion** : Pour analyser les Faux Positifs et Faux Négatifs.

L'hypothèse est que le **CNN 1D** obtiendra un score PR-AUC et F1 supérieur, car il est spécifiquement conçu pour la détection de motifs dans des séquences, surpassant ainsi les caractéristiques conçues manuellement.

*(TODO: Insérer ici un tableau résumé des résultats finaux)*

## 9. Licence

Ce projet est publié sous la Licence MIT. Voir le fichier `LICENSE` pour plus de détails.