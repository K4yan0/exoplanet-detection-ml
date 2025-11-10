# 🪐 Détection d'Exoplanètes : Une Investigation Itérative (RF, CNN, BLS)

![Statut du Projet: Terminé - Données non concluantes](https://img.shields.io/badge/Statut-Termin%C3%A9%20(Donn%C3%A9es%20Non%20Concluantes)-red.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

Ce projet documente une investigation approfondie pour détecter des exoplanètes à partir des courbes de lumière (flux lumineux) des télescopes spatiaux. Commençant par une simple comparaison entre Random Forest et CNN, le projet a évolué en une série de pivots méthodologiques pour répondre aux échecs de chaque approche, aboutissant à une conclusion définitive sur la qualité du jeu de données.

## 1. L'Hypothèse Scientifique 🎯

L'hypothèse de départ était qu'un "transit" d'exoplanète crée une signature morphologique identifiable : un "creux" périodique et de courte durée dans la courbe de lumière de l'étoile. Le but était d'entraîner des modèles à reconnaître cette signature.

## 2. La Tâche de Machine Learning 🤖

* **Tâche** : Classification Binaire Supervisée (fortement déséquilibrée).
* **Input (X)** : Une série temporelle (courbe de lumière, `~3000+` colonnes `FLUX`).
* **Output (Y)** : Prédiction à 2 classes : `0` (Pas de Planète) ou `1` (Planète Détectée).

## 3. Méthodologie et Découvertes Itératives 🧠

Ce qui a commencé comme une simple comparaison (Parties A & B) est devenu une enquête en plusieurs étapes (Parties C, D, E) pour comprendre pourquoi nos modèles échouaient.

### Partie A : Baseline (Random Forest sur Features Simples)

* **Approche** : Ingénierie de caractéristiques (ex: `mean`, `std`, `skew` sur les 3197 points).
* **Résultat (v2)** : **Échec total.** Le score PR-AUC était équivalent au hasard. Les caractéristiques statistiques simples ne sont pas suffisantes.

### Partie B : CNN 1D "End-to-End"

* **Approche** : Fournir les 3197 points de données brutes directement au CNN.
* **Résultat (v3-v9)** : **Échec total.** Le signal de transit est trop faible et noyé dans le bruit. Le CNN n'a pas pu converger (Overfitting / Underfitting).

### Partie C : CNN 1D sur "Pliage de Phase" (Lomb-Scargle)

* **Approche** : Utiliser un périodogramme `LombScargle` pour trouver la "meilleure" période, puis "plier" la courbe de lumière sur cette période pour amplifier le signal.
* **Résultat (v10-v16)** : **Échec "Garbage In, Garbage Out".** Un test de vérification (`sanity check`) a prouvé que notre fonction `find_best_period` était défectueuse et produisait des données inutilisables (du bruit aléatoire ou des lignes plates).

### Partie D : CNN 1D sur Périodogramme Complet

* **Approche** : Au lieu de plier, nous avons donné le graphique complet du périodogramme `LombScargle` (1000 points) au CNN pour qu'il trouve les "pics" de puissance.
* **Résultat (v17-v22)** : **Succès partiel et chaos.** C'était notre meilleure tentative de CNN. En utilisant des poids de classe (`class_weight`) pour forcer l'apprentissage, nous avons **réussi à trouver 6 des 7 planètes** du set de test.
* **Problème** : L'entraînement était **totalement instable**. Le modèle "paniquait" et générait des centaines de Faux Positifs (>360) pour trouver ces 6 planètes, le rendant inutilisable.

### Partie E : Modèle Statistique sur Périodogramme (BLS)

* **Approche** : Abandon du CNN. Nous avons utilisé l'algorithme standard de la NASA, **`BoxLeastSquares` (BLS)**, conçu spécifiquement pour trouver des transits. Nous avons extrait 4 caractéristiques clés (ex: `peak_snr`, `depth`) et les avons données à un Random Forest.
* **Résultat (v23-v28)** : **Échec final et définitif.** L'algorithme BLS a tourné pendant 2 heures sur les 5087 étoiles et a conclu qu'il n'y avait **aucun signal de transit** à trouver. Les 37 "planètes" étaient statistiquement indiscernables du bruit.

## 4. Les Données ✅

* **Source** : [Exoplanet Hunting in TESS Light Curves (Kaggle)](https://www.kaggle.com/datasets/keplersmachines/kepler-labelled-time-series-data)
* **Format** : `train.csv` et `test.csv`.
* **Note** : Les `.csv` sont dans le `.gitignore`. Pour exécuter le projet, téléchargez-les et placez-les dans `data/raw/`.

## 5. Installation

Pour configurer votre environnement local et exécuter ce projet :

1.  **Clonez le dépôt :**
    ```bash
    git clone [https://github.com/K4yan0/exoplanet-detection-ml] (https://github.com/K4yan0/exoplanet-detection-ml.git)
    cd exoplanet-detection-ml
    ```

2.  **Créez un environnement virtuel :**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Sur Windows: .\venv\Scripts\activate
    ```

3.  **Installez les dépendances :**
    ```bash
    pip install -r requirements.txt
    ```

## 6. Structure et Utilisation

Les 14 notebooks de ce projet racontent l'histoire de cette investigation, numérotés par ordre d'expérimentation.

* `01_EDA_...` / `02_RF_...` : Baselines (Échec)
* `03_CNN_...` à `09_CNN_...` : Premiers tests CNN et augmentation (Échec)
* `10_CNN_...` / `11_CNN_...` : Hypothèse du "Pliage de Phase" (Échec)
* `12_CNN_...` : Hypothèse du "Périodogramme" (Succès partiel mais instable)
* `13_Periodogram_Stats_...` : Pivot vers un modèle statistique (Échec)
* `14_BLS_Stats_...` : Test final avec l'algorithme BLS (Échec définitif)

## 7. Conclusion & Résultats Finaux

Ce projet a réussi, non pas à construire un modèle, mais à **prouver de manière concluante que le jeu de données est inutilisable** pour cette tâche.

Après 28 versions et pivots méthodologiques, nous avons démontré qu'il n'existe **aucune corrélation** entre le `LABEL` (Planète) et un quelconque signal de transit détectable dans les données `FLUX`.

Notre test final (v28), utilisant l'algorithme standard de la NASA (`BoxLeastSquares`), a tourné pendant plus de 2 heures et a confirmé que **les 37 courbes de lumière labellées "Planète" sont statistiquement indiscernables du bruit de fond.**

### Leçons Apprises

| Approche | Modèle | Résultat | Leçon Apprise |
| :--- | :--- | :--- | :--- |
| **Données Brutes** | CNN 1D | **Échec** | Le signal est trop faible pour une approche "end-to-end" avec si peu de données. |
| **Pliage de Phase** | CNN 1D | **Échec** | "Garbage In, Garbage Out". Un mauvais pipeline de features (v10) crée des données inutilisables. |
| **Périodogramme** | CNN 1D | **Succès Partiel** | Le signal *est* là ! Nous avons trouvé **6/7 planètes**, mais le CNN est l'outil inadapté : l'entraînement est trop instable et génère >360 Faux Positifs. |
| **Stats (Lomb-Scargle)** | Random Forest | **Échec** | `LombScargle` est le mauvais algorithme (il cherche des sinusoïdes, pas des transits). |
| **Stats (BLS)** | Random Forest | **Échec Définitif** | L'algorithme standard de l'industrie (BLS) a prouvé qu'**il n'y a aucun signal de transit** à trouver. Le jeu de données est défectueux. |

## 8. Licence

Ce projet est publié sous la Licence MIT. Voir le fichier `LICENSE` pour plus de détails.