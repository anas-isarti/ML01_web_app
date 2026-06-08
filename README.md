# UFO Sighting Country Predictor

Application de machine learning qui prédit dans quel pays une observation d'OVNI a eu lieu, en se basant sur la localisation géographique, la durée et l'heure.

---

## Architecture

Le projet propose deux modes de fonctionnement :

```
Mode A — Application standalone (recommandé pour démarrer vite)
    ufo_app.py          Streamlit — charge le modèle directement

Mode B — Architecture backend / frontend séparés
    ufo_backend.py      FastAPI  — expose /predict sur le port 8001
    ufo_frontend.py     Streamlit — appelle le backend via HTTP
```

---

## Fichiers

| Fichier | Rôle |
|---|---|
| `ufo_train.py` | Génère les données, entraîne 3 modèles, sauvegarde le meilleur |
| `ufo_app.py` | App Streamlit standalone (charge le .pkl directement) |
| `ufo_backend.py` | API FastAPI — endpoint `POST /predict` |
| `ufo_frontend.py` | Frontend Streamlit qui appelle le backend |
| `ufo_model.pkl` | Modèle entraîné (GradientBoostingClassifier) |
| `ufo_label_encoder.pkl` | LabelEncoder sklearn pour les 7 pays |
| `ufo_training_report.txt` | Résumé des performances des 3 modèles |
| `requirements.txt` | Dépendances Python |

---

## Données d'entraînement

Les données sont **entièrement synthétiques** (aucun CSV externe), générées avec `numpy` (`seed=42`, `n_samples=20000`).

### Features (6 colonnes, noms exacts)

| Feature | Type | Description |
|---|---|---|
| `latitude` | float | Latitude [-90, 90] |
| `longitude` | float | Longitude [-180, 180] |
| `duration_seconds` | int | Durée de l'observation [1, 7200] |
| `hour` | int | Heure de la journée [0, 23] |
| `month` | int | Mois [1, 12] |
| `day_of_week` | int | Jour de la semaine [0=lun, 6=dim] |

### Cible : `country`

| Pays | Code | Proportion |
|---|---|---|
| United States | `us` | 40% |
| Canada | `ca` | 13% |
| United Kingdom | `gb` | 12% |
| France | `fr` | 9% |
| Germany | `de` | 8% |
| Australia | `au` | 8% |
| Other | `other` | 10% |

### Génération géographique

Les coordonnées sont tirées autour de centres géographiques réalistes par pays, avec bruit gaussien (écart-type latitude ~10, longitude ~13) :

| Pays | Centre lat | Centre lon |
|---|---|---|
| us | 39.8 | -98.6 |
| ca | 56.1 | -106.3 |
| gb | 55.4 | -3.4 |
| au | -25.3 | 133.8 |
| de | 51.2 | 10.4 |
| fr | 46.6 | 2.2 |
| other | 20.0 | 0.0 |

---

## Entraînement des modèles

### Lancer l'entraînement

```
python ufo_train.py
```

Génère `ufo_model.pkl`, `ufo_label_encoder.pkl` et `ufo_training_report.txt`.

### Pipeline

- Split train/test : 80/20, `random_state=42`, `stratify=y`
- Sélection automatique du meilleur modèle (accuracy test, ex-aequo : cv mean)
- Cross-validation 5-fold sur le train set pour chaque modèle

### Résultats

| Modèle | Accuracy test | CV mean (5-fold) |
|---|---|---|
| **GradientBoostingClassifier** | **0.7648** | **0.7709** |
| RandomForestClassifier | 0.7560 | 0.7664 |
| LogisticRegression | 0.7508 | 0.7527 |

Modèle gagnant : **GradientBoostingClassifier**

### Importance des features

| Feature | Importance |
|---|---|
| `longitude` | 69.5% |
| `latitude` | 29.8% |
| `duration_seconds` | 0.4% |
| `month` | 0.1% |
| `hour` | 0.1% |
| `day_of_week` | 0.07% |

La géographie (lat/lon) porte presque tout le signal prédictif — attendu puisque les données sont générées avec des centres géographiques distincts par pays.

---

## Installation

### Dépendances (requirements.txt)

```
streamlit
altair==5.4.1
streamlit-drawable-canvas
opencv-python-headless
numpy
scikit-learn
joblib
pandas
```

Installer :

```
python -m pip install -r requirements.txt
```

### Dépendances supplémentaires pour le mode backend/frontend

`requirements.txt` ne contient pas les libs nécessaires à `ufo_backend.py` et `ufo_frontend.py`. Les installer manuellement si besoin :

```
python -m pip install fastapi uvicorn pydantic requests
```

---

## Lancer l'application

### Mode A — Standalone (recommandé)

```
python -m streamlit run ufo_app.py
```

Ouvre automatiquement : **http://localhost:8501**

> Sur Windows, utiliser `python -m streamlit` plutôt que `streamlit` seul — l'exécutable n'est souvent pas dans le PATH.

### Mode B — Backend + Frontend séparés

**Terminal 1 — démarrer le backend FastAPI :**

```
python ufo_backend.py
```

API disponible sur : **http://localhost:8001**  
Documentation interactive Swagger : **http://localhost:8001/docs**

**Terminal 2 — démarrer le frontend Streamlit :**

```
python -m streamlit run ufo_frontend.py
```

Frontend disponible sur : **http://localhost:8501**

---

## Utilisation de l'interface

Dans la barre latérale, renseignez :

- **Latitude / Longitude** : coordonnées géographiques de l'observation
- **Duration (seconds)** : durée de l'observation (1 à 7200 s)
- **Hour of day** : heure (0 à 23)
- **Month** : mois de l'année
- **Day of week** : jour de la semaine (0=lundi, 6=dimanche)

Cliquez sur **Predict Country** pour obtenir le pays prédit et les probabilités pour chacun des 7 pays.

---

## Environnement testé

- Python 3.12.10
- scikit-learn 1.8.0
- joblib 1.5.3
- numpy 2.4.4
- pandas 3.0.2
