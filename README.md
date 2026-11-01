# 🧬 Surveillance de la Résistance aux Antimicrobiens (AMR) — Déduplication GLASS, Analyse de Tendance Nationale & Forecasting

Ce projet fournit une suite Python complète et reproductible dédiée à la surveillance épidémiologique de la résistance aux antimicrobiens (AMR) selon les standards **OMS GLASS** (période 2019–2025) :

1. **Déduplication épidémiologique des isolats** (`amr_deduplication.py` & `deduplication_glass_episodes.ipynb`) :
   - Déduplication annuelle **WHO GLASS** (`df_glass`) avec gestion explicite de la priorité **LCR > Sang** le même jour.
   - Déduplication par épisodes cliniques continus de **14 jours** (`df_14j`), **30 jours** (`df_30j`) et **365 jours** (`df_365j`) via le package `AMR`.
2. **Analyse descriptive des résistances (`%R`)** (`dashboard_data.py` & `analyse_resistance_forecasting.ipynb`) :
   - Calcul des effectifs interprétables $N = S + I + R$ et $\%R = 100 \times R / N$.
   - Matrices / cartes de chaleur **Bactérie × Antibiotique**, évolutions annuelles et comparaisons par hôpital et par service.
3. **Analyse statistique de la tendance nationale mensuelle** (`trend_analysis.py` & `analyse_resistance_forecasting.ipynb`) :
   - Régression binomiale groupée ($R_t$ succès sur $N_t$ essais) avec temps continu en années, diagnostics de **surdispersion** ($\hat{\phi}$) et d'**autocorrélation temporelle** (Ljung–Box / lag-1), inférence robuste adaptative (**Quasi-Binomial** / **HAC Newey–West**) et correction des tests multiples par **Benjamini–Hochberg (FDR)**.
4. **Prévision temporelle (Forecasting mensuel et trimestriel)** (`forecasting.py` & `quarterly_forecasting.py`) :
   - Modèles *Naïve*, *Naïve saisonnière*, *Holt*, *ETS / Holt-Winters*, *ARIMA*, *SARIMA* et *GLM binomial*, avec backtesting chronologique multi-horizons et test final (*Holdout*).

---

## 📁 Structure du Projet

### Notebooks principaux
- `deduplication_glass_episodes.ipynb` : **Notebook de déduplication OMS GLASS & par épisodes (14 j, 30 j, 365 j)** — audit complet, tableaux comparatifs **D1–D3**, figures **F1–F3**, tests de validation sur cas synthétiques et données réelles, et export des bases dédupliquées.
- `analyse_resistance_forecasting.ipynb` : **Notebook principal d'analyse AMR, de tendance statistique nationale et de forecasting** — accepte directement en entrée n'importe laquelle des bases dédupliquées (`"df_glass"`, `"df_14j"`, `"df_30j"`, `"df_365j"`, `"raw"`) ou un fichier externe (`.parquet`, `.csv`, `.tsv`, `.xlsx`).

### Modules Python
- `amr_deduplication.py` : Moteur de déduplication épidémiologique (`dedup_glass`, `dedup_episode`, `run_all_deduplications`, `validate_deduplication`, `summarize_deduplication`, `compare_methods`, `generate_tables_and_figures`).
- `dashboard_data.py` : Chargement et standardisation des données (`load_source_data` compatible formats long et large `.parquet` / `.csv` / `.xlsx`), filtrage multi-critères, calcul des cubes `S/I/R` et ré-export des fonctions d'analyse de tendance et de prévision.
- `trend_analysis.py` : Analyse statistique de la tendance nationale mensuelle par couple Bactérie × Antibiotique (`build_national_monthly_sir_cube`, `get_national_monthly_sir_series`, `estimate_national_monthly_trend`, `analyze_all_national_trends`).
- `forecasting.py` : Modèles de prévision mensuelle (*Naïve*, *Naïve saisonnière*, *Holt*, *Holt-Winters*, *ARIMA*, *SARIMA*, *GLM binomial*) et comparateur chronologique sur les 5 méthodes communes.
- `quarterly_forecasting.py` : Agrégation trimestrielle nationale ($p_R = R / (S + I + R)$) et prévision trimestrielle avec validation glissante (*Expanding Window*) et évaluation *Holdout*.
- `time_series_prediction.py` : Script en ligne de commande (CLI) pour exécuter rapidement une prévision mensuelle sur un couple Bactérie × Antibiotique.

### Tests unitaires & Dépendances
- `test_trend_analysis.py` : Tests unitaires de l'analyse statistique de tendance nationale (tendance croissante, absence de tendance, données insuffisantes / quasi-séparation, non-imputation à `0 %` des mois absents, inférence robuste face à la surdispersion/autocorrélation, correction FDR).
- `test_forecasting.py` : Tests unitaires des modèles de prévision mensuelle et de la sélection AICc.
- `test_quarterly_forecasting.py` : Tests unitaires de l'agrégation et du backtesting trimestriels.
- `requirements.txt` : Dépendances Python du projet (`pandas`, `numpy`, `scipy`, `statsmodels`, `matplotlib`, `seaborn`, `AMR`, `pyarrow`, `openpyxl`).

### Répertoires de sortie (générés par la déduplication)
- `outputs_deduplication/` :
  - Bases dédupliquées en `.parquet` et `.csv` : `df_amr`, `df_glass`, `df_14j`, `df_30j`, `df_365j`.
  - Tableaux récapitulatifs : `D1_deduplication_glass_annuelle.csv`, `D2_comparaison_methodes_annuelle.csv`, `D3_comparaison_espece_prelevement.csv`.
- `figures_deduplication/` :
  - Graphiques haute résolution (`.png` 300 dpi et `.pdf`) : `F1_isolats_retenus_par_annee_et_methode`, `F2_pourcentage_exclus_par_annee_et_methode`, `F3_top10_especes_comparaison_methodes`.

---

## 🚀 Installation et Démarrage Rapide

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Étape 1 — Générer et auditer les bases dédupliquées (optionnel si déjà présentes dans `outputs_deduplication/`)
```bash
jupyter notebook deduplication_glass_episodes.ipynb
# ou en ligne de commande :
python amr_deduplication.py
```

### Étape 2 — Lancer le notebook d'analyse des résistances, de tendance nationale et de forecasting
```bash
jupyter notebook analyse_resistance_forecasting.ipynb
```

Au début du notebook `analyse_resistance_forecasting.ipynb` (Section 1), configurez la variable `DATASET_CHOICE` pour choisir la base à analyser :
- `"df_glass"` *(recommandé OMS GLASS)* : déduplication annuelle WHO GLASS avec priorité LCR > Sang.
- `"df_14j"` : déduplication par épisode de 14 jours.
- `"df_30j"` : déduplication par épisode de 30 jours.
- `"df_365j"` : déduplication par épisode de 365 jours.
- `"raw"` : base complète avant déduplication (`data.parquet`).
- Tout chemin vers un fichier `.parquet`, `.csv`, `.tsv` ou `.xlsx` (au format long `antibiotique`/`sir` ou au format large avec une colonne par antibiotique) ou directement un `pandas.DataFrame`.

---

## 📈 Méthodologie de l'Analyse Statistique de Tendance Nationale (`trend_analysis.py`)

Intégrée à la **Section 2.4** de `analyse_resistance_forecasting.ipynb`, l'analyse statistique de la tendance nationale respecte les principes biostatistiques suivants :

1. **Périmètre national et définition de la résistance** :
   - Agrégation nationale de tous les hôpitaux (sans stratification ni ajustement sur l'hôpital) pour chaque couple `bactérie × antibiotique`.
   - Conservation stricte de $N_t = S_t + I_t + R_t$ et $\%R_t = 100 \times R_t / N_t$.
   - **Aucune période sans résultat ($N_t = 0$) n'est interprétée ni imputée comme un taux de résistance nul (`0 %`)**.
2. **Régression binomiale groupée sur les effectifs mensuels** :
   - Modélisation de $(R_t, N_t - R_t)$ par un GLM Binomial à lien logit :
     $$R_t \sim \text{Binomial}(N_t, p_t), \qquad \text{logit}(p_t) = \beta_0 + \beta_1 \cdot t_{\text{années}}$$
     où $t_{\text{années}}$ est le temps continu exprimé en années exactes depuis le premier mois observé. Chaque mois pèse selon son effectif réel $N_t$.
3. **Diagnostics et inférence robuste adaptative** :
   - Vérification systématique de la **surdispersion** ($\hat{\phi} = \chi^2_{\text{Pearson}} / \text{ddl}_{\text{résid}}$) et de l'**autocorrélation temporelle** des résidus de Pearson (autocorrélation lag-1 sur mois consécutifs et test de **Ljung–Box**).
   - Sélection automatique de l'inférence valide :
     - **GLM Binomial + erreurs-types robustes HAC (Newey–West)** en présence d'autocorrélation temporelle ;
     - **GLM Quasi-Binomial** (`scale="X2"`, loi $t$ de Student) en présence de surdispersion sans autocorrélation ;
     - **GLM Binomial (MLE standard)** lorsque ni surdispersion ni autocorrélation ne sont détectées.
4. **Effets rapportés et correction des tests multiples** :
   - Double échelle d'interprétation : **effet marginal annuel moyen en points de pourcentage par an (`pts % / an`)** avec IC à 95 % par méthode Delta, et **échelle du modèle logit** (`log-odds / an` et **Odds Ratio annuel** $\text{OR}_{\text{an}} = \exp(\beta_1)$ avec IC à 95 %).
   - Correction des p-valeurs par **Benjamini–Hochberg (FDR)** sur la famille explicitement définie des couples calculables ($\ge 6$ mois observés, $N_{\text{total}} \ge 30$, hors résistance constante à 0 %/100 % ou quasi-séparation).

---

## 🔮 Utilisation en Ligne de Commande (`time_series_prediction.py`)

```bash
python time_series_prediction.py "<Nom_Bactérie>" "<Nom_Antibiotique>" --method holt --horizon 12
```

### Exemples :
```bash
python time_series_prediction.py "Klebsiella pneumoniae" "cefotaxime" --method arima
python time_series_prediction.py "Escherichia coli" "ciprofloxacine" --method sarima
python time_series_prediction.py "Pseudomonas aeruginosa" "ceftazidime" --method glm
python time_series_prediction.py "Acinetobacter baumannii" "imipeneme" --method holt_winters
python time_series_prediction.py "Staphylococcus aureus" "cefoxitine" --method naive
```

---

## 🧪 Exécuter les Tests Unitaires

```bash
python -m unittest test_trend_analysis.py test_forecasting.py test_quarterly_forecasting.py -v
```
