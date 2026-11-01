"""Analyse statistique de la tendance nationale de la résistance aux antibiotiques (%R).

Ce module implémente l'estimation des tendances nationales mensuelles par couple
Bactérie × Antibiotique à partir des effectifs interprétables S, I, R (N = S + I + R,
%R = 100 * R / N) :
- Agrégation nationale mensuelle sans stratification hospitalière et sans imputer
  les mois sans observation à 0 %.
- Régression binomiale groupée (GLM Binomial à lien logit) pondérée par les effectifs
  mensuels N, avec le temps exprimé en années continues.
- Diagnostics automatiques de surdispersion (ratio de Pearson φ) et d'autocorrélation
  résiduelle (Ljung–Box et autocorrélation d'ordre 1 sur mois adjacents).
- Inférence adaptative : erreurs-types robustes HAC (Newey–West) en présence
  d'autocorrélation, Quasi-Binomiale (échelle de Pearson) en présence de surdispersion
  sans autocorrélation, ou MLE binomial standard lorsque les hypothèses sont respectées.
- Expression de l'effet sur l'échelle du modèle (log-odds/an et Odds Ratio annuel)
  et en points de pourcentage par an (effet marginal annuel moyen par méthode Delta).
- Correction de Benjamini–Hochberg (FDR) sur la famille des couples analysables.
"""

from __future__ import annotations

import math
import warnings
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.multitest import multipletests

DATE_COLUMN = "specimen_collection_date"
ORGANISM_COLUMN = "organism_name"
ANTIBIOTIC_COLUMN = "antibiotic"
SIR_COLUMN = "interpretation"
SIR_CODES = ("S", "I", "R")

# Seuils explicites alignés sur les conventions existantes du projet :
# - MIN_TREND_MONTHS = 6 : aligné sur le minimum de 6 mois de `run_monthly_forecast`
#   et `monthly_binomial_glm_forecasting(..., seasonal=False)` dans `dashboard_data.py` / `forecasting.py`.
# - DEFAULT_LOW_COUNT_THRESHOLD = 30 : aligné sur `LOW_COUNT_THRESHOLD = 30` dans
#   `compute_resistance_summary` et `aggregate_quarterly_counts`.
MIN_TREND_MONTHS = 6
DEFAULT_LOW_COUNT_THRESHOLD = 30
OVERDISPERSION_RATIO_THRESHOLD = 1.5
ALPHA_DIAGNOSTIC = 0.05


def build_national_monthly_sir_cube(
    frame: pd.DataFrame,
    *,
    date_column: str = DATE_COLUMN,
    organism_column: str = ORGANISM_COLUMN,
    antibiotic_column: str = ANTIBIOTIC_COLUMN,
    sir_column: str = SIR_COLUMN,
) -> pd.DataFrame:
    """Agrège au niveau national les effectifs mensuels S, I, R, N = S+I+R et %R par couple Bactérie × Antibiotique.

    Les mois sans observation ne sont jamais créés avec N=0 ou %R=0 dans ce cube ;
    seuls les mois comportant au moins un test interprétable (S, I ou R) et une date
    valide y figurent.
    """
    if frame.empty:
        return pd.DataFrame(
            columns=["bacterie", "antibiotique", "mois", "S", "I", "R", "N", "%R"]
        )

    df = frame
    if any(str(c) != str(c).strip().lower() for c in df.columns):
        df = df.copy()
        df.columns = [str(c).strip().lower() for c in df.columns]

    date_col = date_column.strip().lower()
    org_col = organism_column.strip().lower()
    abx_col = antibiotic_column.strip().lower()
    sir_col = sir_column.strip().lower()

    required = {date_col, org_col, abx_col, sir_col}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Colonnes requises absentes : {', '.join(missing)}")

    org_vals = df[org_col].astype("string").str.strip().fillna("")
    abx_vals = df[abx_col].astype("string").str.strip().fillna("")
    sir_vals = df[sir_col].astype("string").str.strip().str.upper().fillna("")
    parsed_dates = pd.to_datetime(df[date_col], errors="coerce")

    valid_mask = (
        (org_vals != "")
        & (abx_vals != "")
        & sir_vals.isin(SIR_CODES)
        & parsed_dates.notna()
    )
    if not valid_mask.any():
        return pd.DataFrame(
            columns=["bacterie", "antibiotique", "mois", "S", "I", "R", "N", "%R"]
        )

    usable = pd.DataFrame(
        {
            "bacterie": org_vals.loc[valid_mask].astype(str),
            "antibiotique": abx_vals.loc[valid_mask].astype(str),
            "mois": parsed_dates.loc[valid_mask].dt.strftime("%Y-%m"),
            "_sir": sir_vals.loc[valid_mask].astype(str),
        }
    )
    monthly = (
        usable.groupby(["bacterie", "antibiotique", "mois", "_sir"], dropna=False)
        .size()
        .unstack("_sir", fill_value=0)
        .reset_index()
    )
    for code in SIR_CODES:
        if code not in monthly.columns:
            monthly[code] = 0
        monthly[code] = monthly[code].astype(int)

    monthly["N"] = monthly["S"] + monthly["I"] + monthly["R"]
    monthly["%R"] = (monthly["R"] / monthly["N"] * 100.0).round(2)

    return (
        monthly[["bacterie", "antibiotique", "mois", "S", "I", "R", "N", "%R"]]
        .sort_values(["bacterie", "antibiotique", "mois"])
        .reset_index(drop=True)
    )


def get_national_monthly_sir_series(
    monthly_sir_cube: pd.DataFrame,
    bacterie: str | None = None,
    antibiotique: str | None = None,
    *,
    include_empty_months_as_nan: bool = False,
) -> pd.DataFrame:
    """Extrait la série mensuelle nationale (S, I, R, N, %R) pour un couple Bactérie × Antibiotique.

    Si `include_empty_months_as_nan=True`, réindexe sur la grille mensuelle continue
    du premier au dernier mois observé en laissant N=0 et %R=NaN pour les mois sans
    résultat (utile pour visualiser les ruptures sans jamais imputer 0 %).
    """
    sub = monthly_sir_cube.copy()
    if bacterie and bacterie != "__ALL_BACTERIA__":
        sub = sub.loc[sub["bacterie"].astype(str).str.casefold() == bacterie.strip().casefold()]
    if antibiotique and antibiotique != "__ALL_ANTIBIOTICS__":
        sub = sub.loc[sub["antibiotique"].astype(str).str.casefold() == antibiotique.strip().casefold()]

    if sub.empty:
        return pd.DataFrame(columns=["mois", "S", "I", "R", "N", "%R"])

    grouped = (
        sub.groupby("mois", as_index=False)[["S", "I", "R"]]
        .sum()
        .sort_values("mois")
        .reset_index(drop=True)
    )
    grouped["N"] = grouped["S"] + grouped["I"] + grouped["R"]
    grouped = grouped.loc[grouped["N"] > 0].reset_index(drop=True)
    grouped["%R"] = (grouped["R"] / grouped["N"] * 100.0).round(2)

    if include_empty_months_as_nan and len(grouped) >= 2:
        periods = pd.period_range(
            grouped["mois"].iloc[0], grouped["mois"].iloc[-1], freq="M"
        ).astype(str)
        full_grid = pd.DataFrame({"mois": periods})
        merged = full_grid.merge(grouped, on="mois", how="left")
        for col in ("S", "I", "R", "N"):
            merged[col] = merged[col].fillna(0).astype(int)
        # Les mois où N == 0 conservent explicitement %R = NaN (jamais 0 %)
        merged["%R"] = np.where(
            merged["N"] > 0,
            (merged["R"] / merged["N"] * 100.0).round(2),
            np.nan,
        )
        return merged

    return grouped


def _month_str_to_ordinal(month_str: str) -> int:
    year, month = map(int, str(month_str).split("-")[:2])
    return year * 12 + (month - 1)


def _compute_residual_diagnostics(
    pearson_residuals: np.ndarray,
    ordinals: np.ndarray,
    df_resid: float,
) -> dict[str, Any]:
    """Calcule les diagnostics de surdispersion et d'autocorrélation temporelle."""
    n_obs = len(pearson_residuals)
    pearson_chi2 = float(np.sum(pearson_residuals**2))
    dispersion_phi = float(pearson_chi2 / max(1.0, df_resid))
    dispersion_pvalue = (
        float(stats.chi2.sf(pearson_chi2, df=df_resid)) if df_resid > 0 else np.nan
    )
    overdispersed = bool(
        (dispersion_phi > OVERDISPERSION_RATIO_THRESHOLD)
        or (pd.notna(dispersion_pvalue) and dispersion_pvalue < ALPHA_DIAGNOSTIC and dispersion_phi > 1.0)
    )

    # Autocorrélation d'ordre 1 uniquement sur les paires de mois réellement consécutifs (écart = 1 mois)
    diffs = np.diff(ordinals)
    adjacent_mask = diffs == 1
    n_adjacent_pairs = int(np.sum(adjacent_mask))
    lag1_autocorr = np.nan
    lag1_pvalue = np.nan

    if n_adjacent_pairs >= 6 and np.std(pearson_residuals) > 1e-6:
        r_prev = pearson_residuals[:-1][adjacent_mask]
        r_next = pearson_residuals[1:][adjacent_mask]
        if np.std(r_prev) > 1e-6 and np.std(r_next) > 1e-6:
            lag1_autocorr, lag1_pvalue = stats.pearsonr(r_prev, r_next)
            lag1_autocorr = float(lag1_autocorr)
            lag1_pvalue = float(lag1_pvalue)

    # Test de Ljung–Box si la série est contiguë (ou >= 80 % de paires contiguës) et non dégénérée
    ljung_box_stat = np.nan
    ljung_box_pvalue = np.nan
    ljung_box_lag = None
    contiguous_ratio = float(n_adjacent_pairs / max(1, n_obs - 1)) if n_obs > 1 else 0.0

    if n_obs >= 8 and contiguous_ratio >= 0.80 and np.std(pearson_residuals) > 1e-6:
        lag = max(1, min(6, n_obs // 3))
        try:
            lb_df = acorr_ljungbox(pearson_residuals, lags=[lag], return_df=True)
            ljung_box_lag = int(lag)
            ljung_box_stat = float(lb_df["lb_stat"].iloc[0])
            ljung_box_pvalue = float(lb_df["lb_pvalue"].iloc[0])
        except Exception:
            pass

    autocorrelated = bool(
        (pd.notna(ljung_box_pvalue) and ljung_box_pvalue < ALPHA_DIAGNOSTIC)
        or (pd.notna(lag1_pvalue) and lag1_pvalue < ALPHA_DIAGNOSTIC)
    )

    return {
        "pearson_chi2": pearson_chi2,
        "df_resid": float(df_resid),
        "dispersion_phi": dispersion_phi,
        "dispersion_pvalue": dispersion_pvalue,
        "overdispersed": overdispersed,
        "n_adjacent_pairs": n_adjacent_pairs,
        "contiguous_ratio": contiguous_ratio,
        "lag1_autocorr": lag1_autocorr,
        "lag1_pvalue": lag1_pvalue,
        "ljung_box_lag": ljung_box_lag,
        "ljung_box_stat": ljung_box_stat,
        "ljung_box_pvalue": ljung_box_pvalue,
        "autocorrelated": autocorrelated,
    }


def estimate_national_monthly_trend(
    monthly_series: pd.DataFrame,
    *,
    min_months: int = MIN_TREND_MONTHS,
    min_total_n: int = DEFAULT_LOW_COUNT_THRESHOLD,
) -> dict[str, Any]:
    """Estime la tendance temporelle nationale d'un couple Bactérie × Antibiotique par régression binomiale.

    Modèle :
        logit(p_t) = β_0 + β_1 * t_années
    où R_t ~ Binomial(N_t, p_t) avec N_t = S_t + I_t + R_t > 0 et t_années est le temps
    continu en années depuis le premier mois observé.

    Inférence adaptative selon les diagnostics :
    - Si autocorrélation temporelle détectée (ou surdispersion + autocorrélation) :
      erreurs-types robustes sandwich HAC (Newey–West).
    - Si surdispersion détectée sans autocorrélation :
      modèle Quasi-Binomial (variance multipliée par φ = χ²_Pearson / ddl).
    - Sinon :
      GLM Binomial standard (maximum de vraisemblance).
    """
    if monthly_series is None or monthly_series.empty:
        return {
            "status": "Non calculé : aucune observation interprétable (S/I/R)",
            "calculable": False,
            "n_months": 0,
            "total_R": 0,
            "total_N": 0,
            "pooled_pct_R": np.nan,
        }

    df = monthly_series.copy()
    if "N" not in df.columns:
        df["N"] = df["S"] + df["I"] + df["R"]

    # Exclusion stricte des mois sans observation (N == 0 ou NaN) : jamais imputés à 0 %
    valid = (
        df["N"].notna()
        & df["R"].notna()
        & (df["N"] > 0)
        & (df["R"] >= 0)
        & (df["R"] <= df["N"])
    )
    obs = df.loc[valid].sort_values("mois").reset_index(drop=True)

    n_months = int(len(obs))
    total_r = int(obs["R"].sum()) if n_months > 0 else 0
    total_n = int(obs["N"].sum()) if n_months > 0 else 0
    pooled_pct_r = float(100.0 * total_r / total_n) if total_n > 0 else np.nan
    first_month = str(obs["mois"].iloc[0]) if n_months > 0 else None
    last_month = str(obs["mois"].iloc[-1]) if n_months > 0 else None

    base_info = {
        "n_months": n_months,
        "first_month": first_month,
        "last_month": last_month,
        "total_R": total_r,
        "total_N": total_n,
        "pooled_pct_R": round(pooled_pct_r, 2) if pd.notna(pooled_pct_r) else np.nan,
    }

    if n_months < min_months:
        return {
            **base_info,
            "status": f"Non calculé : nombre de mois insuffisant ({n_months} < {min_months})",
            "calculable": False,
        }

    if total_n < min_total_n:
        return {
            **base_info,
            "status": f"Non calculé : effectif total insuffisant (N={total_n} < {min_total_n})",
            "calculable": False,
        }

    if total_r == 0:
        return {
            **base_info,
            "status": "Non calculé : résistance constante à 0 % (aucun isolat R)",
            "calculable": False,
        }

    if total_r == total_n:
        return {
            **base_info,
            "status": "Non calculé : résistance constante à 100 % (aucun isolat S/I)",
            "calculable": False,
        }

    ordinals = np.asarray([_month_str_to_ordinal(m) for m in obs["mois"]], dtype=float)
    time_years = (ordinals - ordinals[0]) / 12.0
    span_years = float(time_years[-1] - time_years[0])
    if span_years <= 0:
        return {
            **base_info,
            "status": "Non calculé : étendue temporelle nulle",
            "calculable": False,
        }

    r_arr = obs["R"].to_numpy(dtype=float)
    n_arr = obs["N"].to_numpy(dtype=float)
    s_i_arr = n_arr - r_arr
    endog_binom = np.column_stack([r_arr, s_i_arr])
    exog = sm.add_constant(time_years, has_constant="add")

    try:
        from statsmodels.tools.sm_exceptions import PerfectSeparationWarning

        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            glm_model = sm.GLM(
                endog_binom,
                exog,
                family=sm.families.Binomial(),
            )
            mle_res = glm_model.fit(maxiter=100, tol=1e-8)
        has_separation = any(
            issubclass(w.category, PerfectSeparationWarning)
            or "separation" in str(w.message).lower()
            for w in caught_warnings
        )
    except Exception as exc:
        return {
            **base_info,
            "status": f"Non calculé : échec de convergence du GLM binomial ({exc})",
            "calculable": False,
        }

    if not bool(getattr(mle_res, "converged", True)):
        return {
            **base_info,
            "status": "Non calculé : le GLM binomial n'a pas convergé",
            "calculable": False,
        }

    mle_params = np.asarray(mle_res.params, dtype=float)
    mle_bse = np.asarray(mle_res.bse, dtype=float)
    mle_fitted = np.asarray(mle_res.fittedvalues, dtype=float)
    true_separation = has_separation and (
        np.min(mle_fitted) < 1e-6 or np.max(mle_fitted) > 1.0 - 1e-6
    )
    if (
        true_separation
        or not np.all(np.isfinite(mle_params))
        or not np.all(np.isfinite(mle_bse))
        or abs(float(mle_params[1])) > 20.0
        or float(mle_bse[1]) > 20.0
    ):
        return {
            **base_info,
            "status": (
                "Non calculé : quasi-séparation complète "
                f"(événements R={total_r} ou non-R={total_n - total_r} trop rares/concentrés)"
            ),
            "calculable": False,
        }

    # Diagnostics sur le modèle MLE
    p_hat = np.clip(np.asarray(mle_res.fittedvalues, dtype=float), 1e-9, 1.0 - 1e-9)
    pearson_resid = (r_arr - n_arr * p_hat) / np.sqrt(n_arr * p_hat * (1.0 - p_hat))
    diag = _compute_residual_diagnostics(
        pearson_resid, ordinals=ordinals, df_resid=float(mle_res.df_resid)
    )

    # Sélection de la méthode d'inférence appropriée selon les diagnostics
    hac_maxlags = max(1, min(6, int(math.floor(4.0 * (n_months / 100.0) ** (2.0 / 9.0)))))
    if diag["autocorrelated"]:
        inf_res = glm_model.fit(
            maxiter=100,
            tol=1e-8,
            cov_type="HAC",
            cov_kwds={"maxlags": hac_maxlags},
        )
        inference_method = (
            f"GLM Binomial + HAC Newey–West (maxlags={hac_maxlags})"
            if not diag["overdispersed"]
            else f"GLM Binomial + HAC Newey–West (surdispersion φ={diag['dispersion_phi']:.2f} & autocorr.)"
        )
    elif diag["overdispersed"]:
        inf_res = glm_model.fit(maxiter=100, tol=1e-8, scale="X2")
        inference_method = f"GLM Quasi-Binomial (surdispersion φ={diag['dispersion_phi']:.2f})"
    else:
        inf_res = mle_res
        inference_method = "GLM Binomial (MLE standard)"

    params = np.asarray(inf_res.params, dtype=float)
    bse = np.asarray(inf_res.bse, dtype=float)
    cov_mat = np.asarray(inf_res.cov_params(), dtype=float)
    pvals = np.asarray(inf_res.pvalues, dtype=float)

    beta0, beta1 = float(params[0]), float(params[1])
    se_beta1 = float(bse[1])
    pval_beta1 = float(pvals[1])
    pval_naive_mle = float(np.asarray(mle_res.pvalues, dtype=float)[1])

    if not np.isfinite(se_beta1) or se_beta1 > 20.0 or not np.isfinite(pval_beta1):
        return {
            **base_info,
            "status": (
                "Non calculé : variance asymptotique dégénérée ou quasi-séparation "
                f"(R={total_r}, N={total_n})"
            ),
            "calculable": False,
        }

    # Utiliser la distribution t de Student à df_resid degrés de liberté lorsque Quasi-Binomial est utilisé,
    # ou la normale standard pour MLE / HAC
    crit_val = (
        float(stats.t.ppf(0.975, df=max(1, int(mle_res.df_resid))))
        if (diag["overdispersed"] and not diag["autocorrelated"])
        else 1.96
    )

    beta1_ci_low = beta1 - crit_val * se_beta1
    beta1_ci_high = beta1 + crit_val * se_beta1

    or_per_year = float(math.exp(max(-50.0, min(50.0, beta1))))
    or_ci_low = float(math.exp(max(-50.0, min(50.0, beta1_ci_low))))
    or_ci_high = float(math.exp(max(-50.0, min(50.0, beta1_ci_high))))

    # Effet marginal annuel moyen (AME) en points de pourcentage par an (pts % / an)
    # g(β) = 100 * β_1 * Σ [w_t * p_t * (1 - p_t)], où w_t = N_t / Σ N_t
    weights = n_arr / np.sum(n_arr)
    dp_dbeta0 = p_hat * (1.0 - p_hat)
    dp_dbeta1 = dp_dbeta0 * time_years
    avg_dp = float(np.sum(weights * dp_dbeta0))
    slope_pp_per_year = float(100.0 * beta1 * avg_dp)

    # Gradient de g(β) par rapport à (β_0, β_1) pour la méthode Delta :
    # ∂g/∂β_0 = 100 * β_1 * Σ w_t * p_t(1-p_t)(1 - 2p_t)
    # ∂g/∂β_1 = 100 * Σ w_t * p_t(1-p_t) + 100 * β_1 * Σ w_t * t * p_t(1-p_t)(1 - 2p_t)
    d2p_factor = dp_dbeta0 * (1.0 - 2.0 * p_hat)
    dg_dbeta0 = float(100.0 * beta1 * np.sum(weights * d2p_factor))
    dg_dbeta1 = float(100.0 * avg_dp + 100.0 * beta1 * np.sum(weights * time_years * d2p_factor))
    grad = np.array([dg_dbeta0, dg_dbeta1], dtype=float)
    var_slope_pp = float(grad @ cov_mat @ grad)
    se_slope_pp = math.sqrt(max(0.0, var_slope_pp))
    slope_pp_ci_low = slope_pp_per_year - crit_val * se_slope_pp
    slope_pp_ci_high = slope_pp_per_year + crit_val * se_slope_pp

    # Courbe de tendance ajustée et bande de confiance à 95 % (via méthode Delta sur le prédicteur linéaire η_t)
    eta_hat = exog @ params
    var_eta = np.einsum("ij,jk,ik->i", exog, cov_mat, exog)
    se_eta = np.sqrt(np.maximum(0.0, var_eta))
    eta_low = eta_hat - crit_val * se_eta
    eta_high = eta_hat + crit_val * se_eta

    fitted_pct = 100.0 / (1.0 + np.exp(-eta_hat))
    ci_low_pct = 100.0 / (1.0 + np.exp(-eta_low))
    ci_high_pct = 100.0 / (1.0 + np.exp(-eta_high))

    obs_with_fit = obs.copy()
    obs_with_fit["time_years"] = time_years
    obs_with_fit["fitted_%R"] = np.round(fitted_pct, 2)
    obs_with_fit["ci_lower_%R"] = np.round(ci_low_pct, 2)
    obs_with_fit["ci_upper_%R"] = np.round(ci_high_pct, 2)

    # Intervalles de confiance binomiaux exacts (Wilson / Clopper-Pearson) pour chaque point mensuel observé
    z95 = 1.959963984540054
    p_obs = r_arr / n_arr
    denom_w = 1.0 + z95**2 / n_arr
    center_w = (p_obs + z95**2 / (2.0 * n_arr)) / denom_w
    half_w = (z95 * np.sqrt((p_obs * (1.0 - p_obs) + z95**2 / (4.0 * n_arr)) / n_arr)) / denom_w
    obs_with_fit["obs_ci_lower_%R"] = np.round(np.clip(center_w - half_w, 0.0, 1.0) * 100.0, 2)
    obs_with_fit["obs_ci_upper_%R"] = np.round(np.clip(center_w + half_w, 0.0, 1.0) * 100.0, 2)

    return {
        **base_info,
        "status": "Calculé",
        "calculable": True,
        "span_years": round(span_years, 2),
        "inference_method": inference_method,
        "beta0_logit": beta0,
        "beta1_logit_per_year": beta1,
        "se_beta1_logit": se_beta1,
        "beta1_ci_lower": beta1_ci_low,
        "beta1_ci_upper": beta1_ci_high,
        "or_per_year": or_per_year,
        "or_ci_lower": or_ci_low,
        "or_ci_upper": or_ci_high,
        "slope_pp_per_year": slope_pp_per_year,
        "se_slope_pp_per_year": se_slope_pp,
        "slope_pp_ci_lower": slope_pp_ci_low,
        "slope_pp_ci_upper": slope_pp_ci_high,
        "fitted_start_pct_R": float(fitted_pct[0]),
        "fitted_end_pct_R": float(fitted_pct[-1]),
        "p_value": pval_beta1,
        "p_value_naive_mle": pval_naive_mle,
        "diagnostics": diag,
        "series_with_fit": obs_with_fit,
    }


def analyze_all_national_trends(
    frame: pd.DataFrame,
    *,
    min_months: int = MIN_TREND_MONTHS,
    min_total_n: int = DEFAULT_LOW_COUNT_THRESHOLD,
    fdr_alpha: float = 0.05,
    format_bacterium_fn: Any = None,
    format_antibiotic_fn: Any = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Analyse la tendance nationale pour chaque couple Bactérie × Antibiotique et applique Benjamini–Hochberg (FDR).

    La famille de tests pour la correction de Benjamini–Hochberg (FDR) est constituée
    exclusivement des couples Bactérie × Antibiotique présents dans le périmètre actif
    dont le modèle de tendance est effectivement calculable (`Statut == 'Calculé'`).
    Les couples non calculables ne sont jamais inclus dans la correction FDR.
    """
    monthly_cube = build_national_monthly_sir_cube(frame)
    if monthly_cube.empty:
        return pd.DataFrame(), {"n_pairs_total": 0, "n_pairs_tested_fdr": 0, "fdr_alpha": fdr_alpha}

    rows: list[dict[str, Any]] = []
    for (bact, abx), grp in monthly_cube.groupby(["bacterie", "antibiotique"], sort=True):
        res = estimate_national_monthly_trend(
            grp,
            min_months=min_months,
            min_total_n=min_total_n,
        )
        bact_label = format_bacterium_fn(bact) if callable(format_bacterium_fn) else str(bact)
        abx_label = format_antibiotic_fn(abx) if callable(format_antibiotic_fn) else str(abx)
        diag = res.get("diagnostics", {})

        rows.append(
            {
                "bacterie_raw": bact,
                "antibiotique_raw": abx,
                "Bactérie": bact_label,
                "Antibiotique": abx_label,
                "Mois observés": res.get("n_months", 0),
                "Période": (
                    f"{res['first_month']} → {res['last_month']}"
                    if res.get("first_month") and res.get("last_month")
                    else "—"
                ),
                "Résistants (R)": res.get("total_R", 0),
                "Testés (N = S+I+R)": res.get("total_N", 0),
                "Total R/N": f"{res.get('total_R', 0):,} / {res.get('total_N', 0):,}",
                "%R global": res.get("pooled_pct_R", np.nan),
                "Pente (%R pts/an)": (
                    round(res["slope_pp_per_year"], 2) if res.get("calculable") else np.nan
                ),
                "IC 95 % pente (pts/an)": (
                    f"[{res['slope_pp_ci_lower']:+.2f} ; {res['slope_pp_ci_upper']:+.2f}]"
                    if res.get("calculable")
                    else "—"
                ),
                "Pente logit (log-odds/an)": (
                    round(res["beta1_logit_per_year"], 4) if res.get("calculable") else np.nan
                ),
                "OR annuel": (
                    round(res["or_per_year"], 3) if res.get("calculable") else np.nan
                ),
                "IC 95 % OR annuel": (
                    f"[{res['or_ci_lower']:.3f} ; {res['or_ci_upper']:.3f}]"
                    if res.get("calculable")
                    else "—"
                ),
                "p-valeur": res.get("p_value", np.nan) if res.get("calculable") else np.nan,
                "p-valeur FDR (BH)": np.nan,
                "Direction / Interprétation": "Non évalué",
                "Méthode d'inférence": res.get("inference_method", "—"),
                "Dispersion (φ)": (
                    round(diag["dispersion_phi"], 2)
                    if pd.notna(diag.get("dispersion_phi"))
                    else np.nan
                ),
                "Autocorr. lag-1 (r1)": (
                    round(diag["lag1_autocorr"], 2)
                    if pd.notna(diag.get("lag1_autocorr"))
                    else np.nan
                ),
                "p Ljung–Box": (
                    round(diag["ljung_box_pvalue"], 4)
                    if pd.notna(diag.get("ljung_box_pvalue"))
                    else np.nan
                ),
                "Statut": res.get("status", "Non calculé"),
                "_calculable": bool(res.get("calculable", False)),
                "_slope_pp": res.get("slope_pp_per_year", np.nan),
            }
        )

    result_df = pd.DataFrame(rows)
    calc_mask = result_df["_calculable"] & result_df["p-valeur"].notna()
    n_tested = int(calc_mask.sum())

    if n_tested > 0:
        raw_pvals = result_df.loc[calc_mask, "p-valeur"].to_numpy(dtype=float)
        reject_fdr, pvals_fdr, _, _ = multipletests(
            raw_pvals, alpha=fdr_alpha, method="fdr_bh"
        )
        result_df.loc[calc_mask, "p-valeur FDR (BH)"] = pvals_fdr

        directions = []
        for is_sig, slope_val, p_fdr in zip(
            reject_fdr,
            result_df.loc[calc_mask, "_slope_pp"].to_numpy(dtype=float),
            pvals_fdr,
        ):
            if is_sig:
                if slope_val > 0:
                    directions.append(f"Hausse significative (FDR < {fdr_alpha:.0%})")
                elif slope_val < 0:
                    directions.append(f"Baisse significative (FDR < {fdr_alpha:.0%})")
                else:
                    directions.append("Stable")
            else:
                directions.append("Tendance non significative après FDR")
        result_df.loc[calc_mask, "Direction / Interprétation"] = directions

    # Tri : couples calculables d'abord (triés par p-valeur FDR croissante puis effectif N décroissant),
    # suivis des couples non calculables triés par effectif N décroissant.
    result_df = result_df.sort_values(
        by=["_calculable", "p-valeur FDR (BH)", "p-valeur", "Testés (N = S+I+R)"],
        ascending=[False, True, True, False],
        na_position="last",
    ).reset_index(drop=True)

    summary_meta = {
        "n_pairs_total": int(len(result_df)),
        "n_pairs_tested_fdr": n_tested,
        "n_pairs_excluded": int(len(result_df) - n_tested),
        "n_significant_fdr": int(
            (result_df.loc[calc_mask, "p-valeur FDR (BH)"] < fdr_alpha).sum()
        )
        if n_tested > 0
        else 0,
        "fdr_alpha": fdr_alpha,
        "min_months": min_months,
        "min_total_n": min_total_n,
    }

    display_cols = [
        "Bactérie",
        "Antibiotique",
        "Mois observés",
        "Période",
        "Total R/N",
        "%R global",
        "Pente (%R pts/an)",
        "IC 95 % pente (pts/an)",
        "Pente logit (log-odds/an)",
        "OR annuel",
        "IC 95 % OR annuel",
        "p-valeur",
        "p-valeur FDR (BH)",
        "Direction / Interprétation",
        "Méthode d'inférence",
        "Dispersion (φ)",
        "Autocorr. lag-1 (r1)",
        "p Ljung–Box",
        "Statut",
        "Résistants (R)",
        "Testés (N = S+I+R)",
        "bacterie_raw",
        "antibiotique_raw",
    ]
    return result_df[display_cols], summary_meta
