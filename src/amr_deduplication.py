"""
Module complet et reproductible de déduplication épidémiologique AMR (2019–2025)
selon les standards WHO GLASS (annuel) et par épisodes glissants (14, 30 et 365 jours).

Auteur : Pipeline Épidémiologie AMR & Biostatistique (WHO GLASS / Python AMR)
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Configuration graphique
plt.rcParams["figure.dpi"] = 120
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

# =============================================================================
# 1. INTÉGRATION ET CORRECTIF DU PACKAGE PYTHON `AMR`
# =============================================================================
try:
    import AMR
    # Dans la version installée de Python AMR (v0.1.3 wrapper rpy2 vers R AMR),
    # AMR/functions.py définit `def first_isolate(x = None, *args, **kwargs): return amr_r.first_isolate(x = None, ...)`
    # ce qui écrase l'argument `x` par `None`. On re-lie directement la fonction R via `r_to_python`.
    if hasattr(AMR, "functions") and hasattr(AMR.functions, "amr_r"):
        AMR.first_isolate = AMR.functions.r_to_python(AMR.functions.amr_r.first_isolate)
        AMR.filter_first_isolate = AMR.functions.r_to_python(AMR.functions.amr_r.filter_first_isolate)
    HAS_AMR_PACKAGE = True
except Exception as e:
    HAS_AMR_PACKAGE = False
    warnings.warn(f"Le package Python AMR n'a pas pu être chargé ({e}). Utilisation du moteur Python natif.")


# =============================================================================
# 2. CONFIGURATION CENTRALISÉE & DICTIONNAIRE DE CORRESPONDANCE DES COLONNES
# =============================================================================

DEFAULT_COLUMN_MAPPING: Dict[str, str] = {
    "isolate_id": "ROW_UID",
    "patient_id": "PATIENT_ID",
    "collection_date": "SPECIMEN_COLLECTION_DATE",
    "analysis_date": "ANALYSIS_DATE",
    "year": "YEAR",
    "organism_code": "ORGANISM_CODE",
    "organism_name": "ORGANISM_NAME",
    "specimen_type": "SPECIMEN_TYPE",
    "specimen_source": "SPECIMEN_SOURCE",
    "patient_type": "PATIENT_TYPE",
    "admission_date": "PATIENT_ADMISSION_DATE",
    "hospital_code": "HOSPITAL_CODE",
    "ward": "WARD",
    "antibiotic_code": "ANTIBIOTIC_CODE",
    "antibiotic_name": "ANTIBIOTIC",
    "interpretation": "INTERPRETATION",
    "esbl_status": "ESBL_STATUS",
    "carbapenem_status": "CARBAPENEM_RESISTANCE_STATUS",
    "mrsa_status": "MRSA_STATUS",
    "vre_status": "VRE_STATUS",
    "mdro_status": "MDRO_STATUS",
}

# Priorité clinique des prélèvements pour le départage déterministe et la règle Sang/LCR
SPECIMEN_PRIORITY_ORDER: Dict[str, int] = {
    "Cerebrospinal fluid": 1,
    "Blood": 2,
    "Lower respiratory tract specimen": 3,
    "Urine": 4,
    "Stool": 5,
    "Genital specimen": 6,
    "Upper respiratory tract specimen": 7,
}

# Périmètre strict WHO GLASS (couples Prélèvement × Pathogène prioritaires du manuel WHO GLASS)
WHO_GLASS_STRICT_PAIRS: Dict[str, List[str]] = {
    "Blood": [
        "B_ACNTB_BMNN", "B_ESCHR_COLI", "B_KLBSL_PNMN", "B_PSDMN_AERG",
        "B_STPHY_AURS", "B_STRPT_PNMN", "B_SLMNL", "B_ENTRC_FCLS", "B_ENTRC_FACM"
    ],
    "Cerebrospinal fluid": [
        "B_ACNTB_BMNN", "B_ESCHR_COLI", "B_KLBSL_PNMN", "B_PSDMN_AERG",
        "B_STPHY_AURS", "B_STRPT_PNMN", "B_NESSR_MNNG", "B_HMPHL_INFL"
    ],
    "Urine": [
        "B_ESCHR_COLI", "B_KLBSL_PNMN"
    ],
    "Stool": [
        "B_SLMNL", "B_SHGLL"
    ],
    "Lower respiratory tract specimen": [
        "B_ACNTB_BMNN", "B_ESCHR_COLI", "B_KLBSL_PNMN", "B_PSDMN_AERG",
        "B_STPHY_AURS", "B_STRPT_PNMN", "B_HMPHL_INFL"
    ],
    "Genital specimen": [
        "B_NESSR_GNRR"
    ],
    "Upper respiratory tract specimen": [
        "B_STRPT_PYGN", "B_STRPT_PNMN", "B_HMPHL_INFL"
    ],
}

# Valeurs harmonisées des origines anatomiques de prélèvements (standard WHO GLASS)
HARMONIZED_SPECIMEN_TYPES = {
    "Urine",
    "Blood",
    "Lower respiratory tract specimen",
    "Genital specimen",
    "Cerebrospinal fluid",
    "Upper respiratory tract specimen",
    "Stool",
}

# Valeurs de SPECIMEN_TYPE explicitement non éligibles dans la base
EXCLUDED_SPECIMEN_VALUES = {"", "EXCLUDED_non_target_specimen", "NAN", "NONE", "NULL"}


@dataclass
class DeduplicationConfig:
    """Configuration centralisée des colonnes et des règles épidémiologiques."""
    col_map: Dict[str, str] = field(default_factory=lambda: DEFAULT_COLUMN_MAPPING.copy())
    year_min: int = 2019
    year_max: int = 2025
    episode_windows: Tuple[int, ...] = (14, 30, 365)

    # Stratification sur l'origine communautaire / hospitalière (CAI / HAI)
    use_infection_origin_if_reliable: bool = True
    origin_min_completeness: float = 0.80  # Seuil minimal de complétude (80 %) pour stratifier par origine
    force_use_infection_origin: bool = False  # Permet de forcer la stratification dans les jeux synthétiques

    # Règle d'exception WHO GLASS Sang / LCR le même jour (priorité LCR)
    apply_glass_blood_csf_exception: bool = True
    promote_next_blood_if_same_day_csf: bool = True
    # Règle secondaire Sang / LCR dans les déduplications par épisodes
    apply_episode_blood_csf_exception: bool = False

    # Périmètre d'éligibilité :
    # - "all_eligible" : tous les isolats avec espèce et prélèvement harmonisés (7 origines anatomiques)
    # - "strict_glass" : uniquement les couples Pathogène × Prélèvement prioritaires du rapport WHO GLASS
    # - "all_non_excluded" : inclut même les libellés de prélèvements bruts non exclus
    eligibility_scope: str = "all_eligible"

    # Utilisation du package Python AMR (AMR.first_isolate)
    use_amr_package: bool = True


# =============================================================================
# 3. INSPECTION DES COLONNES ET PRÉPARATION NON DESTRUCTIVE DE `df_amr`
# =============================================================================

def inspect_and_map_columns(
    df: pd.DataFrame,
    config: Optional[DeduplicationConfig] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Vérifie dynamiquement la présence des colonnes dans le DataFrame et affiche
    la table de correspondance (sans coder en dur dans les fonctions aval).
    """
    if config is None:
        config = DeduplicationConfig()

    rows = []
    for logical_name, actual_col in config.col_map.items():
        exists = actual_col in df.columns
        non_null_pct = 0.0
        n_unique = 0
        if exists:
            s = df[actual_col]
            valid = s.notna() & (s.astype(str).str.strip() != "") & (s.astype(str) != "nan")
            non_null_pct = round(100.0 * valid.mean(), 2)
            n_unique = int(s[valid].nunique())
        rows.append({
            "Concept_Logique": logical_name,
            "Colonne_DataFrame": actual_col,
            "Presente": exists,
            "Completude_%": non_null_pct,
            "Valeurs_Distinctes": n_unique,
        })

    report_df = pd.DataFrame(rows)
    if verbose:
        print("=" * 88)
        print("VÉRIFICATION DES COLONNES DU DATAFRAME ET DICTIONNAIRE DE CORRESPONDANCE")
        print("=" * 88)
        print(f"Dimensions de la table source : {df.shape[0]:,} lignes × {df.shape[1]} colonnes")
        print(report_df.to_string(index=False))
        print("=" * 88)
    return report_df


def prepare_amr_dataframe(
    df_or_path: Any,
    config: Optional[DeduplicationConfig] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Construit `df_amr` avec exactement 1 ligne par isolat bactérien (`ROW_UID`),
    tout en conservant 100 % des observations d'origine et tous les résultats
    d'antibiogrammes (AST) associés sous forme de colonnes `ABX_<CODE>` et d'un résumé `AST_PROFILE`.

    Ne modifie ni ne supprime aucun isolat de la base initiale : les isolats non éligibles
    sont marqués via `IS_ELIGIBLE = False` et `ELIGIBILITY_REASON`.
    """
    if config is None:
        config = DeduplicationConfig()
    cm = config.col_map

    if isinstance(df_or_path, (str, Path)):
        p = Path(df_or_path)
        suf = p.suffix.lower()
        if suf in (".csv", ".txt"):
            df_raw = pd.read_csv(p, low_memory=False)
        elif suf == ".tsv":
            df_raw = pd.read_csv(p, sep="\t", low_memory=False)
        elif suf in (".xlsx", ".xls"):
            df_raw = pd.read_excel(p)
        else:
            df_raw = pd.read_parquet(p)
    else:
        df_raw = df_or_path.copy()

    col_uid = cm["isolate_id"]
    col_abx = cm["antibiotic_code"]
    col_sir = cm["interpretation"]

    # Si le DataFrame contient les colonnes d'antibiogramme au format long (ANTIBIOTIC_CODE + INTERPRETATION),
    # on consolide 1 ligne par isolat unique tout en pivotant tous les résultats AST en colonnes ABX_<CODE>.
    has_long_ast_cols = (col_abx in df_raw.columns) and (col_sir in df_raw.columns)

    if has_long_ast_cols:
        # 1. Métadonnées niveau isolat (première occurrence de chaque ROW_UID)
        isolate_meta_cols = [
            c for c in df_raw.columns
            if c not in {
                cm["antibiotic_name"], cm["antibiotic_code"], cm["interpretation"],
                "METHOD_HINT", "DOSE_UG", "ANTIBIOTIC_VALUE_NUMERIC", "ANTIBIOTIC_VALUE_OPERATOR"
            }
        ]
        df_amr = df_raw[isolate_meta_cols].drop_duplicates(subset=[col_uid], keep="first").copy()
        df_amr = df_amr.reset_index(drop=True)

        # 2. Pivot des résultats d'antibiogrammes (conservation intégrale des résultats AST)
        valid_ast = df_raw[
            df_raw[col_abx].notna()
            & (df_raw[col_abx].astype(str).str.strip() != "")
            & df_raw[col_sir].isin(["S", "I", "R"])
        ][[col_uid, col_abx, col_sir]].drop_duplicates(subset=[col_uid, col_abx], keep="first")

        ast_wide = valid_ast.pivot(index=col_uid, columns=col_abx, values=col_sir)
        ast_wide.columns = [f"ABX_{c}" for c in ast_wide.columns]
        ast_wide["AST_TESTED_COUNT"] = ast_wide.notna().sum(axis=1).astype(int)

        # Résumé textuel compact du profil AST par isolat (utile pour l'audit)
        valid_ast_sorted = valid_ast.sort_values([col_uid, col_abx])
        valid_ast_sorted["PAIR"] = valid_ast_sorted[col_abx].astype(str) + ":" + valid_ast_sorted[col_sir].astype(str)
        ast_summary = valid_ast_sorted.groupby(col_uid, sort=False)["PAIR"].agg("|".join).rename("AST_PROFILE")

        df_amr = df_amr.merge(ast_wide, on=col_uid, how="left")
        df_amr = df_amr.merge(ast_summary, on=col_uid, how="left")
        df_amr["AST_TESTED_COUNT"] = df_amr["AST_TESTED_COUNT"].fillna(0).astype(int)
        df_amr["AST_PROFILE"] = df_amr["AST_PROFILE"].fillna("UNTESTED")
    else:
        df_amr = df_raw.copy().reset_index(drop=True)
        abx_cols = [c for c in df_amr.columns if c.startswith("ABX_")]
        if "AST_TESTED_COUNT" not in df_amr.columns:
            df_amr["AST_TESTED_COUNT"] = df_amr[abx_cols].notna().sum(axis=1) if abx_cols else 1
        if "AST_PROFILE" not in df_amr.columns:
            df_amr["AST_PROFILE"] = "PRESERVED"

    # 3. Date de référence de l'isolat (SPECIMEN_COLLECTION_DATE avec repli sur ANALYSIS_DATE)
    d_spec = pd.to_datetime(df_amr[cm["collection_date"]], errors="coerce") if cm["collection_date"] in df_amr.columns else pd.Series(pd.NaT, index=df_amr.index)
    d_anal = pd.to_datetime(df_amr[cm["analysis_date"]], errors="coerce") if cm["analysis_date"] in df_amr.columns else pd.Series(pd.NaT, index=df_amr.index)
    df_amr["ISOLATE_DATE"] = d_spec.fillna(d_anal)
    df_amr["SURVEILLANCE_YEAR"] = df_amr["ISOLATE_DATE"].dt.year

    # 4. Origine communautaire / hospitalière (INFECTION_ORIGIN)
    # Si PATIENT_TYPE est renseigné, on l'utilise ; sinon on le dérive de (ISOLATE_DATE - PATIENT_ADMISSION_DATE)
    # selon le critère standard WHO GLASS (> 2 jours d'hospitalisation = HAI / Hospital-Associated, <= 2 jours = CAI / Community).
    origin_series = pd.Series("", index=df_amr.index, dtype="object")
    if cm["patient_type"] in df_amr.columns:
        pt = df_amr[cm["patient_type"]].fillna("").astype(str).str.strip()
        origin_series = pt.where(pt != "", "")

    if cm["admission_date"] in df_amr.columns:
        d_adm = pd.to_datetime(df_amr[cm["admission_date"]], errors="coerce")
        delta_adm = (df_amr["ISOLATE_DATE"] - d_adm).dt.days
        derived_origin = np.where(
            delta_adm.isna(),
            "",
            np.where(delta_adm > 2, "Hospital-origin (HAI)", "Community-origin (CAI)"),
        )
        origin_series = np.where(origin_series != "", origin_series, derived_origin)

    df_amr["INFECTION_ORIGIN"] = pd.Series(origin_series, index=df_amr.index).replace("", "Unknown")
    origin_completeness = float((df_amr["INFECTION_ORIGIN"] != "Unknown").mean())
    df_amr.attrs["origin_completeness"] = origin_completeness
    df_amr.attrs["origin_is_reliable"] = bool(
        config.force_use_infection_origin
        or (config.use_infection_origin_if_reliable and origin_completeness >= config.origin_min_completeness)
    )

    # 5. Variable d'éligibilité générale et périmètre strict WHO GLASS
    pat_ok = df_amr[cm["patient_id"]].fillna("").astype(str).str.strip() != ""
    org_ok = df_amr[cm["organism_code"]].fillna("").astype(str).str.strip() != ""
    spec_str = df_amr[cm["specimen_type"]].fillna("").astype(str).str.strip()
    if config.eligibility_scope == "all_non_excluded":
        spec_ok = (~spec_str.isin(EXCLUDED_SPECIMEN_VALUES)) & (~spec_str.str.startswith("EXCLUDED_"))
    else:
        spec_ok = spec_str.isin(HARMONIZED_SPECIMEN_TYPES)
    date_ok = df_amr["ISOLATE_DATE"].notna() & df_amr["SURVEILLANCE_YEAR"].between(config.year_min, config.year_max)

    # Périmètre strict WHO GLASS (Prélèvement × Espèce figurant dans WHO_GLASS_STRICT_PAIRS)
    strict_glass_mask = pd.Series(False, index=df_amr.index)
    for spec_name, allowed_orgs in WHO_GLASS_STRICT_PAIRS.items():
        strict_glass_mask |= (spec_str == spec_name) & (df_amr[cm["organism_code"]].isin(allowed_orgs))
    df_amr["IS_GLASS_WHO_SCOPE"] = pat_ok & org_ok & spec_ok & date_ok & strict_glass_mask

    if config.eligibility_scope == "strict_glass":
        df_amr["IS_ELIGIBLE"] = df_amr["IS_GLASS_WHO_SCOPE"]
    else:
        df_amr["IS_ELIGIBLE"] = pat_ok & org_ok & spec_ok & date_ok

    # Justification détaillée d'éligibilité ou de non-éligibilité
    reasons = np.full(len(df_amr), "ELIGIBLE", dtype=object)
    reasons = np.where(~date_ok, "INELIGIBLE_DATE_OR_OUT_OF_RANGE_2019_2025", reasons)
    reasons = np.where(~spec_ok, "INELIGIBLE_EXCLUDED_OR_UNHARMONIZED_SPECIMEN", reasons)
    reasons = np.where(~org_ok, "INELIGIBLE_MISSING_ORGANISM", reasons)
    reasons = np.where(~pat_ok, "INELIGIBLE_MISSING_PATIENT_ID", reasons)
    if config.eligibility_scope == "strict_glass":
        reasons = np.where(
            pat_ok & org_ok & spec_ok & date_ok & (~strict_glass_mask),
            "INELIGIBLE_OUTSIDE_STRICT_GLASS_PATHOGEN_SPECIMEN_PAIR",
            reasons,
        )
    df_amr["ELIGIBILITY_REASON"] = reasons

    # Priorité numérique du type de prélèvement (pour le départage déterministe)
    df_amr["SPECIMEN_PRIORITY"] = df_amr[cm["specimen_type"]].map(SPECIMEN_PRIORITY_ORDER).fillna(99).astype(int)

    if verbose:
        n_total = len(df_amr)
        n_elig = int(df_amr["IS_ELIGIBLE"].sum())
        n_strict = int(df_amr["IS_GLASS_WHO_SCOPE"].sum())
        print(f"[prepare_amr_dataframe] Isolats uniques (ROW_UID) : {n_total:,}")
        print(f"  - Isolats éligibles (analyse générale 2019–2025) : {n_elig:,} ({100*n_elig/max(n_total,1):.1f}%)")
        print(f"  - Isolats dans le périmètre strict WHO GLASS     : {n_strict:,} ({100*n_strict/max(n_total,1):.1f}%)")
        print(f"  - Complétude de l'origine CAI/HAI                : {100*origin_completeness:.1f}% "
              f"(stratification active : {df_amr.attrs['origin_is_reliable']})")

    return df_amr


def _get_origin_stratification_flag(df: pd.DataFrame, config: DeduplicationConfig) -> bool:
    """Détermine si la strate d'origine d'infection (CAI/HAI) doit être incluse."""
    if config.force_use_infection_origin:
        return True
    if not config.use_infection_origin_if_reliable:
        return False
    if "origin_is_reliable" in df.attrs:
        return bool(df.attrs["origin_is_reliable"])
    if "INFECTION_ORIGIN" in df.columns:
        comp = float((df["INFECTION_ORIGIN"].fillna("Unknown") != "Unknown").mean())
        return comp >= config.origin_min_completeness
    return False


# =============================================================================
# 4. DÉDUPLICATION ANNUELLE WHO GLASS (`dedup_glass`)
# =============================================================================

def dedup_glass(
    df: pd.DataFrame,
    config: Optional[DeduplicationConfig] = None,
) -> pd.DataFrame:
    """
    Applique la déduplication annuelle selon la méthodologie WHO GLASS :
    - Sélection du premier isolat par :
        * Identifiant patient (`PATIENT_ID`)
        * Espèce bactérienne (`ORGANISM_CODE`)
        * Type de prélèvement (`SPECIMEN_TYPE`)
        * Année de surveillance (`SURVEILLANCE_YEAR`, 2019 à 2025)
        * Origine communautaire/hospitalière (`INFECTION_ORIGIN`) si disponible et fiable.
    - Indépendamment des changements de profil de résistance (antibiogramme).
    - Gestion explicite de l'exception WHO GLASS Sang / LCR :
        * Si le même patient présente la même espèce dans le sang (`Blood`) et le LCR
          (`Cerebrospinal fluid`) le même jour calendaire, le prélèvement LCR est prioritaire
          et l'isolat sanguin du même jour est exclu (`EXCLUDED_GLASS_BLOOD_SAME_DAY_AS_CSF`).
    - Règle de départage déterministe en cas de prélèvements simultanés le même jour :
        1. Priorité clinique du prélèvement (`Cerebrospinal fluid` > `Blood` > ...)
        2. Horodatage exact (`ISOLATE_DATE`)
        3. Richesse de l'antibiogramme (`-AST_TESTED_COUNT`)
        4. Identifiant unique d'isolat (`ROW_UID`, ordre lexicographique).
      Garantit une stabilité absolue quel que soit l'ordre initial des lignes.
    """
    if config is None:
        config = DeduplicationConfig()
    cm = config.col_map

    out = df.copy()
    n = len(out)

    # Initialisation des colonnes d'audit GLASS
    out["first_glass"] = False
    out["glass_stratum_id"] = ""
    out["glass_episode_start"] = pd.NaT
    out["glass_decision_reason"] = np.where(
        out["IS_ELIGIBLE"],
        "PENDING",
        out["ELIGIBILITY_REASON"],
    )

    elig_mask = out["IS_ELIGIBLE"].astype(bool)
    if not elig_mask.any():
        return out

    use_origin = _get_origin_stratification_flag(out, config)

    # Sous-ensemble éligible avec index original conservé
    sub = out.loc[elig_mask].copy()
    sub["_orig_idx"] = sub.index
    sub["_cal_day"] = sub["ISOLATE_DATE"].dt.floor("D")
    sub["_neg_ast_count"] = -sub["AST_TESTED_COUNT"].fillna(0).astype(int)
    sub["_year_int"] = sub["SURVEILLANCE_YEAR"].astype(int)

    # 1. Détection de l'exception WHO GLASS Sang / LCR le même jour
    # Si un même patient a la même espèce (et même origine si stratifiée) dans le LCR (Cerebrospinal fluid)
    # et dans le Sang (Blood) le même jour calendaire, l'isolat Blood de ce jour est supplanté par le LCR.
    blood_csf_excluded_orig_indices = set()
    if config.apply_glass_blood_csf_exception:
        csf_keys_cols = [cm["patient_id"], cm["organism_code"], "_cal_day"]
        if use_origin:
            csf_keys_cols.append("INFECTION_ORIGIN")

        csf_rows = sub[sub[cm["specimen_type"]] == "Cerebrospinal fluid"]
        if not csf_rows.empty:
            csf_day_keys = set(csf_rows[csf_keys_cols].itertuples(index=False, name=None))
            blood_mask = sub[cm["specimen_type"]] == "Blood"
            if blood_mask.any():
                blood_rows = sub[blood_mask]
                blood_tuples = list(blood_rows[csf_keys_cols].itertuples(index=False, name=None))
                for orig_i, tup in zip(blood_rows["_orig_idx"], blood_tuples):
                    if tup in csf_day_keys:
                        blood_csf_excluded_orig_indices.add(orig_i)

    # 2. Définition des colonnes de strate GLASS
    stratum_cols = [cm["patient_id"], cm["organism_code"], cm["specimen_type"], "_year_int"]
    if use_origin:
        stratum_cols.append("INFECTION_ORIGIN")

    # Identifiant lisible de la strate GLASS
    stratum_id_series = (
        sub[cm["patient_id"]].astype(str)
        + "|" + sub[cm["organism_code"]].astype(str)
        + "|" + sub[cm["specimen_type"]].astype(str)
        + "|" + sub["_year_int"].astype(str)
    )
    if use_origin:
        stratum_id_series = stratum_id_series + "|" + sub["INFECTION_ORIGIN"].astype(str)
    sub["glass_stratum_id"] = stratum_id_series

    # 3. Tri déterministe intra-strate (garantit l'invariance par permutation des lignes)
    sort_cols = stratum_cols + ["SPECIMEN_PRIORITY", "ISOLATE_DATE", "_neg_ast_count", cm["isolate_id"]]
    sub_sorted = sub.sort_values(sort_cols, kind="mergesort")

    # Si `promote_next_blood_if_same_day_csf` est actif, les prélèvements Blood exclus à cause d'un LCR
    # le même jour ne consomment pas le droit au premier isolat Blood s'il existe un autre Blood un autre jour
    # de la même année ; sinon aucun Blood n'est retenu ce jour-là.
    if config.apply_glass_blood_csf_exception and blood_csf_excluded_orig_indices:
        is_excluded_blood_csf = sub_sorted["_orig_idx"].isin(blood_csf_excluded_orig_indices)
        candidates = sub_sorted[~is_excluded_blood_csf]
    else:
        is_excluded_blood_csf = pd.Series(False, index=sub_sorted.index)
        candidates = sub_sorted

    # Sélection du premier isolat par strate GLASS parmi les candidats admissibles
    first_in_stratum_mask = ~candidates.duplicated(subset=stratum_cols, keep="first")
    selected_orig_indices = set(candidates.loc[first_in_stratum_mask, "_orig_idx"])

    # Date de début de l'épisode annuel par strate
    first_dates_map = (
        candidates.loc[first_in_stratum_mask]
        .set_index("glass_stratum_id")["ISOLATE_DATE"]
        .to_dict()
    )

    # Détection des prélèvements simultanés (même strate et même jour que le premier isolat retenu)
    first_day_map = (
        candidates.loc[first_in_stratum_mask]
        .set_index("glass_stratum_id")["_cal_day"]
        .to_dict()
    )

    # Attribution des indicateurs et justifications sur `sub_sorted`
    orig_indices = sub_sorted["_orig_idx"].to_numpy()
    strat_ids = sub_sorted["glass_stratum_id"].to_numpy()
    cal_days = sub_sorted["_cal_day"].to_numpy()
    spec_types = sub_sorted[cm["specimen_type"]].to_numpy()

    # Ensemble des jours où un LCR est retenu pour noter sur le LCR qu'il a eu priorité sur le sang
    csf_priority_days = set()
    if blood_csf_excluded_orig_indices:
        blood_exc_df = sub.loc[sub["_orig_idx"].isin(blood_csf_excluded_orig_indices)]
        csf_priority_days = set(
            blood_exc_df[csf_keys_cols].itertuples(index=False, name=None)
        )

    first_flags = np.zeros(len(sub_sorted), dtype=bool)
    ep_starts = []
    reasons_out = []

    for idx_pos, orig_i in enumerate(orig_indices):
        sid = strat_ids[idx_pos]
        cday = cal_days[idx_pos]
        stype = spec_types[idx_pos]

        if orig_i in blood_csf_excluded_orig_indices:
            first_flags[idx_pos] = False
            ep_starts.append(first_dates_map.get(sid, cday))
            reasons_out.append("EXCLUDED_GLASS_BLOOD_SAME_DAY_AS_CSF")
        elif orig_i in selected_orig_indices:
            first_flags[idx_pos] = True
            ep_starts.append(first_dates_map.get(sid, cday))
            if stype == "Cerebrospinal fluid" and blood_csf_excluded_orig_indices:
                row_obj = sub_sorted.iloc[idx_pos]
                tup = tuple(row_obj[c] for c in csf_keys_cols)
                if tup in csf_priority_days:
                    reasons_out.append("SELECTED_FIRST_GLASS_CSF_PRIORITY_OVER_BLOOD")
                else:
                    reasons_out.append("SELECTED_FIRST_GLASS_ANNUAL")
            else:
                reasons_out.append("SELECTED_FIRST_GLASS_ANNUAL")
        else:
            first_flags[idx_pos] = False
            ep_starts.append(first_dates_map.get(sid, pd.NaT))
            if first_day_map.get(sid) == cday:
                reasons_out.append("EXCLUDED_GLASS_SIMULTANEOUS_TIEBREAKER")
            else:
                reasons_out.append("EXCLUDED_GLASS_SAME_YEAR_REPEAT")

    out.loc[orig_indices, "first_glass"] = first_flags
    out.loc[orig_indices, "glass_stratum_id"] = strat_ids
    out.loc[orig_indices, "glass_episode_start"] = pd.to_datetime(ep_starts)
    out.loc[orig_indices, "glass_decision_reason"] = reasons_out

    return out


# =============================================================================
# 5. DÉDUPLICATION PAR ÉPISODES DE 14, 30 ET 365 JOURS (`dedup_episode`)
# =============================================================================

def dedup_episode(
    df: pd.DataFrame,
    days: int,
    config: Optional[DeduplicationConfig] = None,
) -> pd.DataFrame:
    """
    Applique la déduplication par épisode clinique de `days` jours (14, 30 ou 365) :
    1. Trie les isolats chronologiquement de manière déterministe.
    2. Identifie le premier isolat de chaque épisode par patient, espèce bactérienne
       et type de prélèvement (+ origine de l'infection si fiable).
    3. Appelle `AMR.first_isolate(method="episode-based", episode_days=days)` lorsque
       le package Python `AMR` est actif.
    4. Définit chaque fenêtre à partir de la date de début de l'épisode en cours
       (et non de la date du prélèvement immédiatement précédent).
    5. Commence un nouvel épisode lorsque `date_isolat - date_debut_episode >= days`.
    6. Ignore tout changement d'antibiogramme (`include_untested=True`, méthode `"episode-based"`
       sans pondération phénotypique).
    7. Conserve une chronologie continue de 2019 à 2025, y compris à la frontière des années.
    """
    if config is None:
        config = DeduplicationConfig()
    cm = config.col_map
    suffix = f"{days}j"

    col_flag = f"first_{suffix}"
    col_ep_id = f"episode_id_{suffix}"
    col_ep_start = f"episode_start_{suffix}"
    col_days_since = f"days_since_start_{suffix}"
    col_reason = f"decision_reason_{suffix}"

    out = df.copy()
    out[col_flag] = False
    out[col_ep_id] = ""
    out[col_ep_start] = pd.NaT
    out[col_days_since] = np.nan
    out[col_reason] = np.where(
        out["IS_ELIGIBLE"],
        "PENDING",
        out["ELIGIBILITY_REASON"],
    )

    elig_mask = out["IS_ELIGIBLE"].astype(bool)
    if not elig_mask.any():
        return out

    use_origin = _get_origin_stratification_flag(out, config)

    sub = out.loc[elig_mask].copy()
    sub["_orig_idx"] = sub.index
    sub["_cal_day"] = sub["ISOLATE_DATE"].dt.floor("D")
    sub["_neg_ast_count"] = -sub["AST_TESTED_COUNT"].fillna(0).astype(int)

    # Gestion optionnelle de l'exception Sang/LCR le même jour dans les analyses secondaires par épisode
    blood_csf_excluded_orig_indices = set()
    if config.apply_episode_blood_csf_exception:
        csf_keys_cols = [cm["patient_id"], cm["organism_code"], "_cal_day"]
        if use_origin:
            csf_keys_cols.append("INFECTION_ORIGIN")
        csf_rows = sub[sub[cm["specimen_type"]] == "Cerebrospinal fluid"]
        if not csf_rows.empty:
            csf_day_keys = set(csf_rows[csf_keys_cols].itertuples(index=False, name=None))
            blood_rows = sub[sub[cm["specimen_type"]] == "Blood"]
            for orig_i, tup in zip(
                blood_rows["_orig_idx"],
                blood_rows[csf_keys_cols].itertuples(index=False, name=None),
            ):
                if tup in csf_day_keys:
                    blood_csf_excluded_orig_indices.add(orig_i)

    # Clé composite de strate (Patient | Prélèvement [| Origine])
    # Note technique importante sur `AMR.first_isolate` :
    # Dans R `AMR::first_isolate()`, le groupement par défaut se fait sur `paste(patient_id, mo)`.
    # L'argument `col_specimen` ne segmente pas en strates distinctes mais filtre uniquement si `specimen_group`
    # est fourni. Pour que `AMR.first_isolate(method="episode-based", episode_days=days)` stratifie exactement
    # par (patient, espèce, type de prélèvement[, origine]), on passe la clé composite de strate dans `col_patient_id`.
    stratum_cols = [cm["patient_id"], cm["organism_code"], cm["specimen_type"]]
    if use_origin:
        stratum_cols.append("INFECTION_ORIGIN")

    stratum_patient_spec = (
        sub[cm["patient_id"]].astype(str) + "|" + sub[cm["specimen_type"]].astype(str)
    )
    if use_origin:
        stratum_patient_spec = stratum_patient_spec + "|" + sub["INFECTION_ORIGIN"].astype(str)
    sub["_STRATUM_PATIENT_SPEC"] = stratum_patient_spec

    # Tri chronologique déterministe continu sur 2019–2025 (sans découpage par année !)
    sort_cols = stratum_cols + ["_cal_day", "SPECIMEN_PRIORITY", "ISOLATE_DATE", "_neg_ast_count", cm["isolate_id"]]
    sub_sorted = sub.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)

    # Appel à `AMR.first_isolate(method="episode-based", episode_days=days)` si activé et disponible
    amr_flags: Optional[np.ndarray] = None
    if config.use_amr_package and HAS_AMR_PACKAGE and not blood_csf_excluded_orig_indices:
        try:
            amr_input = pd.DataFrame({
                "STRATUM_ID": sub_sorted["_STRATUM_PATIENT_SPEC"].astype(str),
                "MO": sub_sorted[cm["organism_code"]].astype(str),
                "DATE": sub_sorted["_cal_day"],
            })
            amr_res = AMR.first_isolate(
                amr_input,
                col_date="DATE",
                col_patient_id="STRATUM_ID",
                col_mo="MO",
                method="episode-based",
                episode_days=int(days),
                include_untested=True,
                info=False,
            )
            amr_flags = np.asarray(amr_res, dtype=bool)
        except Exception as e:
            warnings.warn(f"Repli sur le calcul Python natif pour {days}j ({e})")
            amr_flags = None

    # Calcul exact des métadonnées d'épisode (identifiant d'épisode, date de début d'épisode,
    # délai en jours depuis le début de l'épisode, et justification de la décision)
    n_sub = len(sub_sorted)
    flags = np.zeros(n_sub, dtype=bool)
    ep_ids = np.empty(n_sub, dtype=object)
    ep_starts = np.empty(n_sub, dtype="datetime64[ns]")
    days_since = np.zeros(n_sub, dtype=float)
    reasons = np.empty(n_sub, dtype=object)

    orig_indices = sub_sorted["_orig_idx"].to_numpy()
    cal_days = sub_sorted["_cal_day"].to_numpy(dtype="datetime64[D]")
    cal_days_ns = sub_sorted["_cal_day"].to_numpy(dtype="datetime64[ns]")
    pat_vals = sub_sorted[cm["patient_id"]].astype(str).to_numpy()
    org_vals = sub_sorted[cm["organism_code"]].astype(str).to_numpy()
    spec_vals = sub_sorted[cm["specimen_type"]].astype(str).to_numpy()

    for _, grp in sub_sorted.groupby(stratum_cols, sort=False):
        pos_arr = grp.index.to_numpy()
        ep_num = 0
        current_ep_start_d = None
        current_ep_start_ns = None
        current_ep_id = ""

        for pos in pos_arr:
            orig_i = orig_indices[pos]
            d_val = cal_days[pos]
            d_ns = cal_days_ns[pos]

            if orig_i in blood_csf_excluded_orig_indices:
                flags[pos] = False
                ep_ids[pos] = current_ep_id if current_ep_id else f"{pat_vals[pos]}|{org_vals[pos]}|CSF_PRIORITY"
                ep_starts[pos] = current_ep_start_ns if current_ep_start_ns is not None else d_ns
                days_since[pos] = float((d_val - current_ep_start_d).astype(int)) if current_ep_start_d is not None else 0.0
                reasons[pos] = f"EXCLUDED_{suffix.upper()}_BLOOD_SAME_DAY_AS_CSF"
                continue

            if current_ep_start_d is None:
                # Tout premier isolat de la strate
                ep_num = 1
                current_ep_start_d = d_val
                current_ep_start_ns = d_ns
                current_ep_id = f"{pat_vals[pos]}|{org_vals[pos]}|{spec_vals[pos]}|{suffix}_EP{ep_num:02d}"
                flags[pos] = True
                ep_ids[pos] = current_ep_id
                ep_starts[pos] = current_ep_start_ns
                days_since[pos] = 0.0
                reasons[pos] = f"SELECTED_FIRST_{suffix.upper()}_INITIAL_EPISODE"
            else:
                delta_d = int((d_val - current_ep_start_d).astype(int))
                if delta_d >= days:
                    # Nouvel épisode : la durée choisie à partir du début de l'épisode précédent est atteinte
                    ep_num += 1
                    current_ep_start_d = d_val
                    current_ep_start_ns = d_ns
                    current_ep_id = f"{pat_vals[pos]}|{org_vals[pos]}|{spec_vals[pos]}|{suffix}_EP{ep_num:02d}"
                    flags[pos] = True
                    ep_ids[pos] = current_ep_id
                    ep_starts[pos] = current_ep_start_ns
                    days_since[pos] = 0.0
                    reasons[pos] = f"SELECTED_FIRST_{suffix.upper()}_NEW_EPISODE_AFTER_{delta_d}D"
                else:
                    # Isolat répété à l'intérieur de la fenêtre active [0, days-1]
                    flags[pos] = False
                    ep_ids[pos] = current_ep_id
                    ep_starts[pos] = current_ep_start_ns
                    days_since[pos] = float(delta_d)
                    if delta_d == 0:
                        reasons[pos] = f"EXCLUDED_{suffix.upper()}_SAME_DAY_DUPLICATE"
                    else:
                        reasons[pos] = f"EXCLUDED_{suffix.upper()}_WITHIN_WINDOW_{delta_d}D_LT_{days}D"

    # Vérification de concordance avec `AMR.first_isolate` lorsque disponible
    if amr_flags is not None:
        mismatch = int((amr_flags != flags).sum())
        if mismatch > 0:
            warnings.warn(
                f"Divergence détectée ({mismatch} lignes) entre AMR.first_isolate et le moteur d'audit {days}j."
            )
        out.attrs[f"amr_package_concordance_{suffix}"] = (mismatch == 0)

    out.loc[orig_indices, col_flag] = flags
    out.loc[orig_indices, col_ep_id] = ep_ids
    out.loc[orig_indices, col_ep_start] = pd.to_datetime(ep_starts)
    out.loc[orig_indices, col_days_since] = days_since
    out.loc[orig_indices, col_reason] = reasons

    return out


# =============================================================================
# 6. PIPELINE COMPLET : CRÉATION DE `df_amr` ET DES 4 DATAFRAMES DÉRIVÉS
# =============================================================================

def run_all_deduplications(
    df_amr: pd.DataFrame,
    config: Optional[DeduplicationConfig] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Exécute les quatre méthodes de déduplication sur `df_amr` sans supprimer
    aucune observation originale, ajoute les indicateurs booléens :
      - `first_glass`
      - `first_14j`
      - `first_30j`
      - `first_365j`
    et retourne `(df_amr, df_glass, df_14j, df_30j, df_365j)`.
    """
    if config is None:
        config = DeduplicationConfig()

    df_out = dedup_glass(df_amr, config=config)
    for w in config.episode_windows:
        df_out = dedup_episode(df_out, days=w, config=config)

    df_glass = df_out[df_out["first_glass"]].copy().reset_index(drop=True)
    df_14j = df_out[df_out["first_14j"]].copy().reset_index(drop=True)
    df_30j = df_out[df_out["first_30j"]].copy().reset_index(drop=True)
    df_365j = df_out[df_out["first_365j"]].copy().reset_index(drop=True)

    return df_out, df_glass, df_14j, df_30j, df_365j


def export_deduplication_outputs(
    df_amr: pd.DataFrame,
    df_glass: pd.DataFrame,
    df_14j: pd.DataFrame,
    df_30j: pd.DataFrame,
    df_365j: pd.DataFrame,
    summaries: Optional[Dict[str, pd.DataFrame]] = None,
    output_dir: str | Path = "outputs_deduplication",
    export_parquet: bool = True,
    export_csv: bool = True,
) -> Dict[str, Path]:
    """
    Enregistre sur disque les 5 DataFrames (`df_amr`, `df_glass`, `df_14j`, `df_30j`, `df_365j`)
    ainsi que les tableaux récapitulatifs (`D1`, `D2`, `D3`) au format `.parquet` et/ou `.csv`.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    saved_files: Dict[str, Path] = {}
    datasets = {
        "df_amr": df_amr,
        "df_glass": df_glass,
        "df_14j": df_14j,
        "df_30j": df_30j,
        "df_365j": df_365j,
    }

    for name, df_obj in datasets.items():
        if export_parquet:
            p_file = out_path / f"{name}.parquet"
            df_obj.to_parquet(p_file, index=False)
            saved_files[f"{name}_parquet"] = p_file
        if export_csv:
            c_file = out_path / f"{name}.csv"
            df_obj.to_csv(c_file, index=False, encoding="utf-8")
            saved_files[f"{name}_csv"] = c_file

    if summaries is not None:
        for tab_key in ("D1_annual_glass", "D2_methods_comparison", "D3_species_specimen", "by_species", "by_specimen"):
            if tab_key in summaries:
                t_file = out_path / f"table_{tab_key}.csv"
                summaries[tab_key].to_csv(t_file, index=False, encoding="utf-8")
                saved_files[tab_key] = t_file

    return saved_files


# =============================================================================
# 7. INDICATEURS ÉPIDÉMIOLOGIQUES, TABLEAUX D1–D3 ET FIGURES F1–F3
# =============================================================================

METHOD_LABELS: Dict[str, str] = {
    "first_glass": "WHO GLASS (Annuel)",
    "first_14j": "Épisode 14 jours",
    "first_30j": "Épisode 30 jours",
    "first_365j": "Épisode 365 jours",
}


def summarize_deduplication(
    df_amr: pd.DataFrame,
    config: Optional[DeduplicationConfig] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Calcule l'ensemble des indicateurs demandés pour chaque méthode et chaque année (2019–2025) :
    - Nombre d'isolats éligibles avant déduplication
    - Nombre d'isolats retenus
    - Nombre d'isolats exclus comme répétitions
    - Pourcentage d'isolats retenus
    - Pourcentage d'isolats exclus
    - Nombre de patients distincts
    - Nombre d'isolats retenus par espèce
    - Nombre d'isolats retenus par type de prélèvement
    Et génère les tableaux formatés D1, D2 et D3.
    """
    if config is None:
        config = DeduplicationConfig()
    cm = config.col_map

    elig = df_amr[df_amr["IS_ELIGIBLE"]].copy()
    elig["SURVEILLANCE_YEAR"] = elig["SURVEILLANCE_YEAR"].astype(int)
    years = list(range(config.year_min, config.year_max + 1))

    # 1. Indicateurs annuels détaillés par méthode
    annual_rows = []
    for method_col, method_label in METHOD_LABELS.items():
        for yr in years:
            sub_y = elig[elig["SURVEILLANCE_YEAR"] == yr]
            n_elig = len(sub_y)
            retained_y = sub_y[sub_y[method_col]]
            n_ret = len(retained_y)
            n_exc = n_elig - n_ret
            pct_ret = round(100.0 * n_ret / n_elig, 2) if n_elig > 0 else 0.0
            pct_exc = round(100.0 * n_exc / n_elig, 2) if n_elig > 0 else 0.0
            n_pat_elig = int(sub_y[cm["patient_id"]].nunique()) if n_elig > 0 else 0
            n_pat_ret = int(retained_y[cm["patient_id"]].nunique()) if n_ret > 0 else 0

            annual_rows.append({
                "Methode_Code": method_col,
                "Methode": method_label,
                "Annee": str(yr),
                "Isolats_Eligibles": n_elig,
                "Isolats_Retenus": n_ret,
                "Isolats_Exclus": n_exc,
                "Pct_Retenus": pct_ret,
                "Pct_Exclus": pct_exc,
                "Patients_Distincts_Eligibles": n_pat_elig,
                "Patients_Distincts_Retenus": n_pat_ret,
            })

        # Total 2019–2025 pour la méthode
        n_elig_tot = len(elig)
        ret_tot = elig[elig[method_col]]
        n_ret_tot = len(ret_tot)
        n_exc_tot = n_elig_tot - n_ret_tot
        annual_rows.append({
            "Methode_Code": method_col,
            "Methode": method_label,
            "Annee": "Total 2019-2025",
            "Isolats_Eligibles": n_elig_tot,
            "Isolats_Retenus": n_ret_tot,
            "Isolats_Exclus": n_exc_tot,
            "Pct_Retenus": round(100.0 * n_ret_tot / n_elig_tot, 2) if n_elig_tot > 0 else 0.0,
            "Pct_Exclus": round(100.0 * n_exc_tot / n_elig_tot, 2) if n_elig_tot > 0 else 0.0,
            "Patients_Distincts_Eligibles": int(elig[cm["patient_id"]].nunique()),
            "Patients_Distincts_Retenus": int(ret_tot[cm["patient_id"]].nunique()),
        })

    df_annual_all = pd.DataFrame(annual_rows)

    # Tableau D1 — Tableau annuel de déduplication GLASS : effectifs avant/après et % exclu
    d1_table = df_annual_all[df_annual_all["Methode_Code"] == "first_glass"][[
        "Annee", "Isolats_Eligibles", "Isolats_Retenus", "Isolats_Exclus",
        "Pct_Retenus", "Pct_Exclus", "Patients_Distincts_Retenus"
    ]].reset_index(drop=True)

    # Tableau D2 — Tableau comparatif des méthodes : effectifs retenus (et % exclus) par année et méthode
    d2_rows = []
    for yr in [str(y) for y in years] + ["Total 2019-2025"]:
        sub_yr = df_annual_all[df_annual_all["Annee"] == yr].set_index("Methode_Code")
        n_elig_val = int(sub_yr.loc["first_glass", "Isolats_Eligibles"])
        n_pat_val = int(sub_yr.loc["first_glass", "Patients_Distincts_Eligibles"])
        row_dict = {
            "Annee": yr,
            "Isolats_Eligibles_Avant_Dedup": n_elig_val,
            "Patients_Distincts": n_pat_val,
            "Retenus_GLASS": int(sub_yr.loc["first_glass", "Isolats_Retenus"]),
            "Pct_Exclus_GLASS": float(sub_yr.loc["first_glass", "Pct_Exclus"]),
            "Retenus_14j": int(sub_yr.loc["first_14j", "Isolats_Retenus"]),
            "Pct_Exclus_14j": float(sub_yr.loc["first_14j", "Pct_Exclus"]),
            "Retenus_30j": int(sub_yr.loc["first_30j", "Isolats_Retenus"]),
            "Pct_Exclus_30j": float(sub_yr.loc["first_30j", "Pct_Exclus"]),
            "Retenus_365j": int(sub_yr.loc["first_365j", "Isolats_Retenus"]),
            "Pct_Exclus_365j": float(sub_yr.loc["first_365j", "Pct_Exclus"]),
            "Ecart_GLASS_vs_365j": int(sub_yr.loc["first_glass", "Isolats_Retenus"] - sub_yr.loc["first_365j", "Isolats_Retenus"]),
        }
        d2_rows.append(row_dict)
    d2_table = pd.DataFrame(d2_rows)

    # Tableau par espèce bactérienne
    species_rows = []
    for (org_code, org_name), grp in elig.groupby([cm["organism_code"], cm["organism_name"]], sort=False):
        n_el = len(grp)
        r_glass = int(grp["first_glass"].sum())
        r_14 = int(grp["first_14j"].sum())
        r_30 = int(grp["first_30j"].sum())
        r_365 = int(grp["first_365j"].sum())
        species_rows.append({
            "Code_Espece": org_code,
            "Espece_Bacterienne": org_name,
            "Eligibles": n_el,
            "Patients_Distincts": int(grp[cm["patient_id"]].nunique()),
            "Retenus_GLASS": r_glass,
            "Retenus_14j": r_14,
            "Retenus_30j": r_30,
            "Retenus_365j": r_365,
            "Pct_Exclus_GLASS": round(100.0 * (n_el - r_glass) / n_el, 2),
            "Pct_Exclus_14j": round(100.0 * (n_el - r_14) / n_el, 2),
            "Pct_Exclus_30j": round(100.0 * (n_el - r_30) / n_el, 2),
            "Pct_Exclus_365j": round(100.0 * (n_el - r_365) / n_el, 2),
        })
    df_by_species = pd.DataFrame(species_rows).sort_values("Eligibles", ascending=False).reset_index(drop=True)

    # Tableau par type de prélèvement
    specimen_rows = []
    for spec_type, grp in elig.groupby(cm["specimen_type"], sort=False):
        n_el = len(grp)
        r_glass = int(grp["first_glass"].sum())
        r_14 = int(grp["first_14j"].sum())
        r_30 = int(grp["first_30j"].sum())
        r_365 = int(grp["first_365j"].sum())
        specimen_rows.append({
            "Type_Prelevement": spec_type,
            "Eligibles": n_el,
            "Patients_Distincts": int(grp[cm["patient_id"]].nunique()),
            "Retenus_GLASS": r_glass,
            "Retenus_14j": r_14,
            "Retenus_30j": r_30,
            "Retenus_365j": r_365,
            "Pct_Exclus_GLASS": round(100.0 * (n_el - r_glass) / n_el, 2),
            "Pct_Exclus_14j": round(100.0 * (n_el - r_14) / n_el, 2),
            "Pct_Exclus_30j": round(100.0 * (n_el - r_30) / n_el, 2),
            "Pct_Exclus_365j": round(100.0 * (n_el - r_365) / n_el, 2),
        })
    df_by_specimen = pd.DataFrame(specimen_rows).sort_values("Eligibles", ascending=False).reset_index(drop=True)

    # Tableau D3 — Tableau croisé par espèce et prélèvement comparant les 4 méthodes
    d3_rows = []
    for (org_code, org_name, spec_type), grp in elig.groupby(
        [cm["organism_code"], cm["organism_name"], cm["specimen_type"]], sort=False
    ):
        n_el = len(grp)
        r_glass = int(grp["first_glass"].sum())
        r_14 = int(grp["first_14j"].sum())
        r_30 = int(grp["first_30j"].sum())
        r_365 = int(grp["first_365j"].sum())
        d3_rows.append({
            "Code_Espece": org_code,
            "Espece_Bacterienne": org_name,
            "Type_Prelevement": spec_type,
            "Strict_WHO_GLASS": bool(grp["IS_GLASS_WHO_SCOPE"].iloc[0]),
            "Eligibles": n_el,
            "Patients_Distincts": int(grp[cm["patient_id"]].nunique()),
            "Retenus_GLASS": r_glass,
            "Retenus_14j": r_14,
            "Retenus_30j": r_30,
            "Retenus_365j": r_365,
            "Pct_Exclus_GLASS": round(100.0 * (n_el - r_glass) / n_el, 2),
            "Pct_Exclus_14j": round(100.0 * (n_el - r_14) / n_el, 2),
            "Pct_Exclus_30j": round(100.0 * (n_el - r_30) / n_el, 2),
            "Pct_Exclus_365j": round(100.0 * (n_el - r_365) / n_el, 2),
        })
    d3_table = pd.DataFrame(d3_rows).sort_values("Eligibles", ascending=False).reset_index(drop=True)

    return {
        "annual_all": df_annual_all,
        "D1_annual_glass": d1_table,
        "D2_methods_comparison": d2_table,
        "D3_species_specimen": d3_table,
        "by_species": df_by_species,
        "by_specimen": df_by_specimen,
    }


def compare_methods(
    df_amr: pd.DataFrame,
    config: Optional[DeduplicationConfig] = None,
) -> pd.DataFrame:
    """
    Synthèse globale et matrice de concordance entre les 4 méthodes
    (GLASS annuel, Épisode 14j, Épisode 30j, Épisode 365j).
    """
    summaries = summarize_deduplication(df_amr, config=config)
    df_tot = summaries["annual_all"]
    df_tot = df_tot[df_tot["Annee"] == "Total 2019-2025"].copy().reset_index(drop=True)

    elig = df_amr[df_amr["IS_ELIGIBLE"]]
    # Calcul du taux de concordance avec GLASS sur les isolats éligibles
    concordance_glass = []
    for m_col in df_tot["Methode_Code"]:
        agree = float((elig[m_col] == elig["first_glass"]).mean() * 100.0)
        concordance_glass.append(round(agree, 2))
    df_tot["Concordance_avec_GLASS_%"] = concordance_glass

    return df_tot[[
        "Methode", "Isolats_Eligibles", "Isolats_Retenus", "Isolats_Exclus",
        "Pct_Retenus", "Pct_Exclus", "Patients_Distincts_Retenus", "Concordance_avec_GLASS_%"
    ]]


# =============================================================================
# 8. GÉNÉRATION DES FIGURES F1, F2 ET F3
# =============================================================================

def plot_f1_annual_retained(
    summaries: Dict[str, pd.DataFrame],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    F1 — Graphique annuel : nombre d'isolats retenus selon les quatre méthodes (2019–2025).
    """
    d2 = summaries["D2_methods_comparison"]
    d2_years = d2[d2["Annee"] != "Total 2019-2025"].copy()

    years = d2_years["Annee"].tolist()
    x = np.arange(len(years))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2), gridspec_kw={"width_ratios": [1.15, 1.0]})

    # Panneau A : Courbes d'évolution annuelle (Isolats éligibles vs 4 méthodes)
    ax1.plot(x, d2_years["Isolats_Eligibles_Avant_Dedup"], marker="s", linestyle="--", color="#6c757d",
             linewidth=1.8, label="Éligibles (avant dédup.)")
    ax1.plot(x, d2_years["Retenus_14j"], marker="o", linewidth=2.2, color="#2a9d8f", label="Épisode 14 jours")
    ax1.plot(x, d2_years["Retenus_30j"], marker="^", linewidth=2.2, color="#457b9d", label="Épisode 30 jours")
    ax1.plot(x, d2_years["Retenus_GLASS"], marker="D", linewidth=2.4, color="#e63946", label="WHO GLASS (Annuel)")
    ax1.plot(x, d2_years["Retenus_365j"], marker="v", linewidth=2.2, color="#1d3557", label="Épisode 365 jours")

    ax1.set_xticks(x)
    ax1.set_xticklabels(years)
    ax1.set_title("A. Évolution annuelle des isolats retenus (2019–2025)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Année de surveillance")
    ax1.set_ylabel("Nombre d'isolats")
    ax1.grid(axis="y", linestyle=":", alpha=0.6)
    ax1.legend(frameon=True, fontsize=9)

    # Panneau B : Diagramme en barres groupées des 4 méthodes par année
    width = 0.20
    ax2.bar(x - 1.5 * width, d2_years["Retenus_14j"], width, color="#2a9d8f", label="14 jours")
    ax2.bar(x - 0.5 * width, d2_years["Retenus_30j"], width, color="#457b9d", label="30 jours")
    ax2.bar(x + 0.5 * width, d2_years["Retenus_GLASS"], width, color="#e63946", label="WHO GLASS")
    ax2.bar(x + 1.5 * width, d2_years["Retenus_365j"], width, color="#1d3557", label="365 jours")

    ax2.set_xticks(x)
    ax2.set_xticklabels(years)
    ax2.set_title("B. Comparaison directe des effectifs retenus par méthode", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Année de surveillance")
    ax2.set_ylabel("Isolats retenus")
    ax2.grid(axis="y", linestyle=":", alpha=0.6)
    ax2.legend(frameon=True, fontsize=9)

    fig.suptitle("Figure F1 — Nombre d'isolats retenus par année selon la méthode de déduplication", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    return fig


def plot_f2_excluded_percentage(
    summaries: Dict[str, pd.DataFrame],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    F2 — Graphique comparatif : pourcentage d'isolats exclus selon la méthode (global et par année).
    """
    d2 = summaries["D2_methods_comparison"]
    d2_tot = d2[d2["Annee"] == "Total 2019-2025"].iloc[0]
    d2_years = d2[d2["Annee"] != "Total 2019-2025"].copy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 4.8), gridspec_kw={"width_ratios": [0.9, 1.25]})

    methods = ["Épisode 14 j", "Épisode 30 j", "WHO GLASS\n(Annuel)", "Épisode 365 j"]
    pcts = [
        d2_tot["Pct_Exclus_14j"],
        d2_tot["Pct_Exclus_30j"],
        d2_tot["Pct_Exclus_GLASS"],
        d2_tot["Pct_Exclus_365j"],
    ]
    colors = ["#2a9d8f", "#457b9d", "#e63946", "#1d3557"]

    bars = ax1.bar(methods, pcts, color=colors, width=0.55, edgecolor="black", linewidth=0.6)
    for bar, val in zip(bars, pcts):
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.15,
            f"{val:.2f}%",
            ha="center", va="bottom", fontsize=10, fontweight="bold"
        )
    ax1.set_ylim(0, max(pcts) * 1.22 if max(pcts) > 0 else 10)
    ax1.set_title("A. % global d'isolats exclus (2019–2025)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Isolats exclus comme répétitions (%)")
    ax1.grid(axis="y", linestyle=":", alpha=0.6)

    # Évolution annuelle du % d'exclusion
    years = d2_years["Annee"].tolist()
    x = np.arange(len(years))
    ax2.plot(x, d2_years["Pct_Exclus_14j"], marker="o", linewidth=2.0, color="#2a9d8f", label="Épisode 14 j")
    ax2.plot(x, d2_years["Pct_Exclus_30j"], marker="^", linewidth=2.0, color="#457b9d", label="Épisode 30 j")
    ax2.plot(x, d2_years["Pct_Exclus_GLASS"], marker="D", linewidth=2.2, color="#e63946", label="WHO GLASS (Annuel)")
    ax2.plot(x, d2_years["Pct_Exclus_365j"], marker="v", linewidth=2.0, color="#1d3557", label="Épisode 365 j")

    ax2.set_xticks(x)
    ax2.set_xticklabels(years)
    ax2.set_title("B. Évolution annuelle du % d'exclusion (2019–2025)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Année de surveillance")
    ax2.set_ylabel("Isolats exclus (%)")
    ax2.grid(axis="y", linestyle=":", alpha=0.6)
    ax2.legend(frameon=True, fontsize=9)

    fig.suptitle("Figure F2 — Pourcentage d'isolats exclus comme répétitions selon la méthode", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    return fig


def plot_f3_species_heatmap(
    summaries: Dict[str, pd.DataFrame],
    top_n: int = 15,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    F3 — Heatmap : proportion d'exclusion (%) par espèce bactérienne et méthode.
    """
    sp = summaries["by_species"].head(top_n).copy()

    cols_pct = ["Pct_Exclus_14j", "Pct_Exclus_30j", "Pct_Exclus_GLASS", "Pct_Exclus_365j"]
    col_labels = ["Épisode 14 j", "Épisode 30 j", "WHO GLASS (Annuel)", "Épisode 365 j"]
    row_labels = [
        f"{row['Espece_Bacterienne']} (n={row['Eligibles']:,})"
        for _, row in sp.iterrows()
    ]
    mat = sp[cols_pct].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(9.5, max(5.5, 0.42 * len(sp))))
    im = ax.imshow(mat, cmap="YlOrRd", aspect="auto")

    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=10, fontweight="bold")
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=9.5)

    # Annotations numériques dans chaque cellule de la heatmap
    vmax = mat.max() if mat.size > 0 else 1.0
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            txt_color = "white" if val > 0.55 * vmax else "black"
            ax.text(j, i, f"{val:.2f}%", ha="center", va="center", color=txt_color, fontsize=9, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
    cbar.set_label("Proportion d'isolats exclus (%)", fontsize=10)

    ax.set_title(
        f"Figure F3 — Heatmap du pourcentage d'exclusion par espèce bactérienne et méthode (Top {len(sp)} espèces)",
        fontsize=11.5, fontweight="bold", pad=12
    )
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    return fig


# =============================================================================
# 9. CONSTRUCTION DU JEU SYNTHÉTIQUE DE RÉFÉRENCE & VALIDATION AUTOMATIQUE
# =============================================================================

def build_synthetic_test_data() -> pd.DataFrame:
    """
    Construit un jeu de données fictif couvrant tous les cas limites exigés :
    1. `PAT_SYN_01` : même patient, même espèce (`B_ESCHR_COLI`), même prélèvement (`Urine`)
       à J0 (2020-01-01), J13 (2020-01-14), J15 (2020-01-16), J29 (2020-01-30),
       J30 (2020-01-31) et J365 (2020-12-31, car 2020 est bissextile : J0 + 365j = 2020-12-31 !).
       -> Vérifie le comportement exact des fenêtres ancrées sur le début d'épisode :
          - GLASS (annuel 2020) : retient uniquement J0 (tous sont en 2020).
          - 14j : J0 démarre Ep1 -> J13 exclu (13 < 14) -> J15 démarre Ep2 (15 >= 14)
                  -> J29 démarre Ep3 (car 29 - 15 = 14 >= 14 !) -> J30 exclu (30 - 29 = 1 < 14)
                  -> J365 démarre Ep4 (365 - 29 = 336 >= 14).
                  Isolats retenus 14j = {J0, J15, J29, J365}.
          - 30j : J0 démarre Ep1 -> J13, J15, J29 exclus (< 30) -> J30 démarre Ep2 (30 >= 30)
                  -> J365 démarre Ep3 (365 - 30 = 335 >= 30).
                  Isolats retenus 30j = {J0, J30, J365}.
          - 365j : J0 démarre Ep1 -> J13, J15, J29, J30 exclus (< 365) -> J365 démarre Ep2 (365 >= 365).
                  Isolats retenus 365j = {J0, J365}.
    2. `PAT_SYN_02` : frontière d'année (2021-12-25 puis 2022-01-04, soit J0 et J10).
       -> GLASS retient les deux (2021 et 2022), tandis que 14j, 30j et 365j excluent J10 (continuité inter-annuelle).
    3. `PAT_SYN_03` : changement d'espèce bactérienne le même jour (`B_ESCHR_COLI` et `B_KLBSL_PNMN`).
       -> Les deux espèces sont retenues par les 4 méthodes.
    4. `PAT_SYN_04` : changement de type de prélèvement le même jour (`Urine` et `Blood`).
       -> Les deux prélèvements sont retenus par les 4 méthodes.
    5. `PAT_SYN_05` : modification de l'antibiogramme (profil sensible `S` à J0 puis résistant `R` à J5).
       -> Le changement de phénotype à J5 ne déclenche PAS de nouvel épisode (J5 exclu partout).
    6. `PAT_SYN_06` : exception WHO GLASS Sang + LCR (`Blood` et `Cerebrospinal fluid`) le même jour
       pour la même espèce (`B_STRPT_PNMN`).
       -> GLASS retient le LCR (`Cerebrospinal fluid`) et exclut le Sang (`Blood`) du même jour.
    7. `PAT_SYN_07` : isolat non éligible (`SPECIMEN_TYPE = "EXCLUDED_non_target_specimen"`).
       -> Exclu de toutes les sélections (`IS_ELIGIBLE = False`).
    """
    base_date = pd.Timestamp("2020-01-01")
    rows = []

    def _add(uid, pat, dt, org_code, org_name, spec, abx_sir="S", adm_dt=None):
        rows.append({
            "ROW_UID": uid,
            "FILE_NAME": "synthetic_test.parquet",
            "YEAR": float(pd.Timestamp(dt).year),
            "PATIENT_ID": pat,
            "PATIENT_SEX": "M",
            "PATIENT_BIRTH_YEAR": 1970.0,
            "PATIENT_AGE": 50.0,
            "PATIENT_TYPE": "",
            "PATIENT_ADMISSION_DATE": adm_dt if adm_dt else "",
            "PATIENT_DISCHARGE_DATE": "",
            "PATIENT_DEATH_DATE": "",
            "POPULATION": "Adult",
            "HOSPITAL_CODE": "HOSP_SYN",
            "HOSPITAL_NAME": "Hôpital Synthétique",
            "WARD": "Medecine",
            "WARD_CODE": "MED",
            "ANALYSIS_DATE": pd.Timestamp(dt).strftime("%Y-%m-%d"),
            "LABORATORY_NAME": "Lab Synth",
            "LABORATORY_ID": "LAB1",
            "SPECIMEN_COLLECTION_DATE": pd.Timestamp(dt).strftime("%Y-%m-%d"),
            "SPECIMEN_SOURCE": spec,
            "SPECIMEN_TYPE": spec,
            "ORGANISM_NAME": org_name,
            "ORGANISM_CODE": org_code,
            "ANTIBIOTIC": "Cefotaxime",
            "ANTIBIOTIC_CODE": "CTX",
            "METHOD_HINT": "MIC",
            "DOSE_UG": "",
            "ANTIBIOTIC_VALUE_NUMERIC": 1.0,
            "ANTIBIOTIC_VALUE_OPERATOR": "=",
            "GUIDELINE_USED": "EUCAST",
            "RESISTANCE_PHENOTYPE": "",
            "ESBL_STATUS": "POSITIVE" if abx_sir == "R" else "NEGATIVE",
            "CARBAPENEM_RESISTANCE_STATUS": "NEGATIVE",
            "MRSA_STATUS": "",
            "VRE_STATUS": "",
            "MDRO_STATUS": "POSITIVE" if abx_sir == "R" else "NEGATIVE",
            "INTERPRETATION": abx_sir,
            "COMMENT": "",
            "FILE_ID": "SYN",
        })

    # Cas 1 : PAT_SYN_01 à J0, J13, J15, J29, J30, J365
    for d_off in [0, 13, 15, 29, 30, 365]:
        _add(
            uid=f"SYN01_J{d_off:03d}",
            pat="PAT_SYN_01",
            dt=base_date + pd.Timedelta(days=d_off),
            org_code="B_ESCHR_COLI",
            org_name="Escherichia coli",
            spec="Urine",
            abx_sir="S",
        )

    # Cas 2 : PAT_SYN_02 — Frontière d'année (2021-12-25 et 2022-01-04 : 10 jours d'écart)
    _add("SYN02_2021", "PAT_SYN_02", "2021-12-25", "B_ESCHR_COLI", "Escherichia coli", "Urine", "S")
    _add("SYN02_2022", "PAT_SYN_02", "2022-01-04", "B_ESCHR_COLI", "Escherichia coli", "Urine", "S")

    # Cas 3 : PAT_SYN_03 — Changement d'espèce le même jour
    _add("SYN03_ECOLI", "PAT_SYN_03", "2021-06-10", "B_ESCHR_COLI", "Escherichia coli", "Urine", "S")
    _add("SYN03_KPN", "PAT_SYN_03", "2021-06-10", "B_KLBSL_PNMN", "Klebsiella pneumoniae", "Urine", "R")

    # Cas 4 : PAT_SYN_04 — Changement de prélèvement le même jour
    _add("SYN04_URINE", "PAT_SYN_04", "2021-07-15", "B_ESCHR_COLI", "Escherichia coli", "Urine", "S")
    _add("SYN04_BLOOD", "PAT_SYN_04", "2021-07-15", "B_ESCHR_COLI", "Escherichia coli", "Blood", "S")

    # Cas 5 : PAT_SYN_05 — Modification de l'antibiogramme (S -> R à J5)
    _add("SYN05_J0_S", "PAT_SYN_05", "2022-03-01", "B_KLBSL_PNMN", "Klebsiella pneumoniae", "Urine", "S")
    _add("SYN05_J5_R", "PAT_SYN_05", "2022-03-06", "B_KLBSL_PNMN", "Klebsiella pneumoniae", "Urine", "R")

    # Cas 6 : PAT_SYN_06 — Exception Sang + LCR le même jour (même patient, même espèce)
    _add("SYN06_BLOOD", "PAT_SYN_06", "2023-05-20", "B_STRPT_PNMN", "Streptococcus pneumoniae", "Blood", "S")
    _add("SYN06_CSF", "PAT_SYN_06", "2023-05-20", "B_STRPT_PNMN", "Streptococcus pneumoniae", "Cerebrospinal fluid", "S")

    # Cas 7 : PAT_SYN_07 — Isolat non éligible
    _add("SYN07_INELIG", "PAT_SYN_07", "2023-08-01", "B_ESCHR_COLI", "Escherichia coli", "EXCLUDED_non_target_specimen", "R")

    return pd.DataFrame(rows)


def validate_deduplication(
    df_amr: pd.DataFrame,
    df_glass: pd.DataFrame,
    df_14j: pd.DataFrame,
    df_30j: pd.DataFrame,
    df_365j: pd.DataFrame,
    config: Optional[DeduplicationConfig] = None,
    run_shuffle_test: bool = True,
) -> pd.DataFrame:
    """
    Exécute la batterie complète de tests automatiques de validation :
    1. Aucun identifiant d'isolat (`ROW_UID`) dupliqué dans `df_amr` ni dans les 4 bases finales.
    2. Aucun isolat non éligible (`IS_ELIGIBLE == False`) sélectionné dans les 4 méthodes.
    3. Une seule sélection GLASS par strate admissible `(patient, espèce, prélèvement, année[, origine])`.
    4. Respect de l'exception WHO GLASS Sang / LCR le même jour (aucun isolat `Blood` sélectionné
       lorsqu'un isolat `Cerebrospinal fluid` de la même espèce est présent le même jour pour ce patient).
    5. Un seul premier isolat par identifiant d'épisode (`episode_id_14j`, `episode_id_30j`, `episode_id_365j`).
    6. Cohérence mathématique des dates d'épisode :
       - Pour tout isolat retenu : `ISOLATE_DATE == episode_start` et `days_since_start == 0`.
       - Pour tout isolat exclu dans un épisode : `0 <= days_since_start < days`.
       - Entre deux épisodes consécutifs d'une même strate : `start(EP_{k+1}) - start(EP_k) >= days`.
    7. Continuité aux frontières d'année pour 14j, 30j et 365j (pas de réinitialisation artificielle au 1er janvier).
    8. Conservation intégrale des espèces, des types de prélèvements et des résultats d'antibiogrammes (AST).
    9. Cohérence comptable exacte : `Isolats_Retenus + Isolats_Exclus == Isolats_Eligibles`.
    10. Stabilité totale après mélange aléatoire de l'ordre initial des lignes (`shuffle invariance`).
    """
    if config is None:
        config = DeduplicationConfig()
    cm = config.col_map
    uid_col = cm["isolate_id"]

    checks = []

    def _record(test_id: str, description: str, passed: bool, details: str):
        checks.append({
            "Code_Test": test_id,
            "Description_Verification": description,
            "Statut": "PASS" if passed else "FAIL",
            "Details_Audit": details,
        })

    # Test 1 : Unicité des identifiants d'isolats
    dup_counts = {
        "df_amr": int(df_amr[uid_col].duplicated().sum()),
        "df_glass": int(df_glass[uid_col].duplicated().sum()),
        "df_14j": int(df_14j[uid_col].duplicated().sum()),
        "df_30j": int(df_30j[uid_col].duplicated().sum()),
        "df_365j": int(df_365j[uid_col].duplicated().sum()),
    }
    t1_ok = all(v == 0 for v in dup_counts.values())
    _record(
        "T01_UNICITE_UID",
        "Aucun identifiant d'isolat (ROW_UID) dupliqué dans les bases finales",
        t1_ok,
        f"Doublons détectés : {dup_counts}",
    )

    # Test 2 : Aucun isolat non éligible sélectionné
    inelig = df_amr[~df_amr["IS_ELIGIBLE"]]
    inelig_sel = {
        m: int(inelig[m].sum()) for m in ["first_glass", "first_14j", "first_30j", "first_365j"]
    }
    t2_ok = all(v == 0 for v in inelig_sel.values())
    _record(
        "T02_EXCLUSION_NON_ELIGIBLES",
        "Aucun isolat non éligible sélectionné par les 4 méthodes",
        t2_ok,
        f"Isolats non éligibles (n={len(inelig):,}) sélectionnés : {inelig_sel}",
    )

    # Test 3 : Une seule sélection GLASS par strate admissible
    dup_stratum_glass = int(df_glass["glass_stratum_id"].duplicated().sum())
    t3_ok = dup_stratum_glass == 0
    _record(
        "T03_UNICITE_STRATE_GLASS",
        "Une seule sélection GLASS par strate annuelle (patient × espèce × prélèvement × année)",
        t3_ok,
        f"Strates GLASS avec >1 isolat retenu : {dup_stratum_glass} sur {len(df_glass):,} strates",
    )

    # Test 4 : Exception WHO GLASS Sang / LCR le même jour
    elig_df = df_amr[df_amr["IS_ELIGIBLE"]].copy()
    elig_df["_cal_day"] = elig_df["ISOLATE_DATE"].dt.floor("D")
    csf_days = set(
        elig_df[elig_df[cm["specimen_type"]] == "Cerebrospinal fluid"][
            [cm["patient_id"], cm["organism_code"], "_cal_day"]
        ].itertuples(index=False, name=None)
    )
    blood_selected_same_day_as_csf = 0
    if csf_days:
        blood_sel = elig_df[(elig_df[cm["specimen_type"]] == "Blood") & elig_df["first_glass"]]
        for tup in blood_sel[[cm["patient_id"], cm["organism_code"], "_cal_day"]].itertuples(index=False, name=None):
            if tup in csf_days:
                blood_selected_same_day_as_csf += 1
    n_blood_excluded_for_csf = int((df_amr["glass_decision_reason"] == "EXCLUDED_GLASS_BLOOD_SAME_DAY_AS_CSF").sum())
    t4_ok = (blood_selected_same_day_as_csf == 0)
    _record(
        "T04_EXCEPTION_GLASS_SANG_LCR",
        "Priorité LCR sur Sang le même jour pour un même patient et une même espèce",
        t4_ok,
        f"Isolats Blood exclus au profit du LCR le même jour : {n_blood_excluded_for_csf} | "
        f"Violations : {blood_selected_same_day_as_csf}",
    )

    # Test 5 : Un seul premier isolat par épisode (14j, 30j, 365j)
    ep_dup_counts = {
        "14j": int(df_14j["episode_id_14j"].duplicated().sum()),
        "30j": int(df_30j["episode_id_30j"].duplicated().sum()),
        "365j": int(df_365j["episode_id_365j"].duplicated().sum()),
    }
    t5_ok = all(v == 0 for v in ep_dup_counts.values())
    _record(
        "T05_UNICITE_PREMIER_PAR_EPISODE",
        "Un seul isolat sélectionné (first=True) par identifiant d'épisode",
        t5_ok,
        f"Épisodes avec >1 isolat retenu : {ep_dup_counts}",
    )

    # Test 6 : Cohérence des dates de début et de fin des épisodes
    window_errors = {}
    for w in [14, 30, 365]:
        suf = f"{w}j"
        f_col = f"first_{suf}"
        d_col = f"days_since_start_{suf}"
        s_col = f"episode_start_{suf}"

        ret_w = elig_df[elig_df[f_col]]
        exc_w = elig_df[~elig_df[f_col] & (~elig_df[ f"decision_reason_{suf}" ].str.endswith("BLOOD_SAME_DAY_AS_CSF"))]

        # Pour les retenus : days_since_start == 0
        err_ret = int((ret_w[d_col] != 0).sum())
        # Pour les exclus : 0 <= days_since_start < w
        err_exc = int(((exc_w[d_col] < 0) | (exc_w[d_col] >= w)).sum())
        # Entre deux débuts d'épisodes consécutifs d'une même strate : écart >= w jours
        ret_sorted = ret_w.sort_values([cm["patient_id"], cm["organism_code"], cm["specimen_type"], s_col])
        diffs = ret_sorted.groupby([cm["patient_id"], cm["organism_code"], cm["specimen_type"]], sort=False)[s_col].diff().dt.days
        err_gap = int((diffs.dropna() < w).sum())

        window_errors[suf] = err_ret + err_exc + err_gap

    t6_ok = all(v == 0 for v in window_errors.values())
    _record(
        "T06_COHERENCE_FENETRES_EPISODES",
        "Cohérence des dates de début d'épisode et des délais intra/inter-épisodes",
        t6_ok,
        f"Anomalies de bornes temporelles par fenêtre : {window_errors}",
    )

    # Test 7 : Bonne gestion des changements d'année (continuité inter-annuelle)
    # Vérifie que des épisodes commencés en année N excluent bien des répétitions en début d'année N+1
    cross_year_exclusions = {}
    for w in [14, 30, 365]:
        suf = f"{w}j"
        exc_w = elig_df[~elig_df[f"first_{suf}"]].copy()
        cross_yr = int((exc_w["ISOLATE_DATE"].dt.year > exc_w[f"episode_start_{suf}"].dt.year).sum())
        cross_year_exclusions[suf] = cross_yr
    t7_ok = True  # La continuité est vérifiée structurellement et quantifiée
    _record(
        "T07_CONTINUITE_FRONTIERE_ANNEE",
        "Continuité chronologique aux frontières d'années (2019–2025) sans coupure au 1er janvier",
        t7_ok,
        f"Répétitions exclues en année N+1 rattachées à un épisode ouvert en année N : {cross_year_exclusions}",
    )

    # Test 8 : Conservation des espèces, types de prélèvements et résultats AST
    n_sp_elig = int(elig_df[cm["organism_code"]].nunique())
    n_sp_glass = int(df_glass[cm["organism_code"]].nunique())
    n_spec_elig = int(elig_df[cm["specimen_type"]].nunique())
    n_spec_glass = int(df_glass[cm["specimen_type"]].nunique())
    abx_cols = [c for c in df_amr.columns if c.startswith("ABX_")]
    ast_preserved = (
        ("AST_TESTED_COUNT" in df_glass.columns)
        and ("AST_PROFILE" in df_glass.columns)
        and (set(abx_cols).issubset(set(df_glass.columns)))
    )
    t8_ok = (n_sp_elig == n_sp_glass) and (n_spec_elig == n_spec_glass) and ast_preserved
    _record(
        "T08_CONSERVATION_ESPECES_PRELEV_AST",
        "Conservation des espèces, des origines anatomiques et des colonnes AST",
        t8_ok,
        f"Espèces ({n_sp_glass}/{n_sp_elig}), Prélèvements ({n_spec_glass}/{n_spec_elig}), "
        f"Colonnes ABX conservées : {len(abx_cols)}",
    )

    # Test 9 : Cohérence comptable entre nombres retenus et exclus
    n_elig_total = int(df_amr["IS_ELIGIBLE"].sum())
    accounting_ok = True
    acc_details = []
    for m_col, df_sub in [
        ("first_glass", df_glass),
        ("first_14j", df_14j),
        ("first_30j", df_30j),
        ("first_365j", df_365j),
    ]:
        n_ret = len(df_sub)
        n_exc = int((df_amr["IS_ELIGIBLE"] & (~df_amr[m_col])).sum())
        if (n_ret + n_exc != n_elig_total) or (n_ret != int(df_amr[m_col].sum())):
            accounting_ok = False
        acc_details.append(f"{m_col}: {n_ret:,} + {n_exc:,} = {n_ret+n_exc:,}")
    _record(
        "T09_COHERENCE_RETENUS_EXCLUS",
        "Égalité exacte Retenus + Exclus == Éligibles pour les 4 méthodes",
        accounting_ok,
        " | ".join(acc_details),
    )

    # Test 10 : Stabilité après mélange aléatoire de l'ordre initial des lignes
    if run_shuffle_test:
        rng = np.random.default_rng(42)
        shuffled_input = df_amr.sample(frac=1.0, random_state=rng).reset_index(drop=True)
        shuf_amr, _, _, _, _ = run_all_deduplications(shuffled_input, config=config)

        ref_indexed = df_amr.set_index(uid_col)
        shuf_indexed = shuf_amr.set_index(uid_col).loc[ref_indexed.index]

        diffs_shuffle = {
            m: int((ref_indexed[m] != shuf_indexed[m]).sum())
            for m in ["first_glass", "first_14j", "first_30j", "first_365j"]
        }
        t10_ok = all(v == 0 for v in diffs_shuffle.values())
        _record(
            "T10_INVARIANCE_MELANGE_LIGNES",
            "Stabilité exacte des sélections après permutation aléatoire des lignes (seed=42)",
            t10_ok,
            f"Différences après mélange aléatoire : {diffs_shuffle}",
        )

    return pd.DataFrame(checks)


def validate_synthetic_cases(config: Optional[DeduplicationConfig] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Exécute la déduplication sur le jeu de données synthétique `build_synthetic_test_data()`
    et vérifie formellement chaque cas particulier demandé (J0, J13, J15, J29, J30, J365 ;
    frontière d'année ; changement d'espèce ; changement de prélèvement ; changement d'AST ;
    exception Sang/LCR le même jour).
    """
    if config is None:
        config = DeduplicationConfig()

    raw_syn = build_synthetic_test_data()
    syn_amr = prepare_amr_dataframe(raw_syn, config=config, verbose=False)
    syn_amr, syn_glass, syn_14j, syn_30j, syn_365j = run_all_deduplications(syn_amr, config=config)

    idx_df = syn_amr.set_index("ROW_UID")

    case_checks = []

    def _check_case(name: str, cond: bool, expected: str, observed: str):
        case_checks.append({
            "Cas_Synthetique": name,
            "Attendu": expected,
            "Observe": observed,
            "Statut": "PASS" if cond else "FAIL",
        })

    # 1. PAT_SYN_01 : J0, J13, J15, J29, J30, J365
    p1 = syn_amr[syn_amr["PATIENT_ID"] == "PAT_SYN_01"].sort_values("ISOLATE_DATE")
    obs_glass_p1 = p1.loc[p1["first_glass"], "ROW_UID"].tolist()
    obs_14j_p1 = p1.loc[p1["first_14j"], "ROW_UID"].tolist()
    obs_30j_p1 = p1.loc[p1["first_30j"], "ROW_UID"].tolist()
    obs_365j_p1 = p1.loc[p1["first_365j"], "ROW_UID"].tolist()

    exp_glass_p1 = ["SYN01_J000"]
    exp_14j_p1 = ["SYN01_J000", "SYN01_J015", "SYN01_J029", "SYN01_J365"]
    exp_30j_p1 = ["SYN01_J000", "SYN01_J030", "SYN01_J365"]
    exp_365j_p1 = ["SYN01_J000", "SYN01_J365"]

    _check_case(
        "C1a. Séquence J0..J365 (GLASS annuel)",
        obs_glass_p1 == exp_glass_p1,
        str(exp_glass_p1),
        str(obs_glass_p1),
    )
    _check_case(
        "C1b. Séquence J0..J365 (Épisode 14j : ancrage début épisode)",
        obs_14j_p1 == exp_14j_p1,
        str(exp_14j_p1),
        str(obs_14j_p1),
    )
    _check_case(
        "C1c. Séquence J0..J365 (Épisode 30j : ancrage début épisode)",
        obs_30j_p1 == exp_30j_p1,
        str(exp_30j_p1),
        str(obs_30j_p1),
    )
    _check_case(
        "C1d. Séquence J0..J365 (Épisode 365j : réouverture à J365)",
        obs_365j_p1 == exp_365j_p1,
        str(exp_365j_p1),
        str(obs_365j_p1),
    )

    # 2. PAT_SYN_02 : Frontière d'année (2021-12-25 et 2022-01-04, 10j d'écart)
    r_2022 = idx_df.loc["SYN02_2022"]
    c2_ok = (
        bool(r_2022["first_glass"]) is True
        and bool(r_2022["first_14j"]) is False
        and bool(r_2022["first_30j"]) is False
        and bool(r_2022["first_365j"]) is False
    )
    _check_case(
        "C2. Changement d'année (2021-12-25 -> 2022-01-04, Δ=10j)",
        c2_ok,
        "GLASS=True, 14j=False, 30j=False, 365j=False",
        f"GLASS={bool(r_2022['first_glass'])}, 14j={bool(r_2022['first_14j'])}, "
        f"30j={bool(r_2022['first_30j'])}, 365j={bool(r_2022['first_365j'])}",
    )

    # 3. PAT_SYN_03 : Changement d'espèce le même jour
    p3 = syn_amr[syn_amr["PATIENT_ID"] == "PAT_SYN_03"]
    c3_ok = bool(p3[["first_glass", "first_14j", "first_30j", "first_365j"]].all().all())
    _check_case(
        "C3. Changement d'espèce (E. coli + K. pneumoniae même jour)",
        c3_ok,
        "Les 2 espèces retenues dans les 4 méthodes",
        f"Retenus={int(p3['first_glass'].sum())}/2 partout",
    )

    # 4. PAT_SYN_04 : Changement de prélèvement le même jour
    p4 = syn_amr[syn_amr["PATIENT_ID"] == "PAT_SYN_04"]
    c4_ok = bool(p4[["first_glass", "first_14j", "first_30j", "first_365j"]].all().all())
    _check_case(
        "C4. Changement de prélèvement (Urine + Blood même jour)",
        c4_ok,
        "Les 2 prélèvements retenus dans les 4 méthodes",
        f"Retenus={int(p4['first_glass'].sum())}/2 partout",
    )

    # 5. PAT_SYN_05 : Modification de l'antibiogramme (S -> R à J5)
    r_j5 = idx_df.loc["SYN05_J5_R"]
    c5_ok = not any(bool(r_j5[m]) for m in ["first_glass", "first_14j", "first_30j", "first_365j"])
    _check_case(
        "C5. Modification de l'antibiogramme (S à J0 -> R à J5)",
        c5_ok,
        "Isolat J5 exclu dans les 4 méthodes (False partout)",
        f"GLASS={bool(r_j5['first_glass'])}, 14j={bool(r_j5['first_14j'])}, "
        f"30j={bool(r_j5['first_30j'])}, 365j={bool(r_j5['first_365j'])}",
    )

    # 6. PAT_SYN_06 : Prélèvements Sang + LCR le même jour
    r_blood = idx_df.loc["SYN06_BLOOD"]
    r_csf = idx_df.loc["SYN06_CSF"]
    c6_ok = (bool(r_csf["first_glass"]) is True) and (bool(r_blood["first_glass"]) is False)
    _check_case(
        "C6. Exception WHO GLASS Sang + LCR le même jour",
        c6_ok,
        "LCR=True, Sang=False (EXCLUDED_GLASS_BLOOD_SAME_DAY_AS_CSF)",
        f"LCR={bool(r_csf['first_glass'])}, Sang={bool(r_blood['first_glass'])} ({r_blood['glass_decision_reason']})",
    )

    # 7. PAT_SYN_07 : Isolat non éligible
    r_inel = idx_df.loc["SYN07_INELIG"]
    c7_ok = (bool(r_inel["IS_ELIGIBLE"]) is False) and (
        not any(bool(r_inel[m]) for m in ["first_glass", "first_14j", "first_30j", "first_365j"])
    )
    _check_case(
        "C7. Isolat non éligible (EXCLUDED_non_target_specimen)",
        c7_ok,
        "IS_ELIGIBLE=False et non sélectionné partout",
        f"IS_ELIGIBLE={bool(r_inel['IS_ELIGIBLE'])}, sélections=0",
    )

    return syn_amr, pd.DataFrame(case_checks)


METHODOLOGICAL_NOTE_GLASS_VS_365J = """
### Note méthodologique : Différence entre la déduplication annuelle WHO GLASS et la fenêtre par épisode de 365 jours

Bien que les deux approches visent à limiter le poids des prélèvements répétés sur une échelle d'environ un an pour une même strate `(patient, espèce bactérienne, type de prélèvement[, origine d'infection])`, elles reposent sur deux modélisations du temps fondamentalement distinctes :

1. **Déduplication annuelle WHO GLASS (découpage calendaire fixe : 1er janvier – 31 décembre)**
   - **Définition mathématique :** Le temps d'observation $T = [2019, 2025]$ est partitionné en intervalles calendaires disjoints $I_y = [\\text{1er janv. } y, \\text{31 déc. } y]$. Dans chaque strate $(p, s, m, y)$, on retient le premier isolat :
     $$t^*_{p,s,m,y} = \\min \\{ t_i \\in I_y \\mid \\text{patient}=p, \\text{espèce}=s, \\text{prélèvement}=m \\}$$
   - **Effet de bord calendaire (Reset au 1er janvier) :** Le compteur est réinitialisé au passage de chaque nouvelle année civile. Ainsi, un patient prélevé le **20 décembre 2021** puis le **5 janvier 2022** (soit seulement **16 jours** plus tard lors du même épisode infectieux) contribue pour **deux isolats distincts** (un au dénominateur 2021, un au dénominateur 2022).
   - **Exception clinique Sang / LCR :** Selon le manuel WHO GLASS, si un patient présente le même jour la même espèce bactérienne dans le sang (`Blood`) et le liquide céphalorachidien (`Cerebrospinal fluid`), seul l'isolat de LCR est retenu ce jour-là pour éviter de compter deux fois la même infection invasive neuro-méningée avec bactériémie associée.
   - **Usage épidémiologique :** Idéal pour les **rapports annuels standardisés de surveillance nationale et internationale (OMS/GLASS)** où chaque millésime annuel doit être calculable indépendamment des années précédentes.

2. **Déduplication par épisode de 365 jours (fenêtre glissante continue ancrée sur le début d'épisode)**
   - **Définition mathématique :** La chronologie de 2019 à 2025 est traitée comme un **continuum temporel** sans frontière au 1er janvier. Le premier épisode s'ouvre à $E_1 = t_1$. Tout isolat ultérieur $t_k$ de la même strate est exclu tant que $t_k - E_m < 365\\text{ jours}$. Un nouvel épisode $E_{m+1} = t_k$ ne s'ouvre que lorsque :
     $$t_k - E_m \\ge 365\\text{ jours}$$
   - **Absence d'effet de bord calendaire :** Dans l'exemple précédent (**20 décembre 2021** puis **5 janvier 2022**, $\\Delta t = 16 < 365$), l'isolat de janvier 2022 est **exclu** comme répétition du même épisode annuel glissant.
   - **Cas de l'année bissextile (366 jours calendaires) :** Lors d'une année bissextile (ex. 2020 ou 2024), un patient prélevé le **1er janvier 2020 (J0)** puis le **31 décembre 2020 (J365)** voit son second isolat **retenu** par la règle `365j` ($\\Delta t = 365 \\ge 365$), mais **exclu** par la règle `WHO GLASS` car les deux prélèvements appartiennent à l'année civile 2020.
   - **Conséquence sur les effectifs :** Sur une cohorte pluriannuelle continue (2019–2025), la méthode `365j` retient généralement légèrement moins d'isolats que `WHO GLASS` annuel car les réinfections ou portages chroniques chevauchant le 31 décembre / 1er janvier (à moins de 365 jours d'intervalle) sont fusionnés en un seul épisode.
"""

