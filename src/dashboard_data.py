"""Chargement des données, analyse des résistances et préparation des séries temporelles AMR."""

from __future__ import annotations

from collections import Counter, defaultdict
import math
from pathlib import Path
from typing import Any

import pandas as pd

from forecasting import compare_forecasting_methods, forecast_time_series
from trend_analysis import (
    MIN_TREND_MONTHS,
    analyze_all_national_trends,
    build_national_monthly_sir_cube,
    estimate_national_monthly_trend,
    get_national_monthly_sir_series,
)

MIN_MONTHLY_TESTS = 3
ALL_BACTERIA = "__ALL_BACTERIA__"
ALL_ANTIBIOTICS = "__ALL_ANTIBIOTICS__"
MISSING_LABEL = "Non renseigné"
ANTIBIOTIC_CSV_SUFFIX = "_final_SIR_interpretation"

ROW_UID_COLUMN = "row_uid"
FILE_ID_COLUMN = "file_id"
FILE_NAME_COLUMN = "file_name"
YEAR_COLUMN = "year"
HOSPITAL_COLUMN = "hospital_name"
PATIENT_COLUMN = "patient_id"
SPECIMEN_COLUMN = "specimen_type"
WARD_COLUMN = "ward"
WARD_CODE_COLUMN = "ward_code"
DATE_COLUMN = "specimen_collection_date"
ORGANISM_COLUMN = "organism_name"
ANTIBIOTIC_COLUMN = "antibiotic"
SIR_COLUMN = "interpretation"


DEDUP_METHOD_ALIASES: dict[str, str] = {
    "glass": "first_glass",
    "df_glass": "first_glass",
    "first_glass": "first_glass",
    "14j": "first_14j",
    "df_14j": "first_14j",
    "first_14j": "first_14j",
    "14": "first_14j",
    "30j": "first_30j",
    "df_30j": "first_30j",
    "first_30j": "first_30j",
    "30": "first_30j",
    "365j": "first_365j",
    "df_365j": "first_365j",
    "first_365j": "first_365j",
    "365": "first_365j",
}


def build_deduplicated_databases(
    data_path: str | Path = "data.parquet",
    eligibility_scope: str = "all_eligible",
    verbose: bool = False,
) -> dict[str, pd.DataFrame]:
    """Construit les 4 DataFrames dédupliqués (df_glass, df_14j, df_30j, df_365j) et df_amr."""
    from amr_deduplication import (
        DeduplicationConfig,
        prepare_amr_dataframe,
        run_all_deduplications,
    )

    cfg = DeduplicationConfig(eligibility_scope=eligibility_scope)
    df_amr = prepare_amr_dataframe(data_path, config=cfg, verbose=verbose)
    df_amr, df_glass, df_14j, df_30j, df_365j = run_all_deduplications(df_amr, config=cfg)
    return {
        "df_amr": df_amr,
        "df_glass": df_glass,
        "df_14j": df_14j,
        "df_30j": df_30j,
        "df_365j": df_365j,
    }


def _read_tabular_file(file_path: str | Path) -> pd.DataFrame:
    """Charge un fichier tabulaire (.parquet, .csv, .tsv, .xlsx)."""
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(
            f"Fichier source introuvable : {p}. "
            "Vérifiez le chemin du fichier (.parquet ou .csv)."
        )
    suf = p.suffix.lower()
    if suf in (".csv", ".txt"):
        return pd.read_csv(p, low_memory=False)
    if suf == ".tsv":
        return pd.read_csv(p, sep="\t", low_memory=False)
    if suf in (".xlsx", ".xls"):
        return pd.read_excel(p)
    return pd.read_parquet(p)


def _ensure_long_ast_format(df: pd.DataFrame) -> pd.DataFrame:
    """Convertit un DataFrame large (colonnes ABX_* ou *_final_SIR_interpretation) au format long si nécessaire."""
    cols_lower = {str(c).strip().lower(): c for c in df.columns}
    if ANTIBIOTIC_COLUMN in cols_lower and SIR_COLUMN in cols_lower:
        return df

    # Détection des colonnes d'antibiotiques au format large
    sir_suffix_cols = [c for c in df.columns if str(c).endswith(ANTIBIOTIC_CSV_SUFFIX)]
    abx_prefix_cols = [c for c in df.columns if str(c).upper().startswith("ABX_")]

    wide_abx_cols = sir_suffix_cols or abx_prefix_cols
    if not wide_abx_cols:
        return df

    out = df.copy()
    if "row_uid" not in cols_lower:
        out["ROW_UID"] = [f"ISO_{i:06d}" for i in range(1, len(out) + 1)]

    id_vars = [c for c in out.columns if c not in wide_abx_cols]
    long_df = out.melt(
        id_vars=id_vars,
        value_vars=wide_abx_cols,
        var_name="ANTIBIOTIC",
        value_name="INTERPRETATION",
    )
    if sir_suffix_cols:
        long_df["ANTIBIOTIC"] = (
            long_df["ANTIBIOTIC"].astype(str).str.replace(ANTIBIOTIC_CSV_SUFFIX, "", regex=False)
        )
    else:
        long_df["ANTIBIOTIC"] = (
            long_df["ANTIBIOTIC"].astype(str).str.replace(r"^[aA][bB][xX]_", "", regex=True)
        )
    return long_df


def load_source_data(
    data_path: str | Path = "data.parquet",
    dataset_choice: str | Path | pd.DataFrame | None = None,
    eligibility_scope: str = "all_eligible",
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Load AMR data from a .parquet or .csv file and optionally filter to a deduplicated population.

    Args:
        data_path: Chemin vers le fichier source (.parquet ou .csv, format long ou large).
        dataset_choice: Base de données à analyser :
            - None / "raw" / "none" / "all" : base complète sans déduplication.
            - "df_glass" (ou "glass") : base dédupliquée annuellement selon WHO GLASS.
            - "df_14j"   (ou "14j")   : base dédupliquée par épisode de 14 jours.
            - "df_30j"   (ou "30j")   : base dédupliquée par épisode de 30 jours.
            - "df_365j"  (ou "365j")  : base dédupliquée par épisode de 365 jours.
            - Un DataFrame pandas (ex: df_glass, df_14j, df_30j, df_365j) ou un chemin .parquet / .csv
              contenant les isolats sélectionnés (identifiés par ROW_UID / row_uid).
        eligibility_scope: "all_eligible" (7 prélèvements harmonisés) ou "strict_glass".

    Returns:
        frame: Long-format DataFrame of all antibiotic test rows for the selected population.
        records: Deduplicated isolate/record-level DataFrame (one row per row_uid).
        antibiotic_columns: Sorted list of available antibiotic names.
    """
    raw_table = _ensure_long_ast_format(_read_tabular_file(data_path))

    # Détermination des ROW_UID retenus si une base issue de la déduplication est demandée
    selected_uids: set[str] | None = None
    dedup_label = f"Base complète ({Path(data_path).name}, sans déduplication)"

    if isinstance(dataset_choice, pd.DataFrame):
        uid_col = "ROW_UID" if "ROW_UID" in dataset_choice.columns else "row_uid"
        if uid_col not in dataset_choice.columns:
            raise ValueError("Le DataFrame fourni doit contenir la colonne 'ROW_UID' ou 'row_uid'.")
        selected_uids = set(dataset_choice[uid_col].dropna().astype(str))
        dedup_label = f"DataFrame fourni ({len(selected_uids):,} isolats)"
    elif isinstance(dataset_choice, (str, Path)) and str(dataset_choice).strip().lower() not in (
        "", "none", "raw", "all", "brut", str(data_path).strip().lower()
    ):
        choice_str = str(dataset_choice).strip()
        choice_key = choice_str.lower()
        if choice_key in DEDUP_METHOD_ALIASES:
            flag_col = DEDUP_METHOD_ALIASES[choice_key]
            from amr_deduplication import (
                DeduplicationConfig,
                METHOD_LABELS,
                prepare_amr_dataframe,
                dedup_glass,
                dedup_episode,
            )

            cfg = DeduplicationConfig(eligibility_scope=eligibility_scope)
            df_amr = prepare_amr_dataframe(raw_table, config=cfg, verbose=False)
            if flag_col == "first_glass":
                df_amr = dedup_glass(df_amr, config=cfg)
            else:
                days_val = int(flag_col.replace("first_", "").replace("j", ""))
                df_amr = dedup_episode(df_amr, days=days_val, config=cfg)
            selected_uids = set(df_amr.loc[df_amr[flag_col], "ROW_UID"].astype(str))
            dedup_label = f"{METHOD_LABELS.get(flag_col, choice_str)} ({len(selected_uids):,} isolats)"
        elif Path(choice_str).exists():
            ext_df = _read_tabular_file(choice_str)
            uid_col = "ROW_UID" if "ROW_UID" in ext_df.columns else "row_uid"
            if uid_col not in ext_df.columns:
                raise ValueError(
                    f"Le fichier '{choice_str}' doit contenir la colonne 'ROW_UID' ou 'row_uid'."
                )
            selected_uids = set(ext_df[uid_col].dropna().astype(str))
            dedup_label = f"Fichier {choice_str} ({len(selected_uids):,} isolats)"
        else:
            raise ValueError(
                f"Choix de base inconnu : '{dataset_choice}'. "
                "Valeurs acceptées : 'df_glass', 'df_14j', 'df_30j', 'df_365j', 'raw', "
                "un DataFrame pandas ou un chemin vers un fichier .parquet / .csv."
            )

    frame = raw_table.copy()
    frame.columns = [str(column).strip().lower() for column in frame.columns]

    if selected_uids is not None:
        frame = frame.loc[frame[ROW_UID_COLUMN].astype(str).isin(selected_uids)].copy()
        frame = frame.reset_index(drop=True)

    required_columns = {
        ROW_UID_COLUMN,
        DATE_COLUMN,
        HOSPITAL_COLUMN,
        YEAR_COLUMN,
        WARD_COLUMN,
        ORGANISM_COLUMN,
        ANTIBIOTIC_COLUMN,
        SIR_COLUMN,
    }
    missing_columns = sorted(required_columns.difference(frame.columns))
    if missing_columns:
        raise ValueError(
            "Colonnes requises absentes de la base : " + ", ".join(missing_columns)
        )

    # Normalisation propre de l'année (ex: 2019.0 -> "2019") et repli de date sur analysis_date
    if "analysis_date" in frame.columns:
        spec_dt = frame[DATE_COLUMN].astype("string").str.strip().fillna("")
        anal_dt = frame["analysis_date"].astype("string").str.strip().fillna("")
        frame[DATE_COLUMN] = spec_dt.where(spec_dt != "", anal_dt)

    if YEAR_COLUMN in frame.columns:
        yr_num = pd.to_numeric(frame[YEAR_COLUMN], errors="coerce")
        parsed_dt_yr = pd.to_datetime(frame[DATE_COLUMN], errors="coerce").dt.year
        yr_final = yr_num.fillna(parsed_dt_yr)
        frame[YEAR_COLUMN] = yr_final.apply(
            lambda v: str(int(v)) if pd.notna(v) else ""
        )

    for column in frame.columns:
        if isinstance(frame[column].dtype, pd.CategoricalDtype):
            frame[column] = frame[column].astype("string").str.strip().fillna("")
        elif pd.api.types.is_string_dtype(frame[column].dtype) or frame[column].dtype == object:
            frame[column] = frame[column].astype("string").str.strip().fillna("")

    records = (
        frame.drop_duplicates(subset=[ROW_UID_COLUMN]).reset_index(drop=True).copy()
    )
    frame.attrs["dataset_label"] = dedup_label
    records.attrs["dataset_label"] = dedup_label

    abx_mask = frame[ANTIBIOTIC_COLUMN].astype("string").str.strip() != ""
    antibiotic_columns = sorted(
        str(val).strip()
        for val in frame.loc[abx_mask, ANTIBIOTIC_COLUMN].unique()
        if str(val).strip() != ""
    )
    return frame, records, antibiotic_columns


def format_label(value: Any) -> str:
    if value == ALL_ANTIBIOTICS or value is None:
        return "Tous les antibiotiques"
    text = str(value)
    if "_" in text:
        return text.replace("_", " ").title()
    return text


def format_bacterium(value: Any) -> str:
    if value == ALL_BACTERIA or value is None:
        return "Toutes les bactéries"
    return str(value)


def fill_missing_label(series: pd.Series, missing_label: str = MISSING_LABEL) -> pd.Series:
    cleaned = series.astype("string").str.strip().fillna("")
    return cleaned.where(cleaned != "", missing_label).astype(object)


def missing_value_mask(series: pd.Series) -> pd.Series:
    if pd.api.types.is_string_dtype(series.dtype) or series.dtype == object:
        return series.isna() | series.astype("string").str.strip().eq("").fillna(True)
    return series.isna()


def filter_amr_data(
    frame: pd.DataFrame,
    records: pd.DataFrame | None = None,
    *,
    years: list[str] | None = None,
    hospitals: list[str] | None = None,
    wards: list[str] | None = None,
    specimens: list[str] | None = None,
    organisms: list[str] | None = None,
    antibiotics: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Filter test-level and record-level frames according to selected dimensions."""
    if records is None:
        records = frame.drop_duplicates(subset=[ROW_UID_COLUMN]).reset_index(drop=True)

    dimension_filters = {
        YEAR_COLUMN: years,
        HOSPITAL_COLUMN: hospitals,
        WARD_COLUMN: wards,
        SPECIMEN_COLUMN: specimens,
        ORGANISM_COLUMN: organisms,
    }

    filtered_frame = frame
    filtered_records = records
    for col, allowed in dimension_filters.items():
        if not allowed or col not in filtered_frame.columns:
            continue
        allowed_set = {str(v).strip() for v in allowed}
        include_missing = MISSING_LABEL in allowed_set or "" in allowed_set
        present_values = allowed_set - {MISSING_LABEL, ""}

        mask_f = filtered_frame[col].astype(str).isin(present_values)
        mask_r = filtered_records[col].astype(str).isin(present_values)
        if include_missing:
            mask_f |= missing_value_mask(filtered_frame[col])
            mask_r |= missing_value_mask(filtered_records[col])
        filtered_frame = filtered_frame.loc[mask_f]
        filtered_records = filtered_records.loc[mask_r]

    abx_mask = filtered_frame[ANTIBIOTIC_COLUMN].astype("string").str.strip() != ""
    available_abx = sorted(
        str(val).strip()
        for val in filtered_frame.loc[abx_mask, ANTIBIOTIC_COLUMN].unique()
        if str(val).strip() != ""
    )
    if antibiotics:
        wanted = {str(a).strip().casefold() for a in antibiotics}
        selected_abx = [a for a in available_abx if a.casefold() in wanted]
    else:
        selected_abx = available_abx

    return filtered_frame, filtered_records, selected_abx


def antibiotic_counts(
    frame: pd.DataFrame,
    group_columns: list[str] | None = None,
    antibiotic_columns: list[str] | None = None,
    records_frame: pd.DataFrame | None = None,
    low_count_threshold: int = 30,
) -> pd.DataFrame:
    """Compute S, I, R, tested denominator (S+I+R) and %R by group and antibiotic."""
    group_columns = list(group_columns or [])
    if antibiotic_columns is None:
        abx_mask = frame[ANTIBIOTIC_COLUMN].astype("string").str.strip() != ""
        antibiotic_columns = sorted(
            str(val).strip()
            for val in frame.loc[abx_mask, ANTIBIOTIC_COLUMN].unique()
            if str(val).strip() != ""
        )
    if not antibiotic_columns:
        return pd.DataFrame()

    if records_frame is None:
        records_frame = (
            frame.drop_duplicates(subset=[ROW_UID_COLUMN])
            if ROW_UID_COLUMN in frame.columns
            else frame
        )

    tests = frame.loc[frame[ANTIBIOTIC_COLUMN].isin(antibiotic_columns)]
    sir_vals = tests[SIR_COLUMN].astype("string").str.strip().str.upper().fillna("")
    valid_sir_mask = sir_vals.isin(["S", "I", "R"])
    present_mask = ~missing_value_mask(tests[SIR_COLUMN])
    non_interp_mask = present_mask & ~valid_sir_mask

    if group_columns:
        groups = records_frame[group_columns].drop_duplicates().reset_index(drop=True)
        for col in group_columns:
            groups[col] = groups[col].astype(str)
        row_counts = (
            records_frame.groupby(group_columns, dropna=False, observed=True)
            .size()
            .rename("_n_lignes")
            .reset_index()
        )
        for col in group_columns:
            row_counts[col] = row_counts[col].astype(str)
        groups = groups.merge(row_counts, on=group_columns, how="left")
        groups["_n_lignes"] = groups["_n_lignes"].fillna(0).astype(int)

        abx_df = pd.DataFrame({ANTIBIOTIC_COLUMN: [str(a) for a in antibiotic_columns]})
        grid = groups.merge(abx_df, how="cross")

        if valid_sir_mask.any():
            sir_subset = tests.loc[
                valid_sir_mask, group_columns + [ANTIBIOTIC_COLUMN]
            ].copy()
            for col in group_columns + [ANTIBIOTIC_COLUMN]:
                sir_subset[col] = sir_subset[col].astype(str)
            sir_subset["_sir"] = sir_vals.loc[valid_sir_mask].values
            sir_counts = (
                sir_subset.groupby(
                    group_columns + [ANTIBIOTIC_COLUMN, "_sir"],
                    dropna=False,
                    observed=True,
                )
                .size()
                .unstack("_sir", fill_value=0)
                .reset_index()
            )
            grid = grid.merge(
                sir_counts,
                on=group_columns + [ANTIBIOTIC_COLUMN],
                how="left",
            )

        if non_interp_mask.any():
            ni_subset = tests.loc[
                non_interp_mask, group_columns + [ANTIBIOTIC_COLUMN]
            ].copy()
            for col in group_columns + [ANTIBIOTIC_COLUMN]:
                ni_subset[col] = ni_subset[col].astype(str)
            ni_counts = (
                ni_subset.groupby(
                    group_columns + [ANTIBIOTIC_COLUMN],
                    dropna=False,
                    observed=True,
                )
                .size()
                .rename("_n_non_interpretable")
                .reset_index()
            )
            grid = grid.merge(
                ni_counts,
                on=group_columns + [ANTIBIOTIC_COLUMN],
                how="left",
            )
        else:
            grid["_n_non_interpretable"] = 0

        counts = grid
    else:
        total_records = len(records_frame)
        counts = pd.DataFrame(
            {
                ANTIBIOTIC_COLUMN: [str(a) for a in antibiotic_columns],
                "_n_lignes": total_records,
            }
        )
        if valid_sir_mask.any():
            sir_subset = pd.DataFrame(
                {
                    ANTIBIOTIC_COLUMN: tests.loc[
                        valid_sir_mask, ANTIBIOTIC_COLUMN
                    ].astype(str).values,
                    "_sir": sir_vals.loc[valid_sir_mask].values,
                }
            )
            sir_counts = (
                sir_subset.groupby(
                    [ANTIBIOTIC_COLUMN, "_sir"], dropna=False, observed=True
                )
                .size()
                .unstack("_sir", fill_value=0)
                .reset_index()
            )
            counts = counts.merge(sir_counts, on=ANTIBIOTIC_COLUMN, how="left")
        if non_interp_mask.any():
            ni_subset = tests.loc[non_interp_mask, ANTIBIOTIC_COLUMN].astype(str)
            ni_counts = (
                ni_subset.value_counts(dropna=False)
                .rename_axis(ANTIBIOTIC_COLUMN)
                .reset_index(name="_n_non_interpretable")
            )
            counts = counts.merge(ni_counts, on=ANTIBIOTIC_COLUMN, how="left")
        else:
            counts["_n_non_interpretable"] = 0

    for category in ("S", "I", "R"):
        if category not in counts.columns:
            counts[category] = 0
        counts[category] = counts[category].fillna(0).astype(int)
    counts["_n_lignes"] = counts["_n_lignes"].fillna(0).astype(int)
    counts["_n_non_interpretable"] = (
        counts["_n_non_interpretable"].fillna(0).astype(int)
    )
    counts["Testés (S/I/R)"] = counts[["S", "I", "R"]].sum(axis=1)
    counts["Sans résultat"] = (
        counts["_n_lignes"]
        - counts["Testés (S/I/R)"]
        - counts["_n_non_interpretable"]
    ).clip(lower=0)
    counts["Résistants (R)"] = counts["R"]
    counts["%R"] = (
        counts["R"]
        .div(counts["Testés (S/I/R)"].where(counts["Testés (S/I/R)"] > 0))
        .mul(100)
        .round(2)
    )
    counts["antibiotique"] = counts[ANTIBIOTIC_COLUMN]
    counts["Signalement d’effectif"] = counts["Testés (S/I/R)"].map(
        lambda n: "Aucun résultat"
        if n == 0
        else "Effectif faible"
        if n < low_count_threshold
        else "Effectif au-dessus du seuil"
    )
    return counts


def pooled_antibiotic_counts(
    frame: pd.DataFrame,
    group_columns: list[str],
    antibiotic_columns: list[str] | None = None,
    records_frame: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Pool S/I/R counts across selected antibiotics and aggregate by group_columns."""
    counts = antibiotic_counts(
        frame, group_columns, antibiotic_columns, records_frame=records_frame
    )
    if counts.empty:
        return counts
    pooled = (
        counts.groupby(group_columns, dropna=False, as_index=False, observed=True)[
            ["S", "I", "R", "_n_lignes", "_n_non_interpretable", "Sans résultat"]
        ]
        .sum()
    )
    pooled["Testés (S/I/R)"] = pooled[["S", "I", "R"]].sum(axis=1)
    pooled["Résistants (R)"] = pooled["R"]
    pooled["%R"] = (
        pooled["R"]
        .div(pooled["Testés (S/I/R)"].where(pooled["Testés (S/I/R)"] > 0))
        .mul(100)
        .round(2)
    )
    return pooled


def compute_resistance_summary(
    frame: pd.DataFrame,
    records_frame: pd.DataFrame | None = None,
    antibiotic_columns: list[str] | None = None,
    low_count_threshold: int = 30,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the detailed resistance table and the Bactérie × Antibiotique %R matrix."""
    resistance = antibiotic_counts(
        frame,
        [ORGANISM_COLUMN],
        antibiotic_columns,
        records_frame=records_frame,
        low_count_threshold=low_count_threshold,
    )
    if resistance.empty:
        return pd.DataFrame(), pd.DataFrame()

    resistance["Bactérie"] = fill_missing_label(resistance[ORGANISM_COLUMN])
    resistance["Antibiotique"] = resistance["antibiotique"].map(format_label)
    resistance_table = (
        resistance[
            [
                "Bactérie",
                "Antibiotique",
                "Résistants (R)",
                "Testés (S/I/R)",
                "S",
                "I",
                "%R",
                "Signalement d’effectif",
            ]
        ]
        .sort_values(["Bactérie", "Antibiotique"])
        .reset_index(drop=True)
    )
    matrix = resistance.pivot(
        index="Bactérie", columns="Antibiotique", values="%R"
    )
    return resistance_table, matrix


def style_resistance(value: Any) -> str:
    """Medical color-coding for %R cells (<20% green, 20-50% amber, >50% red)."""
    if pd.isna(value):
        return ""
    if value < 20:
        return "background-color: #dcfce7; color: #166534"
    if value <= 50:
        return "background-color: #fef3c7; color: #92400e"
    return "background-color: #fee2e2; color: #991b1b"


def build_monthly_cube(frame: pd.DataFrame) -> pd.DataFrame:
    """Build the monthly S/R aggregation cube from the long-format frame."""
    abx_mask = frame[ANTIBIOTIC_COLUMN].astype("string").str.strip() != ""
    tests = frame.loc[abx_mask]
    sir_vals = tests[SIR_COLUMN].astype("string").str.strip().str.upper()
    abx_vals = tests[ANTIBIOTIC_COLUMN].astype("string").str.strip()
    valid_sr = (abx_vals != "") & sir_vals.isin(["S", "R"])
    sr_tests = tests.loc[valid_sr]

    if sr_tests.empty:
        return pd.DataFrame(columns=["bacterie", "antibiotique", "mois", "S", "R"])

    sr_bacteria = (
        sr_tests[ORGANISM_COLUMN].astype(str).str.strip().replace("", "Non spécifié")
    )
    sr_abx = sr_tests[ANTIBIOTIC_COLUMN].astype(str).str.strip()
    sr_code = sir_vals.loc[valid_sr].astype(str)
    parsed_dates = pd.to_datetime(sr_tests[DATE_COLUMN], errors="coerce")
    valid_date = parsed_dates.notna()
    if not valid_date.any():
        return pd.DataFrame(columns=["bacterie", "antibiotique", "mois", "S", "R"])

    monthly = (
        pd.DataFrame(
            {
                "bacterie": sr_bacteria.loc[valid_date],
                "antibiotique": sr_abx.loc[valid_date],
                "mois": parsed_dates.loc[valid_date].dt.strftime("%Y-%m"),
                "_sir": sr_code.loc[valid_date],
            }
        )
        .groupby(["bacterie", "antibiotique", "mois", "_sir"], dropna=False)
        .size()
        .unstack("_sir", fill_value=0)
        .reset_index()
    )
    for col in ("S", "R"):
        if col not in monthly.columns:
            monthly[col] = 0
    return monthly[["bacterie", "antibiotique", "mois", "S", "R"]].sort_values(
        ["bacterie", "antibiotique", "mois"]
    ).reset_index(drop=True)


def get_monthly_resistance_series(
    cube_monthly: pd.DataFrame,
    bacterie: str = ALL_BACTERIA,
    antibiotique: str = ALL_ANTIBIOTICS,
    min_tests: int = MIN_MONTHLY_TESTS,
) -> tuple[pd.DataFrame, list[float]]:
    """Return monthly S, R, total and %R series for a bacterium × antibiotic selection."""
    series = cube_monthly.copy()
    if bacterie and bacterie != ALL_BACTERIA:
        series = series.loc[series["bacterie"].str.casefold() == bacterie.casefold()]
    if antibiotique and antibiotique != ALL_ANTIBIOTICS:
        series = series.loc[
            series["antibiotique"].str.casefold() == antibiotique.casefold()
        ]
    series = (
        series.groupby("mois", as_index=False)[["S", "R"]].sum().sort_values("mois")
    )
    if series.empty:
        return series, []

    series["total"] = series["S"] + series["R"]
    series = series[series["total"] >= min_tests].sort_values("mois").copy()
    series["%R"] = series["R"].div(series["total"]).mul(100).round(2)
    values = series["%R"].round(1).tolist()
    return series.reset_index(drop=True), values


def run_monthly_forecast(
    cube_monthly: pd.DataFrame,
    bacterie: str = ALL_BACTERIA,
    antibiotique: str = ALL_ANTIBIOTICS,
    horizon: int = 12,
    method: str = "holt",
    alpha: float = 0.35,
    beta: float = 0.12,
    gamma: float = 0.1,
    arima_order: tuple[int, int, int] = (1, 1, 1),
    arima_auto: bool = True,
    arima_p_max: int = 2,
    arima_d_max: int = 1,
    arima_q_max: int = 2,
    sarima_order: tuple[int, int, int] = (1, 0, 0),
    sarima_seasonal_order: tuple[int, int, int, int] = (0, 0, 0, 12),
    sarima_auto: bool = True,
    sarima_p_max: int = 2,
    sarima_d_max: int = 1,
    sarima_q_max: int = 2,
    sarima_seasonal_p_max: int = 1,
    sarima_seasonal_d_max: int = 1,
    sarima_seasonal_q_max: int = 1,
    glm_seasonal: bool = True,
    training_start: str | None = None,
    training_end: str | None = None,
    min_tests: int = MIN_MONTHLY_TESTS,
) -> dict[str, Any] | None:
    """Run monthly forecasting and diagnostics for a bacterium × antibiotic pair."""
    series, _ = get_monthly_resistance_series(
        cube_monthly, bacterie, antibiotique, min_tests=min_tests
    )
    if series.empty:
        return None
    if training_start is not None:
        series = series.loc[series["mois"] >= training_start]
    if training_end is not None:
        series = series.loc[series["mois"] <= training_end]
    if len(series) < 6:
        return None

    values = series["R"].div(series["total"]).mul(100).round(1).tolist()
    months = series["mois"].tolist()
    months_contiguous = not any(
        (int(current[:4]) * 12 + int(current[5:7]))
        - (int(previous[:4]) * 12 + int(previous[5:7]))
        != 1
        for previous, current in zip(months, months[1:])
    )
    if method in ("seasonal_naive", "holt_winters", "sarima") and not months_contiguous:
        raise ValueError(
            "Cette méthode saisonnière nécessite des mois consécutifs. "
            "Certains mois n'ont pas assez de tests pour être inclus."
        )

    series = series.copy()
    series["N"] = series["total"]
    model_result = forecast_time_series(
        values,
        horizon=horizon,
        method=method,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        arima_order=arima_order,
        arima_auto=arima_auto,
        arima_p_max=arima_p_max,
        arima_d_max=arima_d_max,
        arima_q_max=arima_q_max,
        sarima_order=sarima_order,
        sarima_seasonal_order=sarima_seasonal_order,
        sarima_auto=sarima_auto,
        sarima_p_max=sarima_p_max,
        sarima_d_max=sarima_d_max,
        sarima_q_max=sarima_q_max,
        sarima_seasonal_p_max=sarima_seasonal_p_max,
        sarima_seasonal_d_max=sarima_seasonal_d_max,
        sarima_seasonal_q_max=sarima_seasonal_q_max,
        monthly_counts=series,
        months=months,
        glm_seasonal=glm_seasonal,
        future_n=max(1, int(round(series["N"].median()))),
    )

    year, month = map(int, months[-1].split("-"))
    future = []
    for index in range(horizon):
        month += 1
        if month > 12:
            month = 1
            year += 1
        rate = round(model_result["forecast"][index], 1)
        vigilance = (
            "Critique (>50 %)"
            if rate > 50
            else "Modéré (20–50 %)"
            if rate > 20
            else "Faible (<20 %)"
        )
        future.append(
            {
                "Mois": f"{year}-{month:02d}",
                "Taux prédit (%R)": rate,
                "IC inférieur (95 %)": round(model_result["ci_lower"][index], 1),
                "IC supérieur (95 %)": round(model_result["ci_upper"][index], 1),
                "N futur": (
                    model_result["future_n"][index]
                    if "future_n" in model_result
                    else None
                ),
                "IC observé inférieur": (
                    round(model_result["predictive_lower"][index], 1)
                    if "predictive_lower" in model_result
                    else None
                ),
                "IC observé supérieur": (
                    round(model_result["predictive_upper"][index], 1)
                    if "predictive_upper" in model_result
                    else None
                ),
                "Niveau de vigilance": vigilance,
            }
        )

    fitted_values = model_result.get("fitted", [])
    residuals = [
        float(actual) - float(fitted)
        for actual, fitted in zip(values, fitted_values)
        if pd.notna(fitted) and math.isfinite(float(fitted))
    ]
    monthly_diagnostics: dict[str, Any] = {
        "nombre de résidus": len(residuals),
        "test de Ljung–Box": None,
        "note": None,
    }
    if not months_contiguous:
        monthly_diagnostics["note"] = (
            "Non calculé : des mois sans observation interrompent la série."
        )
    elif len(residuals) < 8:
        monthly_diagnostics["note"] = (
            "Non calculé : au moins 8 résidus consécutifs sont nécessaires."
        )
    else:
        from statsmodels.stats.diagnostic import acorr_ljungbox

        lag = min(12, len(residuals) // 3)
        ljung_box = acorr_ljungbox(residuals, lags=[lag], return_df=True).iloc[0]
        monthly_diagnostics["test de Ljung–Box"] = {
            "retard": int(lag),
            "statistique": float(ljung_box["lb_stat"]),
            "p-valeur": float(ljung_box["lb_pvalue"]),
        }

    return {
        "series": series,
        "months": months,
        "values": values,
        "fitted": fitted_values,
        "future": future,
        "future_df": pd.DataFrame(future),
        "mae": model_result["metrics"]["MAE"],
        "rmse": model_result["metrics"]["RMSE"],
        "trend": model_result["trend_slope_per_month"],
        "trend_per_year": model_result["trend_slope_per_year"],
        "method": model_result["model"],
        "diagnostics": monthly_diagnostics,
        "model_info": {
            key: model_result[key]
            for key in (
                "order",
                "selection",
                "candidate_count",
                "candidate_failures",
                "converged",
                "convergence_warnings",
                "seasonal_order",
                "dispersion",
                "seasonal_effect",
                "fit_warnings",
            )
            if key in model_result
        },
    }


def compare_monthly_methods_for_pair(
    cube_monthly: pd.DataFrame,
    bacterie: str = ALL_BACTERIA,
    antibiotique: str = ALL_ANTIBIOTICS,
    horizon: int = 12,
    holt_alpha: float = 0.35,
    holt_beta: float = 0.12,
    holt_winters_alpha: float = 0.3,
    holt_winters_beta: float = 0.1,
    holt_winters_gamma: float = 0.1,
    training_start: str | None = None,
    training_end: str | None = None,
    min_tests: int = MIN_MONTHLY_TESTS,
) -> pd.DataFrame:
    """Compare the 5 monthly forecasting methods on a chronological holdout window."""
    series, _ = get_monthly_resistance_series(
        cube_monthly, bacterie, antibiotique, min_tests=min_tests
    )
    if training_start is not None:
        series = series.loc[series["mois"] >= training_start]
    if training_end is not None:
        series = series.loc[series["mois"] <= training_end]
    series = series.copy()
    values = series["R"].div(series["total"]).mul(100).round(1).tolist()
    if len(values) <= horizon:
        raise ValueError(
            "Il faut plus d'observations historiques que l'horizon de test choisi."
        )
    months = series["mois"].tolist()
    months_contiguous = all(
        (int(current[:4]) * 12 + int(current[5:7]))
        - (int(previous[:4]) * 12 + int(previous[5:7]))
        == 1
        for previous, current in zip(months, months[1:])
    )
    rows = compare_forecasting_methods(
        tuple(values),
        horizon=horizon,
        alpha=holt_alpha,
        beta=holt_beta,
        seasonal_data_contiguous=months_contiguous,
        holt_winters_alpha=holt_winters_alpha,
        holt_winters_beta=holt_winters_beta,
        holt_winters_gamma=holt_winters_gamma,
        monthly_counts=series.assign(N=series["total"]),
        months=months,
    )
    return pd.DataFrame(rows).sort_values("MAE test (%)", na_position="last").reset_index(drop=True)


def _clean_text_series(series: pd.Series, default_value: str) -> pd.Series:
    cleaned = series.astype("string").str.strip().fillna("")
    return cleaned.where(cleaned != "", default_value)


def _build_dashboard_data_from_parquet_frame(frame: pd.DataFrame) -> dict[str, Any]:
    frame = frame.copy()
    frame.columns = [str(col).strip().lower() for col in frame.columns]

    required = {
        ROW_UID_COLUMN,
        YEAR_COLUMN,
        HOSPITAL_COLUMN,
        ORGANISM_COLUMN,
        WARD_COLUMN,
        DATE_COLUMN,
        ANTIBIOTIC_COLUMN,
        SIR_COLUMN,
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(
            "Colonnes obligatoires absentes de data.parquet : " + ", ".join(missing)
        )

    records = frame.drop_duplicates(subset=[ROW_UID_COLUMN]).copy()
    rec_bacteria = _clean_text_series(records[ORGANISM_COLUMN], "Non spécifié")
    bacteria = Counter(rec_bacteria)

    abx_series = frame[ANTIBIOTIC_COLUMN].astype("string").str.strip().fillna("")
    antibiotic_list = sorted(val for val in abx_series.unique() if val)

    monthly_df = build_monthly_cube(frame)
    monthly_rows = [
        [str(r.bacterie), str(r.antibiotique), str(r.mois), int(r.S), int(r.R)]
        for r in monthly_df.itertuples(index=False)
    ]

    return {
        "bacteria": [name for name, _ in bacteria.most_common()],
        "antibiotics": antibiotic_list,
        "cube_monthly": monthly_rows,
    }


def build_dashboard_data(
    source_path: str | Path | pd.DataFrame = "data.parquet",
    include_forecasts: bool = False,
) -> dict[str, Any]:
    """Aggregate data.parquet into the schema used by CLI monthly predictions."""
    del include_forecasts
    if isinstance(source_path, pd.DataFrame):
        return _build_dashboard_data_from_parquet_frame(source_path)

    frame = pd.read_parquet(Path(source_path))
    return _build_dashboard_data_from_parquet_frame(frame)


def extract_monthly_series(
    data: dict[str, Any],
    bacterium: str,
    antibiotic: str,
    min_tests: int = MIN_MONTHLY_TESTS,
) -> tuple[list[str], list[float], list[int]]:
    """Return a pair's monthly resistance series from the common data payload."""
    grouped: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for row_bacterium, row_antibiotic, month, susceptible, resistant in data.get(
        "cube_monthly", []
    ):
        if row_bacterium == bacterium and row_antibiotic == antibiotic:
            grouped[month][0] += susceptible
            grouped[month][1] += resistant

    selected = [
        (month, counts)
        for month, counts in sorted(grouped.items())
        if sum(counts) >= min_tests
    ]
    return (
        [month for month, _ in selected],
        [
            round(resistant / (susceptible + resistant) * 100, 2)
            for _, (susceptible, resistant) in selected
        ],
        [sum(counts) for _, counts in selected],
    )
