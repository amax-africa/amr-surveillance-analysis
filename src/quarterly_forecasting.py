"""National quarterly antimicrobial-resistance aggregation and forecasting.

The functions in this module are deliberately independent of Streamlit. They
aggregate pooled S/I/R counts and offer chronological backtesting and forecasts
for one organism/antibiotic pair. Missing quarters remain missing; they are
never interpreted as zero resistance or imputed for model fitting.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
import warnings
from typing import Any, Literal, Sequence

import numpy as np
import pandas as pd


DATE_COLUMN = "specimen_collection_date"
ORGANISM_COLUMN = "organism_name"
ANTIBIOTIC_COLUMN = "antibiotic"
SIR_COLUMN = "interpretation"
ROW_UID_COLUMN = "row_uid"
SIR_CODES = ("S", "I", "R")


def aggregate_quarterly_counts(
    data: pd.DataFrame,
    organism: str | None,
    antibiotic: str | None,
    *,
    low_n_threshold: int = 30,
    date_column: str = DATE_COLUMN,
    organism_column: str = ORGANISM_COLUMN,
    antibiotic_column: str = ANTIBIOTIC_COLUMN,
    sir_column: str = SIR_COLUMN,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Pool S/I/R results into a continuous quarterly PeriodIndex from data.parquet."""
    if low_n_threshold < 0:
        raise ValueError("low_n_threshold must be non-negative")

    frame = data
    if any(str(col) != str(col).strip().lower() for col in frame.columns):
        frame = frame.copy()
        frame.columns = [str(col).strip().lower() for col in frame.columns]

    date_col = date_column.strip().lower()
    organism_col = organism_column.strip().lower()
    antibiotic_col = antibiotic_column.strip().lower()
    sir_col = sir_column.strip().lower()

    required = {date_col, organism_col, antibiotic_col, sir_col}
    missing_columns = sorted(required.difference(frame.columns))
    if missing_columns:
        raise KeyError(f"Missing required column(s): {', '.join(missing_columns)}")

    selected = frame
    if organism is not None:
        selected = selected.loc[
            selected[organism_col].astype("string").str.strip().eq(organism.strip())
        ]
    else:
        selected = selected.loc[
            selected[organism_col].astype("string").str.strip().ne("").fillna(False)
        ]

    abx_series = selected[antibiotic_col].astype("string").str.strip().fillna("")
    available_abx = set(val for val in abx_series.unique() if val)
    if antibiotic is not None:
        clean_abx = antibiotic.strip()
        matched_abx = next(
            (a for a in available_abx if a.casefold() == clean_abx.casefold()),
            None,
        )
        if matched_abx is None:
            all_abx = set(
                val
                for val in frame[antibiotic_col].astype("string").str.strip().dropna().unique()
                if val
            )
            matched_all = next(
                (a for a in all_abx if a.casefold() == clean_abx.casefold()),
                None,
            )
            if matched_all is None:
                raise KeyError(f"Antibiotic not found: {antibiotic}")
            selected_tests = selected.iloc[0:0]
        else:
            selected_tests = selected.loc[abx_series.eq(matched_abx)]
        abx_col_meta = matched_abx or clean_abx
    else:
        selected_tests = selected.loc[abx_series.ne("")]
        abx_col_meta = None

    date_basis = (
        selected.drop_duplicates(subset=[ROW_UID_COLUMN])
        if ROW_UID_COLUMN in selected.columns
        else selected
    )
    parsed_basis_dates = pd.to_datetime(date_basis[date_col], errors="coerce")
    raw_dates = date_basis[date_col]
    string_dates = raw_dates.astype("string")
    blank_date_mask = string_dates.str.strip().eq("").fillna(False)
    missing_date_count = int((raw_dates.isna() | blank_date_mask).sum())
    unparseable_date_count = int((raw_dates.notna() & parsed_basis_dates.isna()).sum())
    if raw_dates.dtype == object or pd.api.types.is_string_dtype(raw_dates.dtype):
        unparseable_date_count = int(
            (
                raw_dates.notna()
                & string_dates.str.strip().ne("").fillna(False)
                & parsed_basis_dates.isna()
            ).sum()
        )
    valid_date_mask = parsed_basis_dates.notna()
    organism_valid_dates = parsed_basis_dates.loc[valid_date_mask]
    organism_rows_count = int(len(date_basis))
    valid_date_rows_count = int(valid_date_mask.sum())

    test_dates = pd.to_datetime(selected_tests[date_col], errors="coerce")
    valid_test_dates = test_dates.notna()
    dated_tests = selected_tests.loc[valid_test_dates]
    codes = dated_tests[sir_col].astype("string").str.strip().str.upper()
    valid_code_mask = codes.isin(SIR_CODES)
    usable_rows = pd.DataFrame(
        {
            "_quarter": test_dates.loc[valid_test_dates].loc[valid_code_mask].dt.to_period("Q-DEC"),
            "_sir": codes.loc[valid_code_mask],
        }
    )
    valid_result_count = len(usable_rows)
    non_sir_with_valid_date = int(len(dated_tests) - valid_result_count)

    metadata: dict[str, Any] = {
        "organism": organism or "all",
        "antibiotic": antibiotic or "all",
        "antibiotic_column": abx_col_meta,
        "low_n_threshold": int(low_n_threshold),
        "organism_rows": organism_rows_count,
        "missing_date_rows": missing_date_count,
        "unparseable_date_rows": unparseable_date_count,
        "invalid_date_rows": missing_date_count + unparseable_date_count,
        "valid_date_rows": valid_date_rows_count,
        "non_sir_result_rows_with_valid_date": non_sir_with_valid_date,
        "usable_sir_rows": int(valid_result_count),
        "coverage_start": None,
        "coverage_end": None,
        "incomplete_quarters": [],
        "incomplete_quarter_definition": (
            "First/last calendar quarter intersecting the selected organism's "
            "parseable date span without full boundary-to-boundary coverage."
        ),
    }
    if usable_rows.empty:
        empty = pd.DataFrame(
            columns=[
                "S", "I", "R", "N", "p_R", "low_n", "boundary_incomplete",
                "quarter_start", "quarter_end",
            ],
            index=pd.PeriodIndex([], freq="Q-DEC", name="quarter"),
        )
        return empty, metadata

    quarter_counts = (
        usable_rows.groupby(["_quarter", "_sir"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=SIR_CODES, fill_value=0)
    )
    all_quarters = pd.period_range(
        quarter_counts.index.min(), quarter_counts.index.max(), freq="Q-DEC", name="quarter"
    )
    counts = quarter_counts.reindex(all_quarters, fill_value=0).astype("int64")
    counts["N"] = counts.loc[:, list(SIR_CODES)].sum(axis=1)
    counts["p_R"] = counts["R"].div(counts["N"].where(counts["N"].gt(0)))
    counts["low_n"] = counts["N"].gt(0) & counts["N"].lt(low_n_threshold)

    span_start = organism_valid_dates.min().normalize()
    span_end = organism_valid_dates.max().normalize()
    quarter_starts = pd.Series(all_quarters.start_time, index=all_quarters)
    quarter_ends = pd.Series(all_quarters.end_time.normalize(), index=all_quarters)
    boundary_incomplete = (quarter_starts.index == all_quarters[0]) & (
        quarter_starts.lt(span_start).to_numpy()
    )
    boundary_incomplete |= (quarter_ends.index == all_quarters[-1]) & (
        quarter_ends.gt(span_end).to_numpy()
    )
    counts["boundary_incomplete"] = boundary_incomplete
    counts["quarter_start"] = quarter_starts
    counts["quarter_end"] = quarter_ends
    counts.index.name = "quarter"

    incomplete = [
        {
            "quarter": str(period),
            "quarter_start": period.start_time.date().isoformat(),
            "quarter_end": period.end_time.date().isoformat(),
        }
        for period, partial in zip(all_quarters, boundary_incomplete)
        if partial
    ]
    metadata.update(
        {
            "coverage_start": span_start.date().isoformat(),
            "coverage_end": span_end.date().isoformat(),
            "first_quarter": str(all_quarters[0]),
            "last_quarter": str(all_quarters[-1]),
            "quarter_count": len(all_quarters),
            "incomplete_quarters": incomplete,
            "total_S": int(counts["S"].sum()),
            "total_I": int(counts["I"].sum()),
            "total_R": int(counts["R"].sum()),
            "total_N": int(counts["N"].sum()),
        }
    )
    return counts, metadata


@dataclass(frozen=True)
class ForecastConfig:
    """Controls model complexity, validation and interval behavior."""

    initial_window: int = 12
    rolling_step: int = 1
    validation_horizons: tuple[int, ...] = (1, 2, 3, 4)
    window: Literal["growing", "fixed"] = "growing"
    fixed_window: int | None = None
    holdout_size: int = 4
    seasonal_period: int = 4
    naive_seasonal: bool = False
    ets_trend: Literal["add", "none"] = "none"
    ets_damped_trend: bool = False
    ets_seasonal: Literal["add", "none"] = "none"
    ets_smoothing_mode: Literal["automatic", "manual"] = "automatic"
    ets_alpha: float = 0.3
    ets_beta: float = 0.1
    ets_gamma: float = 0.1
    arima_auto: bool = True
    arima_order: tuple[int, int, int] = (1, 0, 0)
    arima_p_max: int = 2
    arima_d_max: int = 1
    arima_q_max: int = 2
    sarima_auto: bool = True
    sarima_order: tuple[int, int, int] = (1, 0, 0)
    sarima_seasonal_order: tuple[int, int, int] = (0, 0, 0)
    sarima_p_seasonal_max: int = 1
    sarima_d_seasonal_max: int = 1
    sarima_q_seasonal_max: int = 1
    max_iterations: int = 100
    tolerance: float = 1e-6
    logit_transform: bool = True
    logit_epsilon: float = 1e-4
    max_candidates: int = 24
    max_order_complexity: int = 4
    glm_trend: Literal["none", "linear", "quadratic"] = "linear"
    glm_seasonal: bool = False
    glm_max_iterations: int = 100
    glm_tolerance: float = 1e-8
    future_n: int | Sequence[int] | None = None

    def __post_init__(self) -> None:
        if self.initial_window < 2:
            raise ValueError("initial_window must be at least 2")
        if self.rolling_step < 1 or self.holdout_size < 0:
            raise ValueError("rolling_step must be positive and holdout_size non-negative")
        if not self.validation_horizons or any(h < 1 for h in self.validation_horizons):
            raise ValueError("validation_horizons must contain positive integers")
        if self.window not in ("growing", "fixed"):
            raise ValueError("window must be 'growing' or 'fixed'")
        if self.window == "fixed" and (self.fixed_window is None or self.fixed_window < 2):
            raise ValueError("fixed_window >= 2 is required for a fixed window")
        if self.seasonal_period is not None and self.seasonal_period < 2:
            raise ValueError("seasonal_period must be at least 2")
        if not 0 < self.logit_epsilon < 0.5:
            raise ValueError("logit_epsilon must be between 0 and 0.5")
        if self.max_candidates < 1 or self.max_order_complexity < 0:
            raise ValueError("ARIMA candidate and complexity caps are invalid")
        if self.max_iterations < 1 or self.tolerance <= 0:
            raise ValueError("max_iterations and tolerance must be positive")
        if self.glm_max_iterations < 1 or self.glm_tolerance <= 0:
            raise ValueError("GLM iteration controls must be positive")
        if any(not 0 <= value <= 1 for value in (self.ets_alpha, self.ets_beta, self.ets_gamma)):
            raise ValueError("ETS smoothing parameters must be between 0 and 1")
        for order in (self.arima_order, self.sarima_order, self.sarima_seasonal_order):
            if len(order) != 3 or any(value < 0 for value in order):
                raise ValueError("ARIMA order values must be non-negative triples")


@dataclass
class _Fit:
    mean: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    interval_available: bool
    interval_reason: str | None
    lower_80: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    upper_80: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    residuals: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    model_info: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)


def _inverse_logit(values: np.ndarray) -> np.ndarray:
    """Numerically stable inverse logit without post-hoc clipping."""
    values = np.asarray(values, dtype=float)
    result = np.empty_like(values)
    positive = values >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_value = np.exp(values[~positive])
    result[~positive] = exp_value / (1.0 + exp_value)
    return result


def _prepare_data(data: pd.Series | pd.DataFrame) -> pd.DataFrame:
    if isinstance(data, pd.Series):
        frame = pd.DataFrame({"p_R": pd.to_numeric(data, errors="coerce")})
        frame["N"] = np.nan
        frame["R"] = np.nan
        frame["S"] = np.nan
        frame["I"] = np.nan
    elif isinstance(data, pd.DataFrame):
        frame = data.copy()
        if "p_R" not in frame:
            raise KeyError("Input DataFrame must contain a 'p_R' column")
        frame["p_R"] = pd.to_numeric(frame["p_R"], errors="coerce")
    else:
        raise TypeError("data must be a pandas Series or DataFrame")

    if not isinstance(frame.index, pd.PeriodIndex):
        raise TypeError("Input must have a quarterly pandas PeriodIndex")
    if frame.index.has_duplicates:
        raise ValueError("Quarterly input index must not contain duplicates")
    if len(frame) == 0:
        raise ValueError("At least one quarter is required")
    if not all(period.freqstr.startswith("Q") for period in frame.index):
        raise ValueError("Input PeriodIndex must use a quarterly frequency")
    frame = frame.sort_index()
    full_index = pd.period_range(frame.index.min(), frame.index.max(), freq="Q-DEC", name="quarter")
    return frame.reindex(full_index)


def _as_float_array(values: Any) -> np.ndarray:
    return np.asarray(values, dtype=float).reshape(-1)


def _transform_probability(values: np.ndarray, config: ForecastConfig) -> np.ndarray:
    if not config.logit_transform:
        return values.copy()
    epsilon = config.logit_epsilon
    adjusted = values * (1.0 - 2.0 * epsilon) + epsilon
    return np.log(adjusted / (1.0 - adjusted))


def _score_aicc(result: Any, n_observed: int) -> float:
    parameter_count = len(result.params)
    if n_observed <= parameter_count + 1:
        return math.inf
    return float(result.aic + (2 * parameter_count * (parameter_count + 1)) /
                 (n_observed - parameter_count - 1))


def _fit_arima_with_status(model: Any, max_iterations: int) -> tuple[Any, bool, list[str]]:
    """Fit ARIMA and retain convergence diagnostics instead of hiding warnings."""
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        result = model.fit(method_kwargs={"maxiter": max_iterations})
    convergence_warnings = [
        f"{warning.category.__name__}: {warning.message}"
        for warning in caught_warnings
        if warning.category.__name__ == "ConvergenceWarning"
        or "converg" in str(warning.message).lower()
    ]
    mle_retvals = getattr(result, "mle_retvals", {}) or {}
    converged = bool(mle_retvals.get("converged", True)) and not convergence_warnings
    return result, converged, convergence_warnings


def _candidate_orders(
    config: ForecastConfig, *, seasonal: bool
) -> list[tuple[tuple[int, int, int], tuple[int, int, int, int]]]:
    from itertools import product

    p_max, d_max, q_max = (
        config.arima_p_max,
        config.arima_d_max,
        config.arima_q_max,
    )
    nonseasonal = [
        (p, d, q)
        for p, d, q in product(range(p_max + 1), range(d_max + 1), range(q_max + 1))
        if p + d + q <= config.max_order_complexity
    ]
    period = config.seasonal_period or 4
    if seasonal:
        seasonal_orders = [
            (P, D, Q, period)
            for P, D, Q in product(
                range(config.sarima_p_seasonal_max + 1),
                range(config.sarima_d_seasonal_max + 1),
                range(config.sarima_q_seasonal_max + 1),
            )
            if P + D + Q <= 1
        ]
    else:
        seasonal_orders = [(0, 0, 0, 0)]
    orders = [
        (order, seasonal_order)
        for order in nonseasonal
        for seasonal_order in seasonal_orders
        if sum(order) + sum(seasonal_order[:3]) <= config.max_order_complexity
    ]
    orders.sort(
        key=lambda item: (
            sum(item[0]) + sum(item[1][:3]),
            item[0],
            item[1],
        )
    )
    return orders[: config.max_candidates]


def _fit_arima(
    train: pd.DataFrame,
    future_periods: pd.PeriodIndex,
    config: ForecastConfig,
    *,
    seasonal: bool,
) -> _Fit:
    from statsmodels.tsa.arima.model import ARIMA

    y = train["p_R"].to_numpy(dtype=float)
    model_y = _transform_probability(y, config)
    n_observed = int(np.isfinite(model_y).sum())
    if n_observed < 6:
        raise ValueError("ARIMA-family models need at least 6 observed quarters")

    failures: list[str] = []
    candidates: list[
        tuple[float, tuple[int, int, int], tuple[int, int, int, int], bool]
    ] = []
    candidate_convergence_warnings: list[str] = []
    nonconverged_candidate_count = 0
    automatic = config.sarima_auto if seasonal else config.arima_auto
    if automatic:
        orders = _candidate_orders(config, seasonal=seasonal)
        for order, seasonal_order in orders:
            try:
                model = ARIMA(
                    model_y,
                    order=order,
                    seasonal_order=seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                result, converged, convergence_warnings = _fit_arima_with_status(
                    model, config.max_iterations
                )
                if not converged:
                    nonconverged_candidate_count += 1
                candidate_convergence_warnings.extend(
                    f"{order}/{seasonal_order}: {message}"
                    for message in convergence_warnings
                )
                score = _score_aicc(result, n_observed)
                if math.isfinite(score):
                    candidates.append((score, order, seasonal_order, converged))
                else:
                    failures.append(
                        f"{order}/{seasonal_order}: insufficient AICc sample"
                    )
            except Exception as exc:  # individual bounded candidates may be inadmissible
                failures.append(f"{order}/{seasonal_order}: {type(exc).__name__}: {exc}")
        if not candidates:
            detail = "; ".join(failures[:3]) or "no admissible candidate"
            raise ValueError(f"No ARIMA candidate fitted successfully ({detail})")
        converged_candidates = [candidate for candidate in candidates if candidate[3]]
        selection_pool = converged_candidates or candidates
        _, chosen_order, chosen_seasonal_order, _ = min(
            selection_pool, key=lambda item: item[0]
        )
    elif seasonal:
        chosen_order = config.sarima_order
        chosen_seasonal_order = (*config.sarima_seasonal_order, config.seasonal_period)
    else:
        chosen_order = config.arima_order
        chosen_seasonal_order = (0, 0, 0, 0)
    model = ARIMA(
        model_y,
        order=chosen_order,
        seasonal_order=chosen_seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    result, converged, final_convergence_warnings = _fit_arima_with_status(
        model, config.max_iterations
    )
    prediction = result.get_forecast(steps=len(future_periods))
    mean = _as_float_array(prediction.predicted_mean)
    intervals_80 = np.asarray(prediction.conf_int(alpha=0.2), dtype=float)
    intervals_95 = np.asarray(prediction.conf_int(alpha=0.05), dtype=float)
    if config.logit_transform:
        mean = _inverse_logit(mean)
        lower_80 = _inverse_logit(intervals_80[:, 0])
        upper_80 = _inverse_logit(intervals_80[:, 1])
        lower = _inverse_logit(intervals_95[:, 0])
        upper = _inverse_logit(intervals_95[:, 1])
    else:
        lower_80, upper_80 = intervals_80[:, 0], intervals_80[:, 1]
        lower, upper = intervals_95[:, 0], intervals_95[:, 1]
    return _Fit(
        mean=mean,
        lower=lower,
        upper=upper,
        lower_80=lower_80,
        upper_80=upper_80,
        interval_available=True,
        interval_reason=None,
        residuals=_as_float_array(result.resid),
        model_info={
            "order": list(chosen_order),
            "seasonal_order": list(chosen_seasonal_order),
            "selection": (
                (
                    "minimum AICc among converged bounded candidates fitted on training data"
                    if converged_candidates
                    else "minimum AICc among bounded candidates; none converged"
                )
                if automatic
                else "manual orders"
            ),
            "candidate_count": len(candidates) if automatic else 1,
            "converged_candidate_count": (
                len(converged_candidates) if automatic else int(converged)
            ),
            "candidate_failures": failures,
            "nonconverged_candidate_count": nonconverged_candidate_count,
            "candidate_convergence_warnings": candidate_convergence_warnings[:10],
            "converged": converged,
            "convergence_warnings": final_convergence_warnings,
        },
    )


def _fit_glm(
    train: pd.DataFrame,
    future_periods: pd.PeriodIndex,
    config: ForecastConfig,
    *,
    global_origin: int,
    future_n: np.ndarray | None,
) -> _Fit:
    import statsmodels.api as sm

    required = ("R", "N")
    if any(column not in train for column in required):
        raise ValueError("binomial_glm requires quarterly R and N count columns")
    successes = pd.to_numeric(train["R"], errors="coerce").to_numpy(dtype=float)
    totals = pd.to_numeric(train["N"], errors="coerce").to_numpy(dtype=float)
    valid = (
        np.isfinite(successes)
        & np.isfinite(totals)
        & (totals > 0)
        & (successes >= 0)
        & (successes <= totals)
    )
    if valid.sum() < 5:
        raise ValueError("binomial_glm needs at least 5 quarters with valid S/I/R counts")
    periods = train.index
    train_x = _glm_design(
        periods[valid],
        global_origin,
        config.glm_trend,
        config.glm_seasonal,
    )
    endog = successes[valid] / totals[valid]
    model = sm.GLM(
        endog,
        train_x,
        family=sm.families.Binomial(),
        freq_weights=totals[valid],
    )
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        result = model.fit(
            maxiter=config.glm_max_iterations,
            tol=config.glm_tolerance,
        )
    convergence_warnings = [
        f"{warning.category.__name__}: {warning.message}"
        for warning in caught_warnings
        if warning.category.__name__ == "ConvergenceWarning"
        or "converg" in str(warning.message).lower()
    ]
    converged = bool(result.converged) and not convergence_warnings
    future_x = _glm_design(
        future_periods,
        global_origin,
        config.glm_trend,
        config.glm_seasonal,
    )
    linear = np.asarray(future_x @ result.params, dtype=float)
    covariance = np.asarray(result.cov_params(), dtype=float)
    standard_error = np.sqrt(np.maximum(0.0, np.einsum("ij,jk,ik->i", future_x, covariance, future_x)))
    mean = _inverse_logit(linear)
    z80 = 1.2815515655446004
    lower = _inverse_logit(linear - 1.96 * standard_error)
    upper = _inverse_logit(linear + 1.96 * standard_error)
    lower_80 = _inverse_logit(linear - z80 * standard_error)
    upper_80 = _inverse_logit(linear + z80 * standard_error)
    extra: dict[str, Any] = {}
    if future_n is not None:
        from scipy.stats import binom

        if len(future_n) != len(future_periods):
            raise ValueError("future_n must have one value per forecast quarter")
        if np.any(~np.isfinite(future_n)) or np.any(future_n < 0):
            raise ValueError("future_n values must be finite and non-negative")
        if np.any(future_n != np.floor(future_n)):
            raise ValueError("future_n values must be integers")
        n_values = future_n.astype(int)
        count_lower_80 = binom.ppf(0.1, n_values, mean)
        count_upper_80 = binom.ppf(0.9, n_values, mean)
        count_lower = binom.ppf(0.025, n_values, mean)
        count_upper = binom.ppf(0.975, n_values, mean)
        pred_lower_80 = np.divide(
            count_lower_80,
            n_values,
            out=np.full(len(n_values), np.nan),
            where=n_values > 0,
        )
        pred_upper_80 = np.divide(
            count_upper_80,
            n_values,
            out=np.full(len(n_values), np.nan),
            where=n_values > 0,
        )
        pred_lower = np.divide(
            count_lower, n_values, out=np.full(len(n_values), np.nan), where=n_values > 0
        )
        pred_upper = np.divide(
            count_upper, n_values, out=np.full(len(n_values), np.nan), where=n_values > 0
        )
        extra.update(
            {
                "future_n": n_values,
                "predictive_lower_80": pred_lower_80,
                "predictive_upper_80": pred_upper_80,
                "predictive_lower": pred_lower,
                "predictive_upper": pred_upper,
                "predictive_count_lower_80": count_lower_80,
                "predictive_count_upper_80": count_upper_80,
                "predictive_count_lower": count_lower,
                "predictive_count_upper": count_upper,
            }
        )
    dispersion = (
        float(result.pearson_chi2 / result.df_resid)
        if result.df_resid > 0
        else None
    )
    return _Fit(
        mean=mean,
        lower=lower,
        upper=upper,
        lower_80=lower_80,
        upper_80=upper_80,
        interval_available=True,
        interval_reason=None,
        residuals=_as_float_array(result.resid_pearson),
        model_info={
            "formula": (
                "logit(p_R) = intercept"
                + (f" + {config.glm_trend} quarter trend" if config.glm_trend != "none" else "")
                + (" + quarter-of-year indicators" if config.glm_seasonal else "")
            ),
            "dispersion_ratio": dispersion,
            "overdispersion_flag": dispersion is not None and dispersion > 1.5,
            "interval_kind": "95% Wald confidence interval for mean proportion",
            "converged": converged,
            "convergence_warnings": convergence_warnings,
        },
        extra=extra,
    )


def _glm_design(
    periods: pd.PeriodIndex,
    global_origin: int,
    trend_type: str,
    seasonal: bool,
) -> np.ndarray:
    trend = (
        np.asarray([period.ordinal for period in periods]) - global_origin
    ).astype(float)
    columns = [np.ones(len(periods))]
    if trend_type in ("linear", "quadratic"):
        trend = trend / 4.0
        columns.append(trend)
    if trend_type == "quadratic":
        columns.append(trend**2)
    if seasonal:
        quarter_numbers = np.asarray([period.quarter for period in periods])
        columns.extend((quarter_numbers == quarter).astype(float) for quarter in (2, 3, 4))
    return np.column_stack(columns)


def _fit_model(
    train: pd.DataFrame,
    future_periods: pd.PeriodIndex,
    method: str,
    config: ForecastConfig,
    *,
    global_origin: int,
    future_n: np.ndarray | None = None,
) -> _Fit:
    y = train["p_R"].to_numpy(dtype=float)
    if method == "naive":
        observed = np.flatnonzero(np.isfinite(y))
        if not len(observed):
            raise ValueError("No observed quarters in training window")
        season = config.seasonal_period if config.naive_seasonal else None
        if season is None:
            mean = np.repeat(y[observed[-1]], len(future_periods))
            model_name = "last observed proportion"
        else:
            if len(train) < season:
                raise ValueError(f"Seasonal naïve requires at least {season} calendar quarters")
            source = y[-season:]
            if not np.isfinite(source).all():
                raise ValueError(
                    "Seasonal naïve needs an observed value in each quarter of its final cycle"
                )
            mean = np.asarray(
                [source[(step - 1) % season] for step in range(1, len(future_periods) + 1)]
            )
            model_name = f"seasonal naïve (period={season})"
        return _Fit(
            mean=mean,
            lower=np.full(len(mean), np.nan),
            upper=np.full(len(mean), np.nan),
            interval_available=False,
            interval_reason="No calibrated prediction interval is implemented for naïve forecasts.",
            residuals=np.diff(y),
            model_info={"model": model_name},
        )

    if method == "ets":
        from statsmodels.tsa.holtwinters import ExponentialSmoothing

        if not np.isfinite(y).all():
            raise ValueError("ETS cannot fit missing quarters; no imputation is performed")
        period = (
            config.seasonal_period
            if config.ets_seasonal == "add"
            else None
        )
        if period is not None and len(y) < 2 * period:
            raise ValueError(f"Seasonal ETS requires at least {2 * period} observations")
        model = ExponentialSmoothing(
            _transform_probability(y, config),
            trend="add" if config.ets_trend == "add" else None,
            damped_trend=config.ets_damped_trend,
            seasonal="add" if period else None,
            seasonal_periods=period,
            initialization_method="estimated",
        )
        if config.ets_smoothing_mode == "automatic":
            result = model.fit(
                optimized=True,
                minimize_kwargs={
                    "options": {
                        "maxiter": config.max_iterations,
                        "ftol": config.tolerance,
                    }
                },
            )
        else:
            result = model.fit(
                smoothing_level=config.ets_alpha,
                smoothing_trend=(
                    config.ets_beta if config.ets_trend == "add" else None
                ),
                smoothing_seasonal=(
                    config.ets_gamma if config.ets_seasonal == "add" else None
                ),
                damping_trend=0.98 if config.ets_damped_trend else None,
                optimized=False,
            )
        transformed_mean = _as_float_array(result.forecast(len(future_periods)))
        mean = (
            _inverse_logit(transformed_mean)
            if config.logit_transform
            else transformed_mean
        )
        return _Fit(
            mean=mean,
            lower=np.full(len(mean), np.nan),
            upper=np.full(len(mean), np.nan),
            interval_available=False,
            interval_reason=(
                "statsmodels ExponentialSmoothing does not provide a reliable "
                "prediction interval through this implementation."
            ),
            residuals=_as_float_array(result.resid),
            model_info={
                "trend": config.ets_trend,
                "damped_trend": config.ets_damped_trend,
                "seasonal": "additive" if period else None,
                "smoothing_mode": config.ets_smoothing_mode,
                "smoothing_level": float(result.params["smoothing_level"]),
                "smoothing_trend": result.params.get("smoothing_trend"),
                "smoothing_seasonal": result.params.get("smoothing_seasonal"),
                "damping_trend": result.params.get("damping_trend"),
                "error_convention": (
                    "Additive observation errors (observed minus fitted) with "
                    "statsmodels Holt-Winters smoothing recursions; not a "
                    "state-space ETS(A, A, A) likelihood."
                ),
                "seasonal_period": period,
            },
        )

    if method in ("arima", "sarima"):
        return _fit_arima(
            train, future_periods, config, seasonal=(method == "sarima")
        )
    if method == "binomial_glm":
        return _fit_glm(
            train,
            future_periods,
            config,
            global_origin=global_origin,
            future_n=future_n,
        )
    raise ValueError(f"Unknown forecasting method: {method}")


def _future_n_values(
    config: ForecastConfig, horizon: int, explicit: int | Sequence[int] | None
) -> np.ndarray | None:
    source = explicit if explicit is not None else config.future_n
    if source is None:
        return None
    if np.isscalar(source):
        return np.repeat(float(source), horizon)
    values = np.asarray(list(source), dtype=float)
    if len(values) != horizon:
        raise ValueError("future_n must be a scalar or have one value per horizon step")
    return values


def _add_observed_proportion_intervals(
    fit: _Fit, future_n: np.ndarray | None
) -> None:
    """Add conditional binomial intervals for observed proportions at scenario N."""
    if future_n is None:
        return
    from scipy.stats import binom

    if len(future_n) != len(fit.mean):
        raise ValueError("future_n must have one value per forecast quarter")
    if np.any(~np.isfinite(future_n)) or np.any(future_n < 0):
        raise ValueError("future_n values must be finite and non-negative")
    if np.any(future_n != np.floor(future_n)):
        raise ValueError("future_n values must be integers")
    n_values = future_n.astype(int)
    lower_80_counts = binom.ppf(0.1, n_values, fit.mean)
    upper_80_counts = binom.ppf(0.9, n_values, fit.mean)
    lower_95_counts = binom.ppf(0.025, n_values, fit.mean)
    upper_95_counts = binom.ppf(0.975, n_values, fit.mean)
    fit.extra.update(
        {
            "future_n": n_values,
            "predictive_lower_80": np.divide(
                lower_80_counts,
                n_values,
                out=np.full(len(n_values), np.nan),
                where=n_values > 0,
            ),
            "predictive_upper_80": np.divide(
                upper_80_counts,
                n_values,
                out=np.full(len(n_values), np.nan),
                where=n_values > 0,
            ),
            "predictive_lower": np.divide(
                lower_95_counts,
                n_values,
                out=np.full(len(n_values), np.nan),
                where=n_values > 0,
            ),
            "predictive_upper": np.divide(
                upper_95_counts,
                n_values,
                out=np.full(len(n_values), np.nan),
                where=n_values > 0,
            ),
            "predictive_count_lower_80": lower_80_counts,
            "predictive_count_upper_80": upper_80_counts,
            "predictive_count_lower": lower_95_counts,
            "predictive_count_upper": upper_95_counts,
        }
    )


def _forecast_frame(
    fit: _Fit, periods: pd.PeriodIndex, future_n: np.ndarray | None
) -> pd.DataFrame:
    if fit.lower_80.size == len(periods):
        lower_80 = fit.lower_80
        upper_80 = fit.upper_80
    else:
        lower_80 = np.full(len(periods), np.nan)
        upper_80 = np.full(len(periods), np.nan)
    result = pd.DataFrame(
        {
            "quarter": periods.astype(str),
            "forecast": fit.mean,
            "lower_80": lower_80,
            "upper_80": upper_80,
            "lower": fit.lower,
            "upper": fit.upper,
            "interval_available": fit.interval_available,
            "interval_reason": fit.interval_reason,
        }
    )
    for key, column in (
        ("future_n", "future_n"),
        ("predictive_lower_80", "predictive_lower_80"),
        ("predictive_upper_80", "predictive_upper_80"),
        ("predictive_lower", "predictive_lower"),
        ("predictive_upper", "predictive_upper"),
        ("predictive_count_lower_80", "predictive_count_lower_80"),
        ("predictive_count_upper_80", "predictive_count_upper_80"),
        ("predictive_count_lower", "predictive_count_lower"),
        ("predictive_count_upper", "predictive_count_upper"),
    ):
        if key in fit.extra:
            result[column] = fit.extra[key]
    return result


def _diagnostics(fit: _Fit, train: pd.DataFrame, method: str) -> dict[str, Any]:
    from statsmodels.stats.diagnostic import acorr_ljungbox

    diagnostics: dict[str, Any] = {
        "residual_count": int(np.isfinite(fit.residuals).sum()),
        "ljung_box": None,
        "ljung_box_note": None,
        "overdispersion_ratio": fit.model_info.get("dispersion_ratio"),
    }
    residuals = fit.residuals
    has_gaps = train["p_R"].isna().any()
    if has_gaps:
        diagnostics["ljung_box_note"] = (
            "Not calculated: missing calendar quarters break the contiguous residual sequence."
        )
    else:
        residuals = residuals[np.isfinite(residuals)]
        if len(residuals) >= 8:
            lag = min(4, len(residuals) // 3)
            try:
                lb = acorr_ljungbox(residuals, lags=[lag], return_df=True).iloc[0]
                diagnostics["ljung_box"] = {
                    "lag": int(lag),
                    "statistic": float(lb["lb_stat"]),
                    "p_value": float(lb["lb_pvalue"]),
                }
            except Exception as exc:
                diagnostics["ljung_box_note"] = f"Unavailable: {type(exc).__name__}: {exc}"
        else:
            diagnostics["ljung_box_note"] = "At least 8 contiguous residuals are required."
    if method == "binomial_glm":
        ratio = diagnostics["overdispersion_ratio"]
        diagnostics["overdispersion_note"] = (
            "Pearson chi-square / residual degrees of freedom; values above 1.5 "
            "are flagged as possible overdispersion, not corrected."
        )
        diagnostics["overdispersion_flag"] = ratio is not None and ratio > 1.5
    return diagnostics


def forecast_quarterly(
    data: pd.Series | pd.DataFrame,
    *,
    method: Literal["naive", "ets", "arima", "sarima", "binomial_glm"],
    horizon: int = 4,
    config: ForecastConfig | None = None,
    future_n: int | Sequence[int] | None = None,
) -> dict[str, Any]:
    """Backtest and forecast one quarterly AMR proportion series.

    The last ``holdout_size`` calendar quarters are reserved for a final
    chronological test and excluded from rolling-origin tuning/validation.
    The final future forecast is then refit on the full available history.
    Rolling origins use only earlier data; the final holdout is never used for
    model fitting or ARIMA AICc selection.

    The result contains pandas DataFrames for convenient dashboard display.
    Use :func:`to_jsonable` to convert the complete result into JSON-compatible
    dictionaries/lists.
    """
    if horizon < 1:
        raise ValueError("horizon must be at least 1")
    cfg = config or ForecastConfig()
    frame = _prepare_data(data)
    method = str(method).lower()
    if method not in {"naive", "ets", "arima", "sarima", "binomial_glm"}:
        raise ValueError("method must be naive, ets, arima, sarima, or binomial_glm")
    if method == "binomial_glm" and not {"R", "N"}.issubset(frame.columns):
        raise ValueError("binomial_glm requires a DataFrame with R and N count columns")

    final_index = frame.index
    holdout_size = min(cfg.holdout_size, max(0, len(frame) - cfg.initial_window))
    split = len(frame) - holdout_size
    pre_holdout = frame.iloc[:split]
    holdout = frame.iloc[split:]
    global_origin = int(final_index[0].ordinal)
    failures: list[dict[str, Any]] = []

    holdout_frame = pd.DataFrame()
    if holdout_size:
        holdout_periods = pd.PeriodIndex(holdout.index, freq="Q-DEC", name="quarter")
        training = _window_training(pre_holdout, len(pre_holdout), cfg)
        try:
            holdout_future_n = (
                holdout["N"].to_numpy(dtype=float)
                if "N" in holdout and holdout["N"].notna().any()
                else None
            )
            holdout_fit = _fit_model(
                training,
                holdout_periods,
                method,
                cfg,
                global_origin=global_origin,
                future_n=holdout_future_n,
            )
            if method != "binomial_glm":
                _add_observed_proportion_intervals(
                    holdout_fit, holdout_future_n
                )
            holdout_frame = _forecast_frame(holdout_fit, holdout_periods, None)
            holdout_frame["horizon"] = np.arange(1, len(holdout_periods) + 1)
            holdout_frame["actual"] = holdout["p_R"].to_numpy(dtype=float)
            holdout_frame["N"] = (
                holdout["N"].to_numpy(dtype=float)
                if "N" in holdout
                else np.full(len(holdout), np.nan)
            )
            holdout_frame["error"] = holdout_frame["forecast"] - holdout_frame["actual"]
            holdout_frame["absolute_error"] = holdout_frame["error"].abs()
            holdout_frame["squared_error"] = holdout_frame["error"] ** 2
            holdout_fit_diagnostics = _diagnostics(holdout_fit, training, method)
        except Exception as exc:
            holdout_fit_diagnostics = {}
            failures.append(
                {"stage": "holdout", "error": f"{type(exc).__name__}: {exc}"}
            )
    else:
        holdout_fit_diagnostics = {}

    rolling_rows: list[dict[str, Any]] = []
    max_horizon = max(cfg.validation_horizons)
    rolling_end = split
    origin = cfg.initial_window
    while origin < rolling_end:
        available_horizon = min(max_horizon, rolling_end - origin)
        target_periods = pd.PeriodIndex(
            frame.index[origin : origin + available_horizon],
            freq="Q-DEC",
            name="quarter",
        )
        training = _window_training(frame.iloc[:origin], origin, cfg)
        try:
            fit = _fit_model(
                training,
                target_periods,
                method,
                cfg,
                global_origin=global_origin,
                future_n=(
                    frame.loc[target_periods, "N"].to_numpy(dtype=float)
                    if "N" in frame and frame["N"].notna().any()
                    else None
                ),
            )
            target_n = (
                frame.loc[target_periods, "N"].to_numpy(dtype=float)
                if "N" in frame and frame["N"].notna().any()
                else None
            )
            if method != "binomial_glm":
                _add_observed_proportion_intervals(fit, target_n)
            for step, period in enumerate(target_periods, start=1):
                if step not in cfg.validation_horizons:
                    continue
                actual = float(frame.loc[period, "p_R"]) if pd.notna(frame.loc[period, "p_R"]) else np.nan
                prediction = float(fit.mean[step - 1])
                rolling_rows.append(
                    {
                        "origin": str(frame.index[origin - 1]),
                        "quarter": str(period),
                        "horizon": step,
                        "actual": actual,
                        "forecast": prediction,
                        "error": prediction - actual if np.isfinite(actual) else np.nan,
                        "interval_available": fit.interval_available,
                        "lower": float(fit.lower[step - 1]) if fit.interval_available else np.nan,
                        "upper": float(fit.upper[step - 1]) if fit.interval_available else np.nan,
                        "lower_80": (
                            float(fit.lower_80[step - 1])
                            if fit.lower_80.size == len(target_periods)
                            else np.nan
                        ),
                        "upper_80": (
                            float(fit.upper_80[step - 1])
                            if fit.upper_80.size == len(target_periods)
                            else np.nan
                        ),
                        "predictive_lower_80": fit.extra.get(
                            "predictive_lower_80", [np.nan] * len(target_periods)
                        )[step - 1],
                        "predictive_upper_80": fit.extra.get(
                            "predictive_upper_80", [np.nan] * len(target_periods)
                        )[step - 1],
                        "predictive_lower": fit.extra.get(
                            "predictive_lower", [np.nan] * len(target_periods)
                        )[step - 1],
                        "predictive_upper": fit.extra.get(
                            "predictive_upper", [np.nan] * len(target_periods)
                        )[step - 1],
                    }
                )
        except Exception as exc:
            failures.append(
                {
                    "stage": "rolling_origin",
                    "origin": str(frame.index[origin - 1]),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        origin += cfg.rolling_step

    rolling = pd.DataFrame(rolling_rows)
    if rolling.empty:
        rolling = pd.DataFrame(
            columns=[
                "origin", "quarter", "horizon", "actual", "forecast", "error",
                "interval_available", "lower", "upper", "lower_80", "upper_80",
                "predictive_lower_80", "predictive_upper_80",
                "predictive_lower", "predictive_upper",
            ]
        )
    rolling_errors = pd.to_numeric(rolling["error"], errors="coerce")
    observed_errors = rolling.loc[np.isfinite(rolling_errors), "error"].to_numpy(dtype=float)
    validation_metrics = _error_metrics(observed_errors)
    validation_by_horizon = _horizon_metrics(rolling)
    holdout_errors = (
        holdout_frame.loc[np.isfinite(holdout_frame["error"]), "error"].to_numpy(dtype=float)
        if not holdout_frame.empty
        else np.array([], dtype=float)
    )
    holdout_metrics = _error_metrics(holdout_errors)
    holdout_by_horizon = _horizon_metrics(holdout_frame)

    future_periods = pd.period_range(
        final_index[-1] + 1, periods=horizon, freq="Q-DEC", name="quarter"
    )
    requested_future_n = _future_n_values(cfg, horizon, future_n)
    try:
        final_fit = _fit_model(
            frame,
            future_periods,
            method,
            cfg,
            global_origin=global_origin,
            future_n=requested_future_n,
        )
        if method != "binomial_glm":
            _add_observed_proportion_intervals(final_fit, requested_future_n)
        future = _forecast_frame(final_fit, future_periods, requested_future_n)
        final_model_info = final_fit.model_info
        diagnostics = _diagnostics(final_fit, frame, method)
        future_status = "ok"
        future_error = None
    except Exception as exc:
        final_fit = None
        future = pd.DataFrame(
            {
                "quarter": future_periods.astype(str),
                "forecast": np.nan,
                "lower": np.nan,
                "upper": np.nan,
                "lower_80": np.nan,
                "upper_80": np.nan,
                "interval_available": False,
                "interval_reason": None,
            }
        )
        final_model_info = {}
        diagnostics = {}
        future_status = "failed"
        future_error = f"{type(exc).__name__}: {exc}"
        failures.append({"stage": "final_fit", "error": future_error})

    return {
        "method": method,
        "status": future_status,
        "error": future_error,
        "config": asdict(cfg),
        "history": frame,
        "holdout": holdout_frame,
        "holdout_metrics": holdout_metrics,
        "holdout_by_horizon": holdout_by_horizon,
        "rolling_validation": rolling,
        "validation_metrics": validation_metrics,
        "validation_by_horizon": validation_by_horizon,
        "future": future,
        "model_info": final_model_info,
        "diagnostics": diagnostics,
        "holdout_diagnostics": holdout_fit_diagnostics,
        "failures": failures,
        "limitations": {
            "missing_quarters": (
                "Gaps remain NaN. ETS and seasonal naïve refuse incomplete training "
                "windows rather than filling or compressing time; ARIMA uses the "
                "quarterly positions as supplied."
            ),
            "logit_correction": (
                f"When enabled, p is transformed as logit(p*(1-2ε)+ε), "
                f"ε={cfg.logit_epsilon:g}; inverse-logit intervals are returned "
                "without post-hoc clipping."
            ),
            "intervals": (
                "Intervals are available for ARIMA/SARIMA from model forecast "
                "variance and for binomial GLM as mean-proportion Wald intervals. "
                "Naïve and statsmodels Holt-Winters ETS intervals are not reported."
            ),
            "holdout_refit": (
                "Final future forecasts are fit on all available quarters after "
                "the final holdout evaluation."
            ),
        },
    }


def _window_training(frame: pd.DataFrame, origin: int, config: ForecastConfig) -> pd.DataFrame:
    if config.window == "growing":
        return frame.copy()
    assert config.fixed_window is not None
    return frame.iloc[max(0, origin - config.fixed_window) :].copy()


def _error_metrics(errors: np.ndarray) -> dict[str, Any]:
    finite = errors[np.isfinite(errors)]
    if not len(finite):
        return {"n": 0, "MAE": None, "RMSE": None, "bias": None}
    return {
        "n": int(len(finite)),
        "MAE": float(np.mean(np.abs(finite))),
        "RMSE": float(np.sqrt(np.mean(np.square(finite)))),
        "bias": float(np.mean(finite)),
    }


def _horizon_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    """Return point and observed-proportion interval scores by lead time."""
    if frame.empty or "horizon" not in frame:
        return pd.DataFrame(
            columns=[
                "horizon", "n_forecasts", "MAE", "RMSE", "bias",
                "coverage_80", "mean_width_80", "interval_score_80",
                "coverage_95", "mean_width_95", "interval_score_95",
            ]
        )

    rows: list[dict[str, Any]] = []
    for horizon, group in frame.groupby("horizon", sort=True):
        actual = pd.to_numeric(group["actual"], errors="coerce").to_numpy(dtype=float)
        forecast = pd.to_numeric(group["forecast"], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(actual) & np.isfinite(forecast)
        errors = forecast[valid] - actual[valid]
        metrics = _error_metrics(errors)
        row: dict[str, Any] = {
            "horizon": int(horizon),
            "n_forecasts": int(len(group)),
            "n_evaluated": metrics["n"],
            "MAE": metrics["MAE"],
            "RMSE": metrics["RMSE"],
            "bias": metrics["bias"],
        }
        for level, alpha, lower_name, upper_name in (
            (80, 0.2, "predictive_lower_80", "predictive_upper_80"),
            (95, 0.05, "predictive_lower", "predictive_upper"),
        ):
            if lower_name not in group or upper_name not in group:
                row[f"coverage_{level}"] = None
                row[f"mean_width_{level}"] = None
                row[f"interval_score_{level}"] = None
                continue
            lower = pd.to_numeric(group[lower_name], errors="coerce").to_numpy(dtype=float)
            upper = pd.to_numeric(group[upper_name], errors="coerce").to_numpy(dtype=float)
            interval_valid = valid & np.isfinite(lower) & np.isfinite(upper)
            if not interval_valid.any():
                row[f"coverage_{level}"] = None
                row[f"mean_width_{level}"] = None
                row[f"interval_score_{level}"] = None
                continue
            observed = actual[interval_valid]
            lower = lower[interval_valid]
            upper = upper[interval_valid]
            below = np.maximum(lower - observed, 0)
            above = np.maximum(observed - upper, 0)
            scores = (upper - lower) + (2.0 / alpha) * (below + above)
            row[f"coverage_{level}"] = float(
                np.mean((observed >= lower) & (observed <= upper))
            )
            row[f"mean_width_{level}"] = float(np.mean(upper - lower))
            row[f"interval_score_{level}"] = float(np.mean(scores))
        rows.append(row)
    return pd.DataFrame(rows)


def to_jsonable(value: Any) -> Any:
    """Recursively convert pandas/numpy objects and non-finite values for JSON."""
    if isinstance(value, pd.DataFrame):
        return [to_jsonable(record) for record in value.to_dict(orient="records")]
    if isinstance(value, pd.Series):
        return [to_jsonable(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray, pd.Index)):
        return [to_jsonable(item) for item in list(value)]
    if isinstance(value, (pd.Period, pd.Timestamp)):
        return str(value)
    if value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


__all__ = [
    "ForecastConfig",
    "aggregate_quarterly_counts",
    "forecast_quarterly",
    "to_jsonable",
]
