"""Forecasting methods shared by the Streamlit dashboard and CLI."""

import math
import warnings


def _aicc(result, sample_size):
    parameter_count = len(result.params)
    if sample_size <= parameter_count + 1:
        return math.inf
    return float(
        result.aic
        + (2 * parameter_count * (parameter_count + 1))
        / (sample_size - parameter_count - 1)
    )


def _fit_arima(model):
    from statsmodels.tools.sm_exceptions import ConvergenceWarning

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fitted = model.fit(method_kwargs={"maxiter": 100})
    convergence_warnings = [
        str(item.message)
        for item in caught
        if issubclass(item.category, ConvergenceWarning)
        or "converg" in str(item.message).lower()
    ]
    converged = bool(getattr(fitted, "mle_retvals", {}).get("converged", True))
    return fitted, converged and not convergence_warnings, convergence_warnings


def _metrics_and_intervals(errors, forecast):
    mae = sum(abs(error) for error in errors) / len(errors)
    rmse = math.sqrt(sum(error**2 for error in errors) / len(errors))
    ci_lower = [
        round(max(0.0, prediction - 1.96 * rmse * math.sqrt(step)), 2)
        for step, prediction in enumerate(forecast, 1)
    ]
    ci_upper = [
        round(min(100.0, prediction + 1.96 * rmse * math.sqrt(step)), 2)
        for step, prediction in enumerate(forecast, 1)
    ]
    return {
        "metrics": {"MAE": round(mae, 2), "RMSE": round(rmse, 2)},
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
    }


def holt_linear_trend_forecasting(series, horizon=12, alpha=0.3, beta=0.1):
    """Forecast a bounded percentage series with Holt's linear trend method."""
    if len(series) < 3:
        raise ValueError("Série trop courte pour la prédiction Holt (minimum 3 points).")
    if horizon < 1:
        raise ValueError("L'horizon doit être d'au moins un mois.")
    if not 0 <= alpha <= 1 or not 0 <= beta <= 1:
        raise ValueError("Les paramètres alpha et beta doivent être entre 0 et 1.")

    level = float(series[0])
    trend = float(series[1]) - level
    fitted = [level]

    for value in series[1:]:
        previous_level = level
        previous_trend = trend
        level = alpha * value + (1 - alpha) * (previous_level + previous_trend)
        trend = beta * (level - previous_level) + (1 - beta) * previous_trend
        fitted.append(round(previous_level + previous_trend, 2))

    forecast = [
        round(max(0.0, min(100.0, level + step * trend)), 2)
        for step in range(1, horizon + 1)
    ]
    errors = [actual - predicted for actual, predicted in zip(series[1:], fitted[1:])]
    evaluation = _metrics_and_intervals(errors, forecast)

    return {
        "fitted": fitted,
        "forecast": forecast,
        **evaluation,
        "final_level": round(level, 2),
        "trend_slope_per_month": round(trend, 3),
        "trend_slope_per_year": round(trend * 12, 2),
        "model": "Holt",
    }


def naive_forecasting(series, horizon=12):
    """Forecast every future point with the latest observed value."""
    if len(series) < 2:
        raise ValueError("Série trop courte pour la prévision naïve (minimum 2 points).")
    if horizon < 1:
        raise ValueError("L'horizon doit être d'au moins un mois.")

    fitted = [series[0], *series[:-1]]
    forecast_value = max(0.0, min(100.0, float(series[-1])))
    forecast = [round(forecast_value, 2)] * horizon
    errors = [
        actual - predicted
        for actual, predicted in zip(series[1:], fitted[1:])
    ]
    evaluation = _metrics_and_intervals(errors, forecast)

    return {
        "fitted": fitted,
        "forecast": forecast,
        **evaluation,
        "trend_slope_per_month": None,
        "trend_slope_per_year": None,
        "model": "Naïve",
    }


def seasonal_naive_forecasting(series, horizon=12, season_length=12):
    """Repeat the last observed seasonal cycle for the forecast horizon."""
    if season_length < 1:
        raise ValueError("La période saisonnière doit être d'au moins un mois.")
    if len(series) <= season_length:
        raise ValueError(
            "La prévision naïve saisonnière nécessite au moins "
            f"{season_length + 1} observations pour calculer une erreur historique."
        )
    if horizon < 1:
        raise ValueError("L'horizon doit être d'au moins un mois.")

    fitted = [None] * season_length + list(series[:-season_length])
    errors = [
        actual - predicted
        for actual, predicted in zip(
            series[season_length:], fitted[season_length:]
        )
    ]
    forecast = [
        round(
            max(
                0.0,
                min(100.0, float(series[-season_length + (step % season_length)])),
            ),
            2,
        )
        for step in range(horizon)
    ]
    evaluation = _metrics_and_intervals(errors, forecast)

    return {
        "fitted": fitted,
        "forecast": forecast,
        **evaluation,
        "trend_slope_per_month": None,
        "trend_slope_per_year": None,
        "model": f"Naïve saisonnière ({season_length} mois)",
        "season_length": season_length,
    }


def holt_winters_forecasting(
    series,
    horizon=12,
    alpha=0.3,
    beta=0.1,
    gamma=0.1,
    season_length=12,
):
    """Forecast with additive Holt-Winters and an annual seasonal cycle."""
    if season_length < 1:
        raise ValueError("La période saisonnière doit être d'au moins un mois.")
    if len(series) < 3 * season_length:
        raise ValueError(
            "Holt-Winters nécessite au moins "
            f"{3 * season_length} observations mensuelles consécutives pour "
            "initialiser deux saisons et évaluer une troisième."
        )
    if horizon < 1:
        raise ValueError("L'horizon doit être d'au moins un mois.")
    if any(not 0 <= value <= 1 for value in (alpha, beta, gamma)):
        raise ValueError("Les paramètres alpha, beta et gamma doivent être entre 0 et 1.")

    first_season = [float(value) for value in series[:season_length]]
    second_season = [
        float(value) for value in series[season_length : 2 * season_length]
    ]
    first_average = sum(first_season) / season_length
    second_average = sum(second_season) / season_length
    level = first_average
    trend = (second_average - first_average) / season_length
    seasonal = [
        ((first_season[index] - first_average)
         + (second_season[index] - second_average))
        / 2
        for index in range(season_length)
    ]

    fitted = [None] * (2 * season_length)
    errors = []
    for index in range(2 * season_length, len(series)):
        seasonal_index = index % season_length
        previous_level = level
        previous_trend = trend
        predicted = level + trend + seasonal[seasonal_index]
        actual = float(series[index])
        fitted.append(round(max(0.0, min(100.0, predicted)), 2))
        errors.append(actual - predicted)

        level = alpha * (actual - seasonal[seasonal_index]) + (
            1 - alpha
        ) * (previous_level + previous_trend)
        trend = beta * (level - previous_level) + (1 - beta) * previous_trend
        seasonal[seasonal_index] = gamma * (actual - level) + (
            1 - gamma
        ) * seasonal[seasonal_index]

    forecast = [
        round(
            max(
                0.0,
                min(
                    100.0,
                    level + step * trend
                    + seasonal[(len(series) + step - 1) % season_length],
                ),
            ),
            2,
        )
        for step in range(1, horizon + 1)
    ]
    evaluation = _metrics_and_intervals(errors, forecast)

    return {
        "fitted": fitted,
        "forecast": forecast,
        **evaluation,
        "trend_slope_per_month": round(trend, 3),
        "trend_slope_per_year": round(trend * 12, 2),
        "model": f"Holt-Winters additif ({season_length} mois)",
        "alpha": alpha,
        "beta": beta,
        "gamma": gamma,
        "season_length": season_length,
    }


def optimize_holt_parameters(
    series,
    validation_horizon=12,
    step=0.05,
    min_train_size=12,
):
    """Select alpha and beta by expanding-window multi-step-ahead MAE."""
    if validation_horizon < 1:
        raise ValueError("L'horizon de validation doit être d'au moins un mois.")
    if len(series) < min_train_size + validation_horizon:
        raise ValueError(
            "Il faut au moins "
            f"{min_train_size + validation_horizon} observations pour optimiser "
            "les paramètres Holt avec cet horizon."
        )
    if not 0 < step <= 1:
        raise ValueError("Le pas de recherche doit être supérieur à 0 et au plus égal à 1.")

    candidates = [
        round(index * step, 4)
        for index in range(int(1 / step) + 1)
    ]
    if candidates[-1] < 1:
        candidates.append(1.0)
    origin_step = max(1, validation_horizon // 2)
    origins = list(
        range(
            min_train_size,
            len(series) - validation_horizon + 1,
            origin_step,
        )
    )
    best_result = None

    for alpha in candidates:
        for beta in candidates:
            absolute_errors = []
            for validation_origin in origins:
                fitted = holt_linear_trend_forecasting(
                    series[:validation_origin],
                    horizon=validation_horizon,
                    alpha=alpha,
                    beta=beta,
                )
                validation_values = series[
                    validation_origin : validation_origin + validation_horizon
                ]
                absolute_errors.extend(
                    abs(actual - predicted)
                    for actual, predicted in zip(
                        validation_values, fitted["forecast"]
                    )
                )

            mae = sum(absolute_errors) / len(absolute_errors)
            distance_from_defaults = abs(alpha - 0.35) + abs(beta - 0.12)
            candidate_result = {
                "alpha": alpha,
                "beta": beta,
                "mae": round(mae, 2),
                "validation_count": len(absolute_errors),
                "validation_origins": len(origins),
            }
            if best_result is None or mae < best_result["_raw_mae"] - 1e-9:
                best_result = {
                    **candidate_result,
                    "_raw_mae": mae,
                    "_default_distance": distance_from_defaults,
                }
            elif abs(mae - best_result["_raw_mae"]) <= 1e-9:
                if distance_from_defaults < best_result["_default_distance"]:
                    best_result = {
                        **candidate_result,
                        "_raw_mae": mae,
                        "_default_distance": distance_from_defaults,
                    }

    best_result.pop("_raw_mae")
    best_result.pop("_default_distance")
    return best_result


def optimize_holt_winters_parameters(
    series,
    validation_horizon=12,
    step=0.1,
    min_train_size=36,
    season_length=12,
):
    """Select Holt-Winters smoothing parameters by expanding-window MAE."""
    if validation_horizon < 1:
        raise ValueError("L'horizon de validation doit être d'au moins un mois.")
    if season_length < 1:
        raise ValueError("La période saisonnière doit être d'au moins un mois.")
    min_train_size = max(min_train_size, 3 * season_length)
    if len(series) < min_train_size + validation_horizon:
        raise ValueError(
            "Il faut au moins "
            f"{min_train_size + validation_horizon} observations pour optimiser "
            "les paramètres Holt-Winters avec cet horizon."
        )
    if not 0 < step <= 1:
        raise ValueError("Le pas de recherche doit être supérieur à 0 et au plus égal à 1.")

    candidates = [
        round(index * step, 4)
        for index in range(int(1 / step) + 1)
    ]
    if candidates[-1] < 1:
        candidates.append(1.0)
    origin_step = max(1, validation_horizon // 2)
    origins = list(
        range(
            min_train_size,
            len(series) - validation_horizon + 1,
            origin_step,
        )
    )
    best_result = None

    for alpha in candidates:
        for beta in candidates:
            for gamma in candidates:
                absolute_errors = []
                for validation_origin in origins:
                    fitted = holt_winters_forecasting(
                        series[:validation_origin],
                        horizon=validation_horizon,
                        alpha=alpha,
                        beta=beta,
                        gamma=gamma,
                        season_length=season_length,
                    )
                    validation_values = series[
                        validation_origin : validation_origin + validation_horizon
                    ]
                    absolute_errors.extend(
                        abs(actual - predicted)
                        for actual, predicted in zip(
                            validation_values, fitted["forecast"]
                        )
                    )

                mae = sum(absolute_errors) / len(absolute_errors)
                distance_from_defaults = (
                    abs(alpha - 0.3) + abs(beta - 0.1) + abs(gamma - 0.1)
                )
                candidate_result = {
                    "alpha": alpha,
                    "beta": beta,
                    "gamma": gamma,
                    "mae": round(mae, 2),
                    "validation_count": len(absolute_errors),
                    "validation_origins": len(origins),
                }
                if best_result is None or mae < best_result["_raw_mae"] - 1e-9:
                    best_result = {
                        **candidate_result,
                        "_raw_mae": mae,
                        "_default_distance": distance_from_defaults,
                    }
                elif abs(mae - best_result["_raw_mae"]) <= 1e-9:
                    if distance_from_defaults < best_result["_default_distance"]:
                        best_result = {
                            **candidate_result,
                            "_raw_mae": mae,
                            "_default_distance": distance_from_defaults,
                        }

    best_result.pop("_raw_mae")
    best_result.pop("_default_distance")
    return best_result


def arima_forecasting(
    series,
    horizon=12,
    order=(1, 1, 1),
    *,
    auto_order=True,
    p_max=2,
    d_max=1,
    q_max=2,
    max_candidates=24,
):
    """Forecast a bounded percentage series with a non-seasonal ARIMA model."""
    if len(series) < 6:
        raise ValueError("Série trop courte pour la prédiction ARIMA (minimum 6 points).")
    if horizon < 1:
        raise ValueError("L'horizon doit être d'au moins un mois.")

    try:
        from statsmodels.tsa.arima.model import ARIMA
    except ImportError as exc:
        raise ImportError(
            "ARIMA nécessite statsmodels. Installez les dépendances avec "
            "`pip install -r requirements.txt`."
        ) from exc

    if min(p_max, d_max, q_max) < 0 or max_candidates < 1:
        raise ValueError("Les bornes ARIMA et le budget de recherche sont invalides.")

    candidate_failures = []
    candidate_convergence_warnings = []
    chosen_order = tuple(order)
    if auto_order:
        candidates = []
        possible_orders = sorted(
            (
                (p, d, q)
                for p in range(p_max + 1)
                for d in range(d_max + 1)
                for q in range(q_max + 1)
                if p + d + q <= 4
            ),
            key=lambda item: (sum(item), item),
        )[:max_candidates]
        for candidate_order in possible_orders:
            try:
                fitted_candidate, converged, convergence_warnings = _fit_arima(
                    ARIMA(series, order=candidate_order)
                )
                candidate_convergence_warnings.extend(
                    f"{candidate_order}: {message}"
                    for message in convergence_warnings
                )
                score = _aicc(fitted_candidate, len(series))
                if math.isfinite(score):
                    candidates.append(
                        (score, candidate_order, converged, fitted_candidate)
                    )
                else:
                    candidate_failures.append(
                        f"{candidate_order}: échantillon trop court pour l’AICc"
                    )
            except Exception as exc:
                candidate_failures.append(
                    f"{candidate_order}: {type(exc).__name__}: {exc}"
                )
        if not candidates:
            detail = "; ".join(candidate_failures[:3]) or "aucun modèle admissible"
            raise ValueError(
                f"Aucune configuration ARIMA n’a pu être estimée ({detail})."
            )
        converged_candidates = [item for item in candidates if item[2]]
        selection_pool = converged_candidates or candidates
        _, chosen_order, final_converged, fitted_result = min(
            selection_pool, key=lambda item: item[0]
        )
        final_convergence_warnings = []
    else:
        fitted_result, final_converged, final_convergence_warnings = _fit_arima(
            ARIMA(series, order=chosen_order)
        )
    forecast_result = fitted_result.get_forecast(steps=horizon)
    forecast = forecast_result.predicted_mean.tolist()
    confidence = forecast_result.conf_int(alpha=0.05)
    ci_lower = confidence[:, 0].tolist()
    ci_upper = confidence[:, 1].tolist()

    fitted_values = fitted_result.fittedvalues.tolist()
    errors = [
        actual - predicted
        for actual, predicted in zip(series, fitted_values)
        if math.isfinite(float(predicted))
    ]
    if not errors:
        raise ValueError("ARIMA n'a pas produit de valeurs ajustées exploitables.")

    mae = sum(abs(error) for error in errors) / len(errors)
    rmse = math.sqrt(sum(error**2 for error in errors) / len(errors))

    return {
        "fitted": fitted_values,
        "forecast": [max(0.0, min(100.0, float(value))) for value in forecast],
        "ci_lower": [max(0.0, min(100.0, float(value))) for value in ci_lower],
        "ci_upper": [max(0.0, min(100.0, float(value))) for value in ci_upper],
        "metrics": {"MAE": mae, "RMSE": rmse},
        "trend_slope_per_month": None,
        "trend_slope_per_year": None,
        "model": f"ARIMA{chosen_order}",
        "order": chosen_order,
        "selection": (
            "AICc sur la série d’apprentissage"
            if auto_order
            else "Ordre manuel; coefficients estimés par maximum de vraisemblance"
        ),
        "candidate_count": len(possible_orders) if auto_order else 1,
        "candidate_failures": candidate_failures,
        "candidate_convergence_warnings": candidate_convergence_warnings[:10],
        "converged": final_converged,
        "convergence_warnings": final_convergence_warnings,
    }


def sarima_forecasting(
    series,
    horizon=12,
    order=(1, 0, 0),
    seasonal_order=(0, 0, 0, 12),
    *,
    auto_order=True,
    p_max=2,
    d_max=1,
    q_max=2,
    seasonal_p_max=1,
    seasonal_d_max=1,
    seasonal_q_max=1,
    max_candidates=24,
):
    """Forecast monthly percentages with a bounded seasonal ARIMA search."""
    if len(series) < 18:
        raise ValueError(
            "SARIMA mensuel nécessite au moins 18 observations pour estimer "
            "une saisonnalité annuelle."
        )
    if horizon < 1:
        raise ValueError("L'horizon doit être d'au moins un mois.")
    if min(p_max, d_max, q_max, seasonal_p_max, seasonal_d_max, seasonal_q_max) < 0:
        raise ValueError("Les bornes SARIMA doivent être positives ou nulles.")
    if max_candidates < 1:
        raise ValueError("Le budget de recherche SARIMA doit être positif.")

    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
    except ImportError as exc:
        raise ImportError(
            "SARIMA nécessite statsmodels. Installez les dépendances avec "
            "`pip install -r requirements.txt`."
        ) from exc

    values = [float(value) for value in series]
    if any(not math.isfinite(value) or value < 0 or value > 100 for value in values):
        raise ValueError("Les proportions SARIMA doivent être comprises entre 0 et 100.")
    epsilon = 1e-4
    probabilities = [min(1 - epsilon, max(epsilon, value / 100)) for value in values]
    transformed = [math.log(value / (1 - value)) for value in probabilities]

    candidate_failures = []
    possible_orders = [(tuple(order), tuple(seasonal_order))]
    if auto_order:
        candidates = sorted(
            (
                (p + d + q + P + D + Q, (p, d, q), (P, D, Q, 12))
                for p in range(p_max + 1)
                for d in range(d_max + 1)
                for q in range(q_max + 1)
                for P in range(seasonal_p_max + 1)
                for D in range(seasonal_d_max + 1)
                for Q in range(seasonal_q_max + 1)
                if p + d + q + P + D + Q <= 4
            ),
            key=lambda item: (item[0], item[1], item[2]),
        )
        possible_orders = [
            (order_values, seasonal_values)
            for _, order_values, seasonal_values in candidates[:max_candidates]
        ]

    fitted_candidates = []
    for candidate_order, candidate_seasonal_order in possible_orders:
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                result = SARIMAX(
                    transformed,
                    order=candidate_order,
                    seasonal_order=candidate_seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                ).fit(disp=False, maxiter=100)
            convergence_warnings = [
                str(item.message)
                for item in caught
                if "converg" in str(item.message).lower()
            ]
            converged = (
                bool(getattr(result, "mle_retvals", {}).get("converged", True))
                and not convergence_warnings
            )
            score = _aicc(result, len(values))
            if math.isfinite(score):
                fitted_candidates.append(
                    (
                        score,
                        candidate_order,
                        candidate_seasonal_order,
                        result,
                        converged,
                        convergence_warnings,
                    )
                )
            else:
                candidate_failures.append(
                    f"{candidate_order}x{candidate_seasonal_order}: AICc indisponible"
                )
        except Exception as exc:
            candidate_failures.append(
                f"{candidate_order}x{candidate_seasonal_order}: "
                f"{type(exc).__name__}: {exc}"
            )
    if not fitted_candidates:
        detail = "; ".join(candidate_failures[:3]) or "aucun modèle admissible"
        raise ValueError(f"Aucune configuration SARIMA n’a pu être estimée ({detail}).")

    converged_candidates = [item for item in fitted_candidates if item[4]]
    selected_pool = converged_candidates or fitted_candidates
    (
        _,
        chosen_order,
        chosen_seasonal_order,
        fitted_result,
        converged,
        convergence_warnings,
    ) = min(
        selected_pool, key=lambda item: item[0]
    )
    predicted = fitted_result.get_forecast(steps=horizon)
    transformed_mean = predicted.predicted_mean
    confidence = predicted.conf_int(alpha=0.05)
    inverse_logit = lambda value: 100 / (1 + math.exp(-max(-700, min(700, float(value)))))
    forecast = [inverse_logit(value) for value in transformed_mean]
    ci_lower = [inverse_logit(value) for value in confidence[:, 0]]
    ci_upper = [inverse_logit(value) for value in confidence[:, 1]]
    fitted = [inverse_logit(value) for value in fitted_result.fittedvalues]
    errors = [
        actual - predicted_value
        for actual, predicted_value in zip(values, fitted)
        if math.isfinite(float(predicted_value))
    ]
    if not errors:
        raise ValueError("SARIMA n'a pas produit de valeurs ajustées exploitables.")

    return {
        "fitted": fitted,
        "forecast": forecast,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "metrics": {
            "MAE": sum(abs(error) for error in errors) / len(errors),
            "RMSE": math.sqrt(sum(error**2 for error in errors) / len(errors)),
        },
        "trend_slope_per_month": None,
        "trend_slope_per_year": None,
        "model": f"SARIMA{chosen_order}x{chosen_seasonal_order}",
        "order": chosen_order,
        "seasonal_order": chosen_seasonal_order,
        "selection": "AICc sur la série d’apprentissage" if auto_order else "Ordre manuel",
        "candidate_count": len(possible_orders),
        "candidate_failures": candidate_failures,
        "converged": converged,
        "convergence_warnings": convergence_warnings,
    }


def monthly_binomial_glm_forecasting(
    resistant_counts,
    tested_counts,
    months,
    horizon=12,
    *,
    seasonal=True,
    future_n=None,
):
    """Fit a grouped-binomial monthly trend and forecast resistance proportions."""
    import numpy as np

    try:
        import statsmodels.api as sm
        from scipy.stats import binom
    except ImportError as exc:
        raise ImportError(
            "La régression binomiale nécessite statsmodels et scipy."
        ) from exc

    resistant = np.asarray(resistant_counts, dtype=float)
    tested = np.asarray(tested_counts, dtype=float)
    month_periods = [str(value) for value in months]
    if horizon < 1 or not (len(resistant) == len(tested) == len(month_periods)):
        raise ValueError("Les comptes mensuels, mois et horizon sont incohérents.")
    usable = (
        np.isfinite(resistant)
        & np.isfinite(tested)
        & (tested > 0)
        & (resistant >= 0)
        & (resistant <= tested)
    )
    if usable.sum() < (18 if seasonal else 6):
        minimum = 18 if seasonal else 6
        raise ValueError(
            f"Le GLM binomial mensuel nécessite au moins {minimum} mois "
            "avec des résultats interprétables."
        )
    ordinals = np.asarray(
        [int(value[:4]) * 12 + int(value[5:7]) - 1 for value in month_periods],
        dtype=float,
    )
    origin = ordinals[0]
    trend = (ordinals - origin) / 12
    columns = [np.ones(len(trend)), trend]
    if seasonal:
        month_numbers = np.asarray([int(value[5:7]) for value in month_periods])
        columns.extend((month_numbers == month).astype(float) for month in range(2, 13))
    design = np.column_stack(columns)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fitted_result = sm.GLM(
            resistant[usable] / tested[usable],
            design[usable],
            family=sm.families.Binomial(),
            var_weights=tested[usable],
        ).fit(maxiter=100, tol=1e-8)
    fit_warnings = [
        f"{item.category.__name__}: {item.message}" for item in caught
    ]

    last_ordinal = int(ordinals[-1])
    future_ordinals = np.arange(last_ordinal + 1, last_ordinal + horizon + 1)
    future_month_numbers = (future_ordinals % 12) + 1
    future_trend = (future_ordinals - origin) / 12
    future_columns = [np.ones(horizon), future_trend]
    if seasonal:
        future_columns.extend(
            (future_month_numbers == month).astype(float) for month in range(2, 13)
        )
    future_design = np.column_stack(future_columns)
    prediction = fitted_result.get_prediction(future_design)
    link_mean = np.asarray(prediction.predicted_mean, dtype=float)
    forecast = np.clip(link_mean, 0, 1) * 100
    confidence = np.asarray(prediction.conf_int(), dtype=float)
    ci_lower = np.clip(confidence[:, 0], 0, 1) * 100
    ci_upper = np.clip(confidence[:, 1], 0, 1) * 100
    n_future = (
        np.repeat(float(np.median(tested[usable])), horizon)
        if future_n is None
        else np.repeat(float(future_n), horizon)
        if np.isscalar(future_n)
        else np.asarray(future_n, dtype=float)
    )
    if len(n_future) != horizon or np.any(n_future < 1):
        raise ValueError("L’effectif futur doit contenir un entier positif par mois.")
    n_future = np.rint(n_future).astype(int)
    predictive_low = binom.ppf(0.025, n_future, forecast / 100) / n_future * 100
    predictive_high = binom.ppf(0.975, n_future, forecast / 100) / n_future * 100
    fitted_values = np.full(len(resistant), np.nan)
    fitted_values[usable] = (
        np.asarray(fitted_result.fittedvalues, dtype=float) * 100
    )
    observed = resistant[usable] / tested[usable] * 100
    errors = observed - fitted_values[usable]
    return {
        "fitted": fitted_values.tolist(),
        "forecast": forecast.tolist(),
        "ci_lower": ci_lower.tolist(),
        "ci_upper": ci_upper.tolist(),
        "predictive_lower": predictive_low.tolist(),
        "predictive_upper": predictive_high.tolist(),
        "future_n": n_future.tolist(),
        "metrics": {
            "MAE": float(np.mean(np.abs(errors))),
            "RMSE": float(np.sqrt(np.mean(errors**2))),
        },
        "trend_slope_per_month": float(fitted_result.params[1] * 100 / 12),
        "trend_slope_per_year": float(fitted_result.params[1] * 100),
        "model": "GLM binomial mensuel",
        "converged": bool(fitted_result.converged),
        "fit_warnings": fit_warnings,
        "dispersion": float(
            fitted_result.pearson_chi2 / max(1, fitted_result.df_resid)
        ),
        "seasonal_effect": seasonal,
    }


def forecast_time_series(
    series,
    horizon=12,
    method="holt",
    alpha=0.3,
    beta=0.1,
    gamma=0.1,
    season_length=12,
    arima_order=(1, 1, 1),
    arima_auto=True,
    arima_p_max=2,
    arima_d_max=1,
    arima_q_max=2,
    arima_max_candidates=24,
    sarima_order=(1, 0, 0),
    sarima_seasonal_order=(0, 0, 0, 12),
    sarima_auto=True,
    sarima_p_max=2,
    sarima_d_max=1,
    sarima_q_max=2,
    sarima_seasonal_p_max=1,
    sarima_seasonal_d_max=1,
    sarima_seasonal_q_max=1,
    monthly_counts=None,
    months=None,
    glm_seasonal=True,
    future_n=None,
):
    """Dispatch to a forecasting method while keeping a consistent result shape."""
    normalized_method = method.casefold()
    if normalized_method == "holt":
        return holt_linear_trend_forecasting(
            series, horizon=horizon, alpha=alpha, beta=beta
        )
    if normalized_method == "arima":
        return arima_forecasting(
            series,
            horizon=horizon,
            order=arima_order,
            auto_order=arima_auto,
            p_max=arima_p_max,
            d_max=arima_d_max,
            q_max=arima_q_max,
            max_candidates=arima_max_candidates,
        )
    if normalized_method == "sarima":
        return sarima_forecasting(
            series,
            horizon=horizon,
            order=sarima_order,
            seasonal_order=sarima_seasonal_order,
            auto_order=sarima_auto,
            p_max=sarima_p_max,
            d_max=sarima_d_max,
            q_max=sarima_q_max,
            seasonal_p_max=sarima_seasonal_p_max,
            seasonal_d_max=sarima_seasonal_d_max,
            seasonal_q_max=sarima_seasonal_q_max,
        )
    if normalized_method in ("binomial_glm", "glm"):
        if monthly_counts is None or months is None:
            raise ValueError("Le GLM binomial nécessite les comptes S/R et les mois.")
        return monthly_binomial_glm_forecasting(
            monthly_counts["R"],
            monthly_counts["N"],
            months,
            horizon=horizon,
            seasonal=glm_seasonal,
            future_n=future_n,
        )
    if normalized_method in ("holt_winters", "holt-winters"):
        return holt_winters_forecasting(
            series,
            horizon=horizon,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            season_length=season_length,
        )
    if normalized_method == "naive":
        return naive_forecasting(series, horizon=horizon)
    if normalized_method in ("seasonal_naive", "naive_seasonal"):
        return seasonal_naive_forecasting(series, horizon=horizon)
    raise ValueError(f"Méthode de prévision inconnue : {method}")


def compare_forecasting_methods(
    series,
    horizon=12,
    alpha=0.35,
    beta=0.12,
    gamma=0.1,
    seasonal_data_contiguous=True,
    holt_winters_alpha=None,
    holt_winters_beta=None,
    holt_winters_gamma=None,
    monthly_counts=None,
    months=None,
):
    """Compare supported methods on one chronological holdout window."""
    if horizon < 1:
        raise ValueError("L'horizon de test doit être d'au moins un mois.")
    if len(series) <= horizon:
        raise ValueError(
            "Il faut au moins une observation d'entraînement en plus "
            "de la période de test."
        )

    train = series[:-horizon]
    test = series[-horizon:]
    holt_winters_alpha = alpha if holt_winters_alpha is None else holt_winters_alpha
    holt_winters_beta = beta if holt_winters_beta is None else holt_winters_beta
    holt_winters_gamma = gamma if holt_winters_gamma is None else holt_winters_gamma
    methods = [
        (
            "Prévision naïve",
            "naive",
            {},
        ),
        (
            "ETS / Holt-Winters",
            "holt_winters",
            {
                "alpha": holt_winters_alpha,
                "beta": holt_winters_beta,
                "gamma": holt_winters_gamma,
            },
        ),
        ("ARIMA", "arima", {}),
        ("SARIMA", "sarima", {}),
        ("Régression binomiale temporelle", "binomial_glm", {}),
    ]
    results = []

    if not seasonal_data_contiguous:
        return [
            {
                "Méthode": label,
                "MAE test (%)": None,
                "RMSE test (%)": None,
                "Observations test": len(test),
                "Statut": (
                    "Non calculée : la série mensuelle contient des mois manquants"
                ),
            }
            for label, _, _ in methods
        ]

    for label, method, parameters in methods:
        row = {
            "Méthode": label,
            "MAE test (%)": None,
            "RMSE test (%)": None,
            "Observations test": len(test),
            "Statut": "Calculée",
        }
        try:
            prediction = forecast_time_series(
                train,
                horizon=horizon,
                method=method,
                monthly_counts=(
                    monthly_counts.iloc[:-horizon].copy()
                    if monthly_counts is not None
                    else None
                ),
                months=months[:-horizon] if months is not None else None,
                future_n=(
                    max(
                        1,
                        int(
                            round(
                                monthly_counts.iloc[:-horizon]["N"].median()
                            )
                        ),
                    )
                    if method == "binomial_glm" and monthly_counts is not None
                    else None
                ),
                **parameters,
            )["forecast"]
        except (ImportError, ValueError) as error:
            row["Statut"] = f"Non calculée : {error}"
            results.append(row)
            continue

        errors = [
            float(actual) - float(predicted)
            for actual, predicted in zip(test, prediction)
        ]
        mae = sum(abs(error) for error in errors) / len(errors)
        rmse = math.sqrt(sum(error**2 for error in errors) / len(errors))
        row["MAE test (%)"] = round(mae, 2)
        row["RMSE test (%)"] = round(rmse, 2)
        results.append(row)

    return results
