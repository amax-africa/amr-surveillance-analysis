"""Command-line monthly forecasts using the dashboard's shared data and models."""

import argparse
import math
from pathlib import Path

from dashboard_data import MIN_MONTHLY_TESTS, build_dashboard_data, extract_monthly_series
from forecasting import (
    forecast_time_series,
    holt_linear_trend_forecasting as holt_linear_trend_forecasting_shared,
)

PROJECT_DIR = Path(__file__).resolve().parent
CSV_PATH = PROJECT_DIR / "data.parquet"
METHODS = ("naive", "holt", "holt_winters", "arima", "sarima", "glm")


def _parse_order(value, length, option_name):
    try:
        order = tuple(int(part.strip()) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"{option_name} doit contenir des nombres séparés par des virgules."
        ) from exc
    if len(order) != length or any(part < 0 for part in order):
        raise argparse.ArgumentTypeError(
            f"{option_name} doit contenir exactement {length} entiers positifs ou nuls."
        )
    return order


def holt_linear_trend_forecasting(series, horizon=12, alpha=0.3, beta=0.1):
    """Keep the original CLI helper while delegating to the shared model."""
    result = holt_linear_trend_forecasting_shared(
        series, horizon=horizon, alpha=alpha, beta=beta
    )
    return {
        **result,
        "fitted": [round(value, 2) for value in result["fitted"]],
        "forecast": [round(value, 2) for value in result["forecast"]],
        "ci_lower": [round(value, 2) for value in result["ci_lower"]],
        "ci_upper": [round(value, 2) for value in result["ci_upper"]],
    }

def extract_monthly_time_series(
    csv_path, bacteria_name, abx_col, min_tests=MIN_MONTHLY_TESTS
):
    """Extract one exact bacterium/antibiotic series via shared CSV aggregation."""
    data = build_dashboard_data(csv_path, include_forecasts=False)
    antibiotic = next(
        (
            item
            for item in data["antibiotics"]
            if item.casefold() == abx_col.casefold()
        ),
        None,
    )
    if antibiotic is None:
        raise ValueError(f"Antibiotique introuvable : {abx_col}")
    bacterium = next(
        (
            item
            for item in data["bacteria"]
            if item.casefold() == bacteria_name.casefold()
        ),
        None,
    )
    if bacterium is None:
        raise ValueError(f"Bactérie introuvable : {bacteria_name}")
    return extract_monthly_series(data, bacterium, antibiotic, min_tests=min_tests)


def _future_month_labels(last_month, horizon):
    year, month = (int(part) for part in last_month.split("-"))
    labels = []
    for _ in range(horizon):
        month += 1
        if month == 13:
            month = 1
            year += 1
        labels.append(f"{year:04d}-{month:02d}")
    return labels


def _monthly_count_arguments(data, bacterium, antibiotic, dates):
    grouped = {}
    for row_bacterium, row_antibiotic, month, susceptible, resistant in data[
        "cube_monthly"
    ]:
        if row_bacterium == bacterium and row_antibiotic == antibiotic:
            grouped[month] = (susceptible, resistant)
    return {
        "S": [grouped[month][0] for month in dates],
        "R": [grouped[month][1] for month in dates],
        "N": [sum(grouped[month]) for month in dates],
    }


def run_prediction_pipeline(
    bacteria,
    abx,
    horizon=12,
    method="holt",
    csv_path=CSV_PATH,
    min_tests=MIN_MONTHLY_TESTS,
    arima_auto=True,
    arima_order=(1, 1, 1),
    sarima_auto=True,
    sarima_order=(1, 0, 0),
    sarima_seasonal_order=(0, 0, 0, 12),
):
    data = build_dashboard_data(csv_path, include_forecasts=False)
    actual_bacterium = next(
        (name for name in data["bacteria"] if name.casefold() == bacteria.casefold()),
        None,
    )
    actual_antibiotic = next(
        (name for name in data["antibiotics"] if name.casefold() == abx.casefold()),
        None,
    )
    if actual_bacterium is None:
        raise ValueError(f"Bactérie introuvable : {bacteria}")
    if actual_antibiotic is None:
        raise ValueError(f"Antibiotique introuvable : {abx}")

    dates, pct_r, n_tests = extract_monthly_series(
        data, actual_bacterium, actual_antibiotic, min_tests=min_tests
    )
    if not dates:
        raise ValueError(f"Aucune donnée trouvée pour {bacteria} × {abx}.")
    if horizon < 1:
        raise ValueError("L’horizon doit être supérieur ou égal à 1 mois.")

    method_arguments = {
        "arima_auto": arima_auto,
        "arima_order": arima_order,
        "sarima_auto": sarima_auto,
        "sarima_order": sarima_order,
        "sarima_seasonal_order": sarima_seasonal_order,
    }
    if method == "glm":
        method_arguments.update(
            months=dates,
            monthly_counts=_monthly_count_arguments(
                data, actual_bacterium, actual_antibiotic, dates
            ),
        )
    result = forecast_time_series(
        pct_r, horizon=horizon, method=method, **method_arguments
    )
    future_dates = _future_month_labels(dates[-1], horizon)

    print("=" * 68)
    print("PRÉVISION MENSUELLE DE RÉSISTANCE ANTIMICROBIENNE")
    print(f"Bactérie : {actual_bacterium}")
    print(f"Antibiotique : {actual_antibiotic}")
    print(f"Méthode : {result['model']}")
    print(f"Historique : {len(dates)} mois retenus ({dates[0]} à {dates[-1]})")
    print(f"Tests S/R : {sum(n_tests):,} ; moyenne %R : {sum(pct_r) / len(pct_r):.2f}%")
    print(f"Erreur ajustée MAE / RMSE : {result['metrics']['MAE']:.2f}% / {result['metrics']['RMSE']:.2f}%")

    test_size = min(12, max(0, len(pct_r) - 3))
    if test_size:
        try:
            train = pct_r[:-test_size]
            actual = pct_r[-test_size:]
            backtest_arguments = method_arguments.copy()
            if method == "glm":
                backtest_arguments.update(
                    months=dates[:-test_size],
                    monthly_counts={
                        key: values[:-test_size]
                        for key, values in method_arguments["monthly_counts"].items()
                    },
                )
            backtest = forecast_time_series(
                train,
                horizon=test_size,
                method=method,
                **backtest_arguments,
            )
            errors = [
                observed - predicted
                for observed, predicted in zip(actual, backtest["forecast"])
            ]
            mae = sum(abs(error) for error in errors) / len(errors)
            rmse = math.sqrt(sum(error**2 for error in errors) / len(errors))
            print(
                f"Validation chronologique ({test_size} mois finaux) "
                f"MAE / RMSE : {mae:.2f}% / {rmse:.2f}%"
            )
        except (ValueError, ImportError) as exc:
            print(f"Validation indisponible pour cette méthode : {exc}")

    print("\nMois       Prévision %R    Intervalle à 95%")
    for month, prediction, lower, upper in zip(
        future_dates, result["forecast"], result["ci_lower"], result["ci_upper"]
    ):
        print(f"{month:<10} {prediction:>8.1f}%       [{lower:.1f}% – {upper:.1f}%]")
    print("=" * 68)
    return {
        "bacteria": actual_bacterium,
        "abx": actual_antibiotic,
        "dates": dates,
        "historical_pct_r": pct_r,
        "historical_n": n_tests,
        "future_dates": future_dates,
        "forecast": result["forecast"],
        "ci_lower": result["ci_lower"],
        "ci_upper": result["ci_upper"],
        "metrics": result["metrics"],
        "trend_slope_per_year": result["trend_slope_per_year"],
        "model": result["model"],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Prévision mensuelle AMR avec les modèles partagés du dashboard."
    )
    parser.add_argument("bacterie", nargs="?", default="Klebsiella pneumoniae")
    parser.add_argument("antibiotique", nargs="?", default="Cefotaxime")
    parser.add_argument("--horizon", type=int, default=12, help="Mois à prévoir.")
    parser.add_argument("--min-tests", type=int, default=MIN_MONTHLY_TESTS)
    parser.add_argument("--method", choices=METHODS, default="holt")
    parser.add_argument("--csv", type=Path, default=CSV_PATH)
    parser.add_argument(
        "--arima-order",
        type=lambda value: _parse_order(value, 3, "--arima-order"),
        default=(1, 1, 1),
        help="Ordre p,d,q si --no-arima-auto est fourni.",
    )
    parser.add_argument("--no-arima-auto", dest="arima_auto", action="store_false")
    parser.add_argument(
        "--sarima-order",
        type=lambda value: _parse_order(value, 3, "--sarima-order"),
        default=(1, 0, 0),
    )
    parser.add_argument(
        "--seasonal-order",
        type=lambda value: _parse_order(value, 4, "--seasonal-order"),
        default=(0, 0, 0, 12),
        help="Ordre saisonnier P,D,Q,s pour SARIMA manuel.",
    )
    parser.add_argument("--no-sarima-auto", dest="sarima_auto", action="store_false")
    args = parser.parse_args()
    if args.horizon < 1 or args.min_tests < 1:
        parser.error("--horizon et --min-tests doivent être positifs.")
    try:
        run_prediction_pipeline(
            args.bacterie,
            args.antibiotique,
            horizon=args.horizon,
            method=args.method,
            csv_path=args.csv.expanduser().resolve(),
            min_tests=args.min_tests,
            arima_auto=args.arima_auto,
            arima_order=args.arima_order,
            sarima_auto=args.sarima_auto,
            sarima_order=args.sarima_order,
            sarima_seasonal_order=args.seasonal_order,
        )
    except (OSError, ValueError, ImportError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
