"""Regression tests for monthly forecasting methods."""

import unittest

import numpy as np
import pandas as pd

from forecasting import (
    arima_forecasting,
    compare_forecasting_methods,
    monthly_binomial_glm_forecasting,
    sarima_forecasting,
)


class ArimaForecastingTests(unittest.TestCase):
    def setUp(self) -> None:
        time = np.arange(36, dtype=float)
        self.series = 20 + 0.15 * time + 2 * np.sin(2 * np.pi * time / 12)

    def test_arima_selects_order_inside_aicc_search_bounds(self) -> None:
        result = arima_forecasting(
            self.series,
            horizon=3,
            auto_order=True,
            p_max=1,
            d_max=1,
            q_max=1,
            max_candidates=4,
        )

        self.assertEqual(len(result["forecast"]), 3)
        self.assertEqual(result["candidate_count"], 4)
        self.assertTrue(
            all(value <= bound for value, bound in zip(result["order"], (1, 1, 1)))
        )
        self.assertIn("AICc", result["selection"])

    def test_arima_accepts_manual_order(self) -> None:
        result = arima_forecasting(
            self.series,
            horizon=2,
            order=(1, 0, 1),
            auto_order=False,
        )

        self.assertEqual(result["order"], (1, 0, 1))
        self.assertEqual(result["candidate_count"], 1)
        self.assertEqual(len(result["forecast"]), 2)

    def test_sarima_produces_monthly_forecasts_and_bounded_intervals(self) -> None:
        result = sarima_forecasting(
            self.series,
            horizon=2,
            auto_order=True,
            p_max=0,
            d_max=0,
            q_max=0,
            seasonal_p_max=0,
            seasonal_d_max=0,
            seasonal_q_max=0,
            max_candidates=1,
        )

        self.assertEqual(len(result["forecast"]), 2)
        self.assertEqual(result["candidate_count"], 1)
        self.assertTrue(all(0 <= value <= 100 for value in result["forecast"]))
        self.assertTrue(
            all(
                lower <= mean <= upper
                for lower, mean, upper in zip(
                    result["ci_lower"], result["forecast"], result["ci_upper"]
                )
            )
        )

    def test_binomial_glm_uses_monthly_counts(self) -> None:
        months = [
            str(period)
            for period in pd.period_range("2021-01", periods=36, freq="M")
        ]
        tested = np.full(36, 40)
        rng = np.random.default_rng(42)
        monthly_probability = np.asarray(
            [
                0.2 + 0.05 * np.sin(index * 2 * np.pi / 12)
                for index in range(len(tested))
            ]
        )
        resistant = rng.binomial(tested, monthly_probability)
        result = monthly_binomial_glm_forecasting(
            resistant,
            tested,
            months,
            horizon=3,
            future_n=40,
        )

        self.assertEqual(len(result["forecast"]), 3)
        self.assertEqual(result["future_n"], [40, 40, 40])
        self.assertTrue(all(0 <= value <= 100 for value in result["forecast"]))
        self.assertGreaterEqual(result["dispersion"], 0)

    def test_monthly_comparison_uses_the_five_shared_methods(self) -> None:
        months = [
            str(period)
            for period in pd.period_range("2021-01", periods=48, freq="M")
        ]
        tested = np.full(48, 60)
        rng = np.random.default_rng(7)
        probabilities = 0.25 + 0.04 * np.sin(np.arange(48) * 2 * np.pi / 12)
        resistant = rng.binomial(tested, probabilities)
        counts = pd.DataFrame({"R": resistant, "N": tested})
        rates = tuple((100 * resistant / tested).tolist())

        results = compare_forecasting_methods(
            rates,
            horizon=4,
            seasonal_data_contiguous=True,
            monthly_counts=counts,
            months=months,
        )

        self.assertEqual(
            [row["Méthode"] for row in results],
            [
                "Prévision naïve",
                "ETS / Holt-Winters",
                "ARIMA",
                "SARIMA",
                "Régression binomiale temporelle",
            ],
        )
        self.assertTrue(all(row["Statut"] == "Calculée" for row in results))


if __name__ == "__main__":
    unittest.main()
