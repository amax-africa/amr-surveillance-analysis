"""Focused regression tests for quarterly_forecasting."""

import unittest

import numpy as np
import pandas as pd

from quarterly_forecasting import (
    ForecastConfig,
    aggregate_quarterly_counts,
    forecast_quarterly,
    to_jsonable,
)


class QuarterlyAggregationTests(unittest.TestCase):
    def test_counts_keep_empty_quarter_and_invalid_dates(self) -> None:
        data = pd.DataFrame(
            {
                "row_uid": ["r1", "r2", "r3", "r4", "r5"],
                "specimen_collection_date": [
                    "2022-01-10", "2022-03-04", "2022-07-01", None, "not-a-date"
                ],
                "organism_name": ["E. coli"] * 5,
                "antibiotic": ["cefotaxime"] * 5,
                "interpretation": ["R", "S", "S", "R", "R"],
            }
        )
        counts, meta = aggregate_quarterly_counts(data, "E. coli", "cefotaxime")
        self.assertEqual(str(counts.index[0]), "2022Q1")
        self.assertEqual(str(counts.index[-1]), "2022Q3")
        self.assertEqual(len(counts), 3)
        self.assertEqual(counts.loc[pd.Period("2022Q1", freq="Q-DEC"), "N"], 2)
        self.assertTrue(np.isnan(counts.loc[pd.Period("2022Q2", freq="Q-DEC"), "p_R"]))
        self.assertEqual(counts.loc[pd.Period("2022Q3", freq="Q-DEC"), "p_R"], 0.0)
        self.assertEqual(meta["missing_date_rows"], 1)
        self.assertEqual(meta["unparseable_date_rows"], 1)
        self.assertEqual(meta["invalid_date_rows"], 2)

    def test_all_organisms_and_antibiotics_pool_interpretable_tests(self) -> None:
        data = pd.DataFrame(
            {
                "row_uid": ["r1", "r1", "r2", "r2", "r3", "r3", "r4", "r4"],
                "specimen_collection_date": [
                    "2022-01-10",
                    "2022-01-10",
                    "2022-01-11",
                    "2022-01-11",
                    "2022-01-12",
                    "2022-01-12",
                    "2022-04-01",
                    "2022-04-01",
                ],
                "organism_name": [
                    "E. coli",
                    "E. coli",
                    "K. pneumoniae",
                    "K. pneumoniae",
                    "E. coli",
                    "E. coli",
                    "",
                    "",
                ],
                "antibiotic": [
                    "drug_a",
                    "drug_b",
                    "drug_a",
                    "drug_b",
                    "drug_a",
                    "drug_b",
                    "drug_a",
                    "drug_b",
                ],
                "interpretation": ["R", "S", "S", "R", "I", None, "R", "R"],
            }
        )

        counts, meta = aggregate_quarterly_counts(data, None, None)

        self.assertEqual(counts.loc[pd.Period("2022Q1", freq="Q-DEC"), "N"], 5)
        self.assertEqual(counts.loc[pd.Period("2022Q1", freq="Q-DEC"), "R"], 2)
        self.assertAlmostEqual(
            counts.loc[pd.Period("2022Q1", freq="Q-DEC"), "p_R"], 2 / 5
        )
        self.assertEqual(meta["usable_sir_rows"], 5)
        self.assertEqual(meta["organism_rows"], 3)


class QuarterlyForecastTests(unittest.TestCase):
    def test_naive_holdout_and_refit_preserve_gap(self) -> None:
        periods = pd.period_range("2020Q1", periods=9, freq="Q-DEC", name="quarter")
        values = pd.Series(
            [0.1, 0.2, np.nan, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            index=periods,
            name="p_R",
        )
        result = forecast_quarterly(
            values,
            method="naive",
            horizon=2,
            config=ForecastConfig(
                initial_window=3,
                validation_horizons=(1, 2),
                holdout_size=2,
            ),
        )
        self.assertEqual(result["status"], "ok")
        self.assertTrue(np.isnan(result["history"].loc[periods[2], "p_R"]))
        self.assertEqual(result["holdout"].iloc[0]["forecast"], 0.6)
        self.assertEqual(result["future"]["forecast"].tolist(), [0.8, 0.8])
        self.assertFalse(bool(result["future"].iloc[0]["interval_available"]))
        self.assertIn("missing_quarters", result["limitations"])
        self.assertIsInstance(to_jsonable(result), dict)

    def test_binomial_glm_emits_mean_and_predictive_intervals(self) -> None:
        periods = pd.period_range("2018Q1", periods=20, freq="Q-DEC", name="quarter")
        n = np.repeat(100, len(periods))
        resistant = np.asarray([25 + (i % 5) * 3 + (i // 5) for i in range(len(periods))])
        frame = pd.DataFrame(
            {"R": resistant, "N": n, "p_R": resistant / n}, index=periods
        )
        result = forecast_quarterly(
            frame,
            method="binomial_glm",
            horizon=2,
            future_n=200,
            config=ForecastConfig(
                initial_window=10,
                validation_horizons=(1,),
                holdout_size=2,
            ),
        )
        self.assertEqual(result["status"], "ok", result["error"])
        self.assertTrue(result["future"]["lower"].between(0, 1).all())
        self.assertTrue(result["future"]["upper"].between(0, 1).all())
        self.assertTrue(result["future"]["predictive_lower"].between(0, 1).all())
        self.assertTrue((result["future"]["future_n"] == 200).all())


if __name__ == "__main__":
    unittest.main()
