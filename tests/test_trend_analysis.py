"""Tests unitaires ciblés pour l'analyse statistique de la tendance nationale AMR."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from trend_analysis import (
    analyze_all_national_trends,
    build_national_monthly_sir_cube,
    estimate_national_monthly_trend,
    get_national_monthly_sir_series,
)


def _make_long_test_frame(
    bacterie: str,
    antibiotique: str,
    monthly_specs: list[tuple[str, int, int, int]],
    hospital: str = "Hôpital A",
) -> pd.DataFrame:
    """Construit un DataFrame au format long à partir de triplets mensuels (mois, S, I, R)."""
    rows = []
    uid_counter = 1
    for month_str, s_cnt, i_cnt, r_cnt in monthly_specs:
        dt_str = f"{month_str}-15"
        for code, count in (("S", s_cnt), ("I", i_cnt), ("R", r_cnt)):
            for _ in range(count):
                rows.append(
                    {
                        "row_uid": f"{bacterie[:3]}_{antibiotique[:3]}_{uid_counter:06d}",
                        "specimen_collection_date": dt_str,
                        "hospital_name": hospital,
                        "year": month_str[:4],
                        "ward": "Médecine",
                        "organism_name": bacterie,
                        "antibiotic": antibiotique,
                        "interpretation": code,
                    }
                )
                uid_counter += 1
    return pd.DataFrame(rows)


class NationalTrendAnalysisTests(unittest.TestCase):
    def test_increasing_trend_detected_with_positive_slope_and_significant_pvalue(self) -> None:
        months = [str(p) for p in pd.period_range("2020-01", periods=36, freq="M")]
        t_years = np.arange(36, dtype=float) / 12.0
        # Tendance log-odds : logit(p_t) = -1.4 + 0.65 * t_years (hausse nette de ~20% à ~63%)
        true_p = 1.0 / (1.0 + np.exp(-(-1.4 + 0.65 * t_years)))
        n_per_month = 100
        r_counts = np.rint(n_per_month * true_p).astype(int)
        s_counts = n_per_month - r_counts

        monthly_df = pd.DataFrame(
            {
                "mois": months,
                "S": s_counts,
                "I": np.zeros(36, dtype=int),
                "R": r_counts,
                "N": np.full(36, n_per_month, dtype=int),
            }
        )
        res = estimate_national_monthly_trend(monthly_df)

        self.assertTrue(res["calculable"])
        self.assertEqual(res["status"], "Calculé")
        self.assertEqual(res["n_months"], 36)
        self.assertEqual(res["total_N"], 3600)
        self.assertEqual(res["total_R"], int(r_counts.sum()))
        self.assertGreater(res["beta1_logit_per_year"], 0.45)
        self.assertGreater(res["or_per_year"], 1.5)
        self.assertGreater(res["slope_pp_per_year"], 8.0)
        self.assertGreater(res["slope_pp_ci_lower"], 0.0)
        self.assertLess(res["p_value"], 1e-4)

    def test_flat_series_shows_no_significant_trend(self) -> None:
        months = [str(p) for p in pd.period_range("2021-01", periods=24, freq="M")]
        # Taux oscillant symétriquement autour de 30 % sans dérive séculaire
        r_pattern = [29, 31, 30, 30, 31, 29] * 4
        n_pattern = [100] * 24
        monthly_df = pd.DataFrame(
            {
                "mois": months,
                "S": [n - r for n, r in zip(n_pattern, r_pattern)],
                "I": [0] * 24,
                "R": r_pattern,
                "N": n_pattern,
            }
        )
        res = estimate_national_monthly_trend(monthly_df)

        self.assertTrue(res["calculable"])
        self.assertAlmostEqual(res["slope_pp_per_year"], 0.0, delta=0.5)
        self.assertAlmostEqual(res["or_per_year"], 1.0, delta=0.05)
        self.assertLess(res["slope_pp_ci_lower"], 0.0)
        self.assertGreater(res["slope_pp_ci_upper"], 0.0)
        self.assertGreater(res["p_value"], 0.50)

    def test_insufficient_months_or_counts_and_constant_resistance_are_reported_not_calculated(self) -> None:
        # 1. Moins de 6 mois observés
        short_df = pd.DataFrame(
            {
                "mois": ["2023-01", "2023-02", "2023-03", "2023-04"],
                "S": [15, 14, 16, 15],
                "I": [0, 0, 0, 0],
                "R": [5, 6, 4, 5],
                "N": [20, 20, 20, 20],
            }
        )
        res_short = estimate_national_monthly_trend(short_df, min_months=6, min_total_n=30)
        self.assertFalse(res_short["calculable"])
        self.assertIn("nombre de mois insuffisant", res_short["status"])

        # 2. Effectif cumulé N < 30 sur >= 6 mois
        low_n_df = pd.DataFrame(
            {
                "mois": [f"2023-0{m}" for m in range(1, 8)],
                "S": [2] * 7,
                "I": [0] * 7,
                "R": [1] * 7,
                "N": [3] * 7,
            }
        )
        res_low_n = estimate_national_monthly_trend(low_n_df, min_months=6, min_total_n=30)
        self.assertFalse(res_low_n["calculable"])
        self.assertIn("effectif total insuffisant", res_low_n["status"])

        # 3. Résistance constante à 0 %
        zero_r_df = pd.DataFrame(
            {
                "mois": [f"2023-0{m}" for m in range(1, 8)],
                "S": [10] * 7,
                "I": [0] * 7,
                "R": [0] * 7,
                "N": [10] * 7,
            }
        )
        res_zero = estimate_national_monthly_trend(zero_r_df, min_months=6, min_total_n=30)
        self.assertFalse(res_zero["calculable"])
        self.assertIn("0 %", res_zero["status"])

        # 4. Quasi-séparation complète (0 R pendant 11 mois, 100 % R au 12e mois)
        sep_months = [str(p) for p in pd.period_range("2023-01", periods=12, freq="M")]
        sep_df = pd.DataFrame(
            {
                "mois": sep_months,
                "S": [10] * 11 + [0],
                "I": [0] * 12,
                "R": [0] * 11 + [10],
                "N": [10] * 12,
            }
        )
        res_sep = estimate_national_monthly_trend(sep_df, min_months=6, min_total_n=30)
        self.assertFalse(res_sep["calculable"])
        self.assertIn("quasi-séparation", res_sep["status"])

    def test_missing_months_are_never_imputed_to_zero_and_time_scale_remains_exact(self) -> None:
        # Série avec un trou de 5 mois en milieu d'année (2022-04 à 2022-08 absents)
        observed_months = [
            "2022-01", "2022-02", "2022-03",
            # trou de 5 mois sans observation
            "2022-09", "2022-10", "2022-11", "2022-12",
            "2023-01", "2023-02", "2023-03",
        ]
        specs = [(m, 6, 2, 12) for m in observed_months]  # S=6, I=2, R=12 -> N=20, %R=60% constant
        long_df = _make_long_test_frame("Escherichia coli", "Cefotaxime", specs)

        cube = build_national_monthly_sir_cube(long_df)
        # Vérifie que N = S + I + R (inclut bien I) et qu'aucun mois absent n'a été ajouté
        self.assertEqual(len(cube), len(observed_months))
        self.assertTrue((cube["N"] == 20).all())
        self.assertTrue((cube["%R"] == 60.0).all())

        # Avec include_empty_months_as_nan=True, les mois manquants ont N=0 et %R=NaN (jamais 0.0)
        continuous_series = get_national_monthly_sir_series(
            cube, "Escherichia coli", "Cefotaxime", include_empty_months_as_nan=True
        )
        missing_rows = continuous_series.loc[continuous_series["N"] == 0]
        self.assertEqual(len(missing_rows), 5)
        self.assertTrue(missing_rows["%R"].isna().all())

        # Si on estime la tendance sur `continuous_series` contenant des lignes N=0/%R=NaN,
        # ces mois sont ignorés (jamais traités comme 0 %) et la pente reste exactement 0
        res = estimate_national_monthly_trend(continuous_series)
        self.assertTrue(res["calculable"])
        self.assertEqual(res["n_months"], 10)
        self.assertEqual(res["total_N"], 200)
        self.assertEqual(res["total_R"], 120)
        self.assertAlmostEqual(res["slope_pp_per_year"], 0.0, places=5)
        # Vérifie que le dernier mois (2023-03) est bien à (14 mois / 12) = 1.1667 an de 2022-01
        self.assertAlmostEqual(res["span_years"], round(14.0 / 12.0, 2), places=2)

    def test_overdispersion_and_autocorrelation_trigger_robust_inference_instead_of_naive_mle(self) -> None:
        months = [str(p) for p in pd.period_range("2020-01", periods=36, freq="M")]
        # Blocs fortement autocorrélés et surdispersés (N=200 par mois)
        # Sans vraie tendance linéaire, mais avec de fortes vagues corrélées qui tromperaient un MLE naïf
        wave = np.repeat([40, 120, 45, 115, 50, 110], 6)
        monthly_df = pd.DataFrame(
            {
                "mois": months,
                "S": 200 - wave,
                "I": np.zeros(36, dtype=int),
                "R": wave,
                "N": np.full(36, 200, dtype=int),
            }
        )
        res = estimate_national_monthly_trend(monthly_df)

        self.assertTrue(res["calculable"])
        self.assertTrue(
            res["diagnostics"]["overdispersed"] or res["diagnostics"]["autocorrelated"]
        )
        self.assertNotEqual(res["inference_method"], "GLM Binomial (MLE standard)")
        # L'erreur-type robuste (Quasi-Binomiale ou HAC) est strictement plus prudente que le MLE naïf
        self.assertGreater(res["p_value"], res["p_value_naive_mle"])

    def test_fdr_correction_applies_only_to_calculable_family_and_preserves_counts(self) -> None:
        months_12 = [str(p) for p in pd.period_range("2022-01", periods=12, freq="M")]
        # Couple 1 : tendance fortement croissante (calculable)
        specs_inc = [
            (m, 90 - 5 * idx, 0, 10 + 5 * idx) for idx, m in enumerate(months_12)
        ]
        # Couple 2 : tendance nulle (calculable)
        specs_flat = [(m, 70, 5, 25) for m in months_12]
        # Couple 3 : données insuffisantes (seulement 3 mois -> non calculable, exclu de FDR)
        specs_short = [("2022-01", 10, 0, 5), ("2022-02", 10, 0, 5), ("2022-03", 10, 0, 5)]

        df_all = pd.concat(
            [
                _make_long_test_frame("Klebsiella pneumoniae", "Cefotaxime", specs_inc, hospital="H1"),
                _make_long_test_frame("Escherichia coli", "Imipenem", specs_flat, hospital="H2"),
                _make_long_test_frame("Pseudomonas aeruginosa", "Amikacin", specs_short, hospital="H1"),
            ],
            ignore_index=True,
        )

        table_df, meta = analyze_all_national_trends(df_all, min_months=6, min_total_n=30)

        self.assertEqual(meta["n_pairs_total"], 3)
        self.assertEqual(meta["n_pairs_tested_fdr"], 2)
        self.assertEqual(meta["n_pairs_excluded"], 1)

        calculable_rows = table_df.loc[table_df["Statut"] == "Calculé"]
        non_calc_rows = table_df.loc[table_df["Statut"] != "Calculé"]
        self.assertEqual(len(calculable_rows), 2)
        self.assertEqual(len(non_calc_rows), 1)
        self.assertTrue(calculable_rows["p-valeur FDR (BH)"].notna().all())
        self.assertTrue(non_calc_rows["p-valeur FDR (BH)"].isna().all())

        # Pour 2 tests dans la famille BH, la plus petite p-valeur corrigée vaut min(1, 2 * p_min)
        p_min = calculable_rows["p-valeur"].min()
        fdr_min = calculable_rows["p-valeur FDR (BH)"].min()
        self.assertAlmostEqual(fdr_min, min(1.0, 2.0 * p_min), places=6)


if __name__ == "__main__":
    unittest.main()

