"""Analysis script for NHANES inflammation markers."""

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
import statsmodels.api as sm
import statsmodels.formula.api as smf

from descriptive_stats import process_cycles, categorize_amalgam



def prepare_groups(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Amalgam Group"] = df["amalgam_surfaces"].apply(categorize_amalgam)
    df["Gender"] = df["RIAGENDR"].replace({1: "Male", 2: "Female"})
    df["Race"] = df["RIDRETH1"].replace({
        1: "Mexican American",
        2: "Other Hispanic",
        3: "Non-Hispanic White",
        4: "Non-Hispanic Black",
        5: "Other Race/Multi-Racial",
    })
    df["AgeGroup"] = pd.cut(
        df["RIDAGEYR"],
        bins=[0, 19, 39, 59, np.inf],
        labels=["0–19", "20–39", "40–59", "60+"],
        right=True,
    )
    return df


def run_t_tests(df: pd.DataFrame) -> pd.DataFrame:
    markers = ["NLR", "MLR", "PLR", "SII", "CRP", "BloodMercury"]
    comparisons = [("None", "Low"), ("None", "Medium"), ("None", "High")]
    strata_vars = ["Gender", "Race", "AgeGroup"]

    results = []
    for cycle, df_cycle in df.groupby("Cycle"):
        for strata in strata_vars:
            for strata_value, df_sub in df_cycle.groupby(strata):
                if pd.isna(strata_value):
                    continue
                for var1, var2 in comparisons:
                    g1 = df_sub[df_sub["Amalgam Group"] == var1]
                    g2 = df_sub[df_sub["Amalgam Group"] == var2]
                    for marker in markers:
                        g1_vals = g1[marker].dropna()
                        g2_vals = g2[marker].dropna()
                        if len(g1_vals) < 10 or len(g2_vals) < 10:
                            continue
                        stat, pval = ttest_ind(g1_vals, g2_vals, equal_var=False)
                        results.append({
                            "Cycle": cycle,
                            "Strata": strata,
                            "Group": strata_value,
                            "Marker": marker,
                            "Comparison": f"{var1} vs {var2}",
                            "Group1 n": len(g1_vals),
                            "Group2 n": len(g2_vals),
                            "t-stat": round(stat, 3),
                            "p-value": round(pval, 5),
                            "Significant": pval < 0.05,
                        })
    return pd.DataFrame(results)


def survey_weighted_anova(df: pd.DataFrame) -> pd.DataFrame:
    """Approximate survey-weighted ANOVA using statsmodels."""

    markers = ["NLR", "MLR", "PLR", "SII", "CRP", "BloodMercury"]
    df = df.rename(columns={"Amalgam Group": "Amalgam_Group"})

    results = []
    for cycle, df_cycle in df.groupby("Cycle"):
        for marker in markers:
            cols = ["WTMEC2YR", "Amalgam_Group", "Gender", "Race", "AgeGroup", marker]
            df_model = df_cycle[cols].dropna()
            if df_model.empty:
                continue
            formula = f"{marker} ~ Amalgam_Group + Gender + Race + AgeGroup"
            try:
                model = smf.wls(formula, data=df_model, weights=df_model["WTMEC2YR"]).fit()
                table = sm.stats.anova_lm(model, typ=2)
            except Exception as exc:  # pragma: no cover - handle regression issues
                print(f"ANOVA failed for {cycle} {marker}: {exc}")
                continue
            for term in ["Amalgam_Group", "Gender", "Race", "AgeGroup"]:
                if term in table.index:
                    pval = table.loc[term, "PR(>F)"]
                    fstat = table.loc[term, "F"]
                    results.append({
                        "Cycle": cycle,
                        "Marker": marker,
                        "Term": term.replace("_", " "),
                        "F_stat": round(fstat, 3),
                        "p_value": round(pval, 5),
                        "Significant": pval < 0.05,
                    })
    return pd.DataFrame(results)


def main():
    combined, _ = process_cycles()
    combined = prepare_groups(combined)
    ttest_df = run_t_tests(combined)
    # Save t-test results to CSV
    ttest_df.to_csv("ttest_results.csv", index=False)
    print(ttest_df.head())
    # anova_df = survey_weighted_anova(combined)
    # anova_df.to_csv("anova_results.csv", index=False)
    # print(anova_df.head())


if __name__ == "__main__":
    main()
