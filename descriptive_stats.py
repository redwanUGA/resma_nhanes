import os
import numpy as np
import pandas as pd
import pyreadstat

CBC_DEMO_DENTAL_FILES = {
    # (CBC, Demographics, Dental, CRP, Mercury)
    "1999-2000": ("L40_0.xpt", "DEMO.xpt", "OHXDENT.xpt", "LAB11.xpt", "LAB06HM.xpt"),
    "2001-2002": ("L25_B.xpt", "DEMO_B.xpt", "OHXDEN_B.xpt", "L11_B.xpt", "L06_2_B.xpt"),
    "2003-2004": ("L25_C.xpt", "DEMO_C.xpt", "OHXDEN_C.xpt", "L11_C.xpt", "L06BMT_C.xpt"),
    "2005-2006": ("CBC_D.xpt", "DEMO_D.xpt", "OHXDEN_D.xpt", "CRP_D.xpt", "PbCd_D.xpt"),
    "2007-2008": ("CBC_E.xpt", "DEMO_E.xpt", "OHXDEN_E.xpt", "CRP_E.xpt", "PbCd_E.xpt"),
    "2009-2010": ("CBC_F.xpt", "DEMO_F.xpt", "OHXDEN_F.xpt", "CRP_F.xpt", "PbCd_F.xpt"),
    "2011-2012": ("CBC_G.xpt", "DEMO_G.xpt", "OHXDEN_G.xpt", "CRP_G.xpt", "PbCd_G.xpt"),
    "2013-2014": ("CBC_H.xpt", "DEMO_H.xpt", "OHXDEN_H.xpt", "CRP_H.xpt", "PBCD_H.xpt"),
    "2015-2016": ("CBC_I.xpt", "DEMO_I.xpt", "OHXDEN_I.xpt", "HSCRP_I.xpt", "PBCD_I.xpt"),
    "2017-2018": ("CBC_J.xpt", "DEMO_J.xpt", "OHXDEN_J.xpt", "HSCRP_J.xpt", "PBCD_J.xpt"),
}


def count_amalgam_surfaces(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in df.columns if c.startswith("OHX") and c.endswith(("TC", "FS", "FT"))]
    df["amalgam_surfaces"] = (df[cols] == 2).sum(axis=1)
    return df[["SEQN", "amalgam_surfaces"]]


def weighted_stats(series: pd.Series, weights: pd.Series):
    try:
        mean = np.average(series, weights=weights)
        variance = np.average((series - mean) ** 2, weights=weights)
    except Exception:
        mean = series.mean()
        variance = series.var()
    std = np.sqrt(variance)
    se = std / np.sqrt(len(series))
    return round(mean, 3), round(std, 3), round(mean - 1.96 * se, 3), round(mean + 1.96 * se, 3)


def process_cycles(data_dir: str = "nhanes_data"):
    df_all = []
    all_summaries = []
    for cycle, (
        cbc_file,
        demo_file,
        dental_file,
        crp_file,
        mercury_file,
    ) in CBC_DEMO_DENTAL_FILES.items():
        try:
            cbc = pyreadstat.read_xport(os.path.join(data_dir, cbc_file))[0]
            demo = pyreadstat.read_xport(os.path.join(data_dir, demo_file))[0]
            dental = pyreadstat.read_xport(os.path.join(data_dir, dental_file))[0]
            crp = pyreadstat.read_xport(os.path.join(data_dir, crp_file))[0]
            mercury = pyreadstat.read_xport(os.path.join(data_dir, mercury_file))[0]
            dental = count_amalgam_surfaces(dental)

            df = (
                demo.merge(cbc, on="SEQN")
                .merge(crp, on="SEQN", how="left")
                .merge(mercury, on="SEQN", how="left")
                .merge(dental, on="SEQN", how="left")
            )
            df["Cycle"] = cycle

            df["WBC"] = df.get("LBXWBCSI")
            df["Neutro"] = df["WBC"] * df.get("LBXNEPCT", 0) / 100
            df["Lympho"] = df["WBC"] * df.get("LBXLYPCT", 0) / 100
            df["Mono"] = df["WBC"] * df.get("LBXMOPCT", 0) / 100
            df["Platelets"] = df.get("LBXPLTSI")
            df["CRP"] = df.get("LBXCRP") if "LBXCRP" in df.columns else df.get("LBXHSCRP")
            df["BloodMercury"] = df.get("LBXTHG")

            df["NLR"] = df["Neutro"] / df["Lympho"]
            df["MLR"] = df["Mono"] / df["Lympho"]
            df["PLR"] = df["Platelets"] / df["Lympho"]
            df["SII"] = (df["Neutro"] * df["Platelets"]) / df["Lympho"]

            df_all.append(df)

            for marker in ["NLR", "MLR", "PLR", "SII", "CRP", "BloodMercury"]:
                sub = df[[marker, "WTMEC2YR"]].dropna()
                if sub.empty:
                    continue
                m, sd, lo, hi = weighted_stats(sub[marker], sub["WTMEC2YR"])
                all_summaries.append({
                    "Cycle": cycle,
                    "Marker": marker,
                    "Mean": m,
                    "SD": sd,
                    "CI_Low": lo,
                    "CI_High": hi,
                    "Sample Size": len(sub),
                })
        except Exception as exc:
            print(f"Skipped {cycle}: {exc}")

    combined_df = pd.concat(df_all, ignore_index=True)
    summary_df = pd.DataFrame(all_summaries)
    return combined_df, summary_df


def categorize_amalgam(surfaces: float):
    if pd.isna(surfaces):
        return np.nan
    if surfaces == 0:
        return "None"
    if surfaces <= 5:
        return "Low"
    if surfaces <= 10:
        return "Medium"
    return "High"


def compute_demographic_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate marker statistics stratified by demographic groups."""
    # Import lazily to avoid circular dependency when analysis.py imports this module
    from analysis import prepare_groups

    df = prepare_groups(df)

    markers = ["NLR", "MLR", "PLR", "SII", "CRP", "BloodMercury"]
    demo_vars = ["Gender", "Race", "AgeGroup"]

    results = []
    for cycle, df_cycle in df.groupby("Cycle"):
        for demo in demo_vars:
            for group_val, df_sub in df_cycle.groupby(demo):
                if pd.isna(group_val):
                    continue
                for marker in markers:
                    sub = df_sub[[marker, "WTMEC2YR"]].dropna()
                    if sub.empty:
                        continue
                    m, sd, lo, hi = weighted_stats(sub[marker], sub["WTMEC2YR"])
                    results.append({
                        "Cycle": cycle,
                        "Demographic": demo,
                        "Group": group_val,
                        "Marker": marker,
                        "Mean": m,
                        "SD": sd,
                        "CI_Low": lo,
                        "CI_High": hi,
                        "Sample Size": len(sub),
                    })
    return pd.DataFrame(results)


if __name__ == "__main__":
    combined_df, summary_df = process_cycles()
    # Save full combined dataset and summary statistics to CSV files
    combined_df.to_csv("combined_dataset.csv", index=False)
    summary_df.to_csv("summary_statistics.csv", index=False)
    demo_stats_df = compute_demographic_stats(combined_df)
    demo_stats_df.to_csv("demographic_statistics.csv", index=False)
    print(summary_df.head())
    print(demo_stats_df.head())
