# agents/cleaning_agent.py

import pandas as pd
import numpy as np
from scipy.stats import mstats
from io import BytesIO
import matplotlib.pyplot as plt
import seaborn as sns


class CleaningAgent:
    """
    Handles full data cleaning process:
    - Remove duplicates
    - Impute missing values
    - Remove low variance columns
    - Winsorization
    - Before/after visualization
    Returns:
        cleaned_df
        {
            "summary": {...},
            "plot_bytes": {...},   # real PNG images
            "plot_meta": {...}     # metadata only (safe for JSON)
        }
    """

    def run(self, df: pd.DataFrame):
        summary = {}
        cleaned_df = df.copy()

        # STEP 1 — Remove duplicates
        duplicate_count = cleaned_df.duplicated().sum()
        cleaned_df = cleaned_df.drop_duplicates()
        summary["duplicates_removed"] = int(duplicate_count)

        # STEP 2 — Missing value imputation
        impute_summary = {}
        for col in cleaned_df.columns:
            if cleaned_df[col].isnull().sum() > 0:

                if cleaned_df[col].dtype in [np.float64, np.int64, float, int]:
                    value = cleaned_df[col].median()
                    cleaned_df[col] = cleaned_df[col].fillna(value)
                    impute_summary[col] = f"median ({value})"

                else:
                    value = cleaned_df[col].mode().iloc[0]
                    cleaned_df[col] = cleaned_df[col].fillna(value)
                    impute_summary[col] = f"mode ({value})"

        summary["imputation"] = impute_summary

        # STEP 3 — Low variance columns
        low_var_cols = []
        for col in cleaned_df.select_dtypes(include=[np.number]).columns:
            if cleaned_df[col].var() < 1e-6:
                low_var_cols.append(col)

        cleaned_df = cleaned_df.drop(columns=low_var_cols)
        summary["low_variance_columns_removed"] = low_var_cols

        # STEP 4 — Winsorization
        winsor_summary = {}
        numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            before_min, before_max = cleaned_df[col].min(), cleaned_df[col].max()

            cleaned_df[col] = mstats.winsorize(cleaned_df[col], limits=[0.05, 0.05])

            after_min, after_max = cleaned_df[col].min(), cleaned_df[col].max()

            winsor_summary[col] = {
                "before": {"min": float(before_min), "max": float(before_max)},
                "after": {"min": float(after_min), "max": float(after_max)}
            }

        summary["winsorization"] = winsor_summary

        # STEP 5 — Generate visuals (SEPARATE BYTES + META)
        plot_bytes, plot_meta = self._generate_visuals(df, cleaned_df)

        # FINAL RETURN STRUCTURE
        return cleaned_df, {
            "summary": summary,
            "plot_bytes": plot_bytes,
            "plot_meta": plot_meta
        }


    def _generate_visuals(self, before_df, after_df):
        plot_bytes = {}
        plot_meta = {}

        numeric_cols = before_df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))

            sns.boxplot(x=before_df[col], ax=axes[0])
            axes[0].set_title(f"Before Winsorization: {col}")

            sns.boxplot(x=after_df[col], ax=axes[1])
            axes[1].set_title(f"After Winsorization: {col}")

            buf = BytesIO()
            fig.savefig(buf, format="png")
            buf.seek(0)

            key = f"winsor_compare_{col}"

            # PNG BYTES
            plot_bytes[key] = buf.read()

            # META ONLY
            plot_meta[key] = {"type": "winsor_compare", "column": col}

            plt.close(fig)

        return plot_bytes, plot_meta
