import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils.plot_utils import fig_to_png_bytes


class EDAAgent:
    """
    Performs full exploratory data analysis.
    Returns two things:
        results (JSON-safe)
        (plot_bytes, plot_meta)
    """

    def run(self, df: pd.DataFrame):
        results = {}

        # Basic information
        results["shape"] = df.shape
        results["columns"] = list(df.columns)
        results["missing_values"] = df.isnull().sum().to_dict()

        # Summary statistics
        results["summary_stats"] = df.describe(include="all").to_dict()

        # Numeric dataframe
        numeric_df = df.select_dtypes(include=[np.number])

        # Skewness & kurtosis
        if not numeric_df.empty:
            results["skewness"] = numeric_df.skew().to_dict()
            results["kurtosis"] = numeric_df.kurt().to_dict()
        else:
            results["skewness"] = {}
            results["kurtosis"] = {}

        # Outliers using IQR
        outliers = {}
        for col in numeric_df.columns:
            Q1 = numeric_df[col].quantile(0.25)
            Q3 = numeric_df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            count = ((numeric_df[col] < lower) | (numeric_df[col] > upper)).sum()

            outliers[col] = {
                "IQR": float(IQR),
                "lower_bound": float(lower),
                "upper_bound": float(upper),
                "outlier_count": int(count),
            }

        results["outliers"] = outliers

        # Correlation matrix
        results["correlation_matrix"] = (
            numeric_df.corr().to_dict() if len(numeric_df.columns) > 1 else {}
        )

        # ----------- VISUALIZATIONS ------------
        plot_bytes, plot_meta = self._generate_visuals(df)

        results["visualizations"] = plot_meta

        return results, (plot_bytes, plot_meta)

    # =============================================
    # INTERNAL: Visualization Generator
    # =============================================
    def _generate_visuals(self, df: pd.DataFrame):
        plot_bytes = {}
        metadata = {}

        numeric_df = df.select_dtypes(include=[np.number])
        n_cols = len(numeric_df.columns)

        # ---------- Distribution plots ----------
        for col in numeric_df.columns:
            fig, ax = plt.subplots(figsize=(4, 3))  # compact dashboard size
            sns.histplot(df[col], kde=True, ax=ax, color='skyblue')
            ax.set_title(f"Distribution of {col}", fontsize=10)
            plt.tight_layout()

            key = f"dist_{col}"
            plot_bytes[key] = fig_to_png_bytes(fig)
            metadata[key] = f"Distribution plot for {col}"

            plt.close(fig)

        # ---------- Correlation Heatmap ----------
        if n_cols > 1:
            # Dynamically scale figure size based on number of columns
            fig_width = max(8, n_cols * 0.5)
            fig_height = max(6, n_cols * 0.5)
            fig, ax = plt.subplots(figsize=(fig_width, fig_height))

            sns.heatmap(
                numeric_df.corr(),
                annot=True,
                cmap="coolwarm",
                fmt=".2f",
                ax=ax,
                cbar=True,
                square=False
            )

            # Rotate x-axis labels if too many columns
            rotation = 45 if n_cols > 10 else 0
            ax.set_xticklabels(ax.get_xticklabels(), rotation=rotation, ha='right')
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
            ax.set_title("Correlation Heatmap", fontsize=12)
            plt.tight_layout()

            key = "correlation_heatmap"
            plot_bytes[key] = fig_to_png_bytes(fig)
            metadata[key] = "Correlation heatmap"

            plt.close(fig)

        return plot_bytes, metadata
