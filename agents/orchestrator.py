# agents/orchestrator.py

import pandas as pd
from .eda_agent import EDAAgent
from .cleaning_agent import CleaningAgent
from .insights_agent import InsightsAgent
from .reporting_agent import ReportingAgent

from .memory.session_memory import SessionMemory
from .memory.long_term_memory import LongTermMemory


class Orchestrator:
    """
    Central orchestrator for the multi-agent data pipeline:
    1. EDA with visualizations
    2. Data cleaning
    3. AI-generated insights
    4. Report generation (PDF + DOCX)
    5. Chat follow-ups
    """

    def __init__(self):
        self.eda_agent = EDAAgent()
        self.cleaning_agent = CleaningAgent()
        self.insights_agent = InsightsAgent()
        self.reporting_agent = ReportingAgent()

        self.memory = SessionMemory()
        self.long_term_memory = LongTermMemory()

    # ----------------------------------------------------
    # FULL PIPELINE
    # ----------------------------------------------------
    def run_full_pipeline(self, df: pd.DataFrame, run_cleaning_after_eda: bool = True):

        # ---------- STEP 1: EDA ----------
        eda_output, (eda_plot_bytes, eda_plot_meta) = self.eda_agent.run(df)

        # Save plots in memory (bytes + metadata)
        eda_output["visualizations"] = eda_plot_meta
        eda_output["visualizations_bytes"] = eda_plot_bytes

        self.memory.set("eda", eda_output)
        self.memory.set("eda_plots", eda_plot_bytes)
        self.memory.set("original_df_shape", df.shape)

        result = {"eda": eda_output, "plots": eda_plot_meta}

        # ---------- STEP 2: CLEANING ----------
        cleaned_df = df
        cleaning_summary = {}

        if run_cleaning_after_eda:
            cleaned_df, cleaning_output = self.cleaning_agent.run(df)

            cleaning_bytes = cleaning_output.get("plot_bytes", {})
            cleaning_meta = cleaning_output.get("plot_meta", {})

            # JSON-safe version
            cleaning_summary = cleaning_output.copy()
            cleaning_summary["visualizations"] = cleaning_meta
            cleaning_summary["visualizations_bytes"] = cleaning_bytes
            cleaning_summary.pop("plot_bytes", None)

            # Save in memory
            self.memory.set("cleaned_df", cleaned_df)
            self.memory.set("cleaning_summary", cleaning_summary)
            self.memory.set("cleaning_plots", cleaning_bytes)

            result.update({
                "cleaned_df": cleaned_df,
                "cleaning_summary": cleaning_summary,
                "plots_after": cleaning_meta
            })

        # ---------- STEP 3: INSIGHTS ----------
        insights = self.insights_agent.generate_insights(
            original_df=df,
            cleaned_df=cleaned_df,
            eda_output=eda_output,
            cleaning_summary=cleaning_summary
        )

        self.memory.set("last_insights", insights)
        result["insights"] = insights

        # ---------- STEP 4: REPORT GENERATION ----------
        reports = self.reporting_agent.generate_reports(
            original_df=df,
            cleaned_df=cleaned_df,
            eda_output=eda_output,
            cleaning_summary=cleaning_summary,
            insights=insights
        )

        result["final_report_pdf"] = reports.get("pdf_bytes")
        result["final_report_docx"] = reports.get("docx_bytes")

        return result

    # ----------------------------------------------------
    # CLEANING ONLY
    # ----------------------------------------------------
    def run_cleaning(self, df: pd.DataFrame):
        cleaned_df, cleaning_output = self.cleaning_agent.run(df)

        cleaning_bytes = cleaning_output.get("plot_bytes", {})
        cleaning_meta = cleaning_output.get("plot_meta", {})

        cleaning_output["visualizations"] = cleaning_meta
        cleaning_output["visualizations_bytes"] = cleaning_bytes
        cleaning_output.pop("plot_bytes", None)

        self.memory.set("cleaned_df", cleaned_df)
        self.memory.set("cleaning_summary", cleaning_output)
        self.memory.set("cleaning_plots", cleaning_bytes)

        return {
            "cleaned_df": cleaned_df,
            "report": cleaning_output,
            "plots_after": cleaning_meta
        }

    # ----------------------------------------------------
    # CHATBOT FOLLOW-UP
    # ----------------------------------------------------
    def handle_chat_query(self, user_input: str, session_state: dict):
        context = {
            "eda": self.memory.get("eda", {}),
            "cleaning_summary": self.memory.get("cleaning_summary", {}),
            "insights": self.memory.get("last_insights", {})
        }

        resp = self.insights_agent.ask_followup(user_input, context)
        text = resp.get("text", "")

        # Check if LLM requested a plot
        if "PLOT_REQUEST" in text:
            try:
                _, plot_name = text.split(":", 1)
                plot_name = plot_name.strip()
            except Exception:
                plot_name = None

            eda_plots = self.memory.get("eda_plots", {})
            cleaning_plots = self.memory.get("cleaning_plots", {})

            plot_bytes = eda_plots.get(plot_name) or cleaning_plots.get(plot_name)

            return {
                "text": f"Returning plot: {plot_name}" if plot_bytes else "Plot not found.",
                "plot": plot_bytes
            }
        return {"text": text}
