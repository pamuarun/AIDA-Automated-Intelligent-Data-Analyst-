# utils/chatbot_parser.py

import re

class ChatbotParser:
    """
    Extracts user intent, column names, and keywords from natural language queries.
    Used by Orchestrator and Insights Agent.
    """

    @staticmethod
    def extract_column_name(user_input: str, columns: list):
        """
        Attempt to detect a column name mentioned inside a user query.
        """
        lower_input = user_input.lower()

        # Exact matches
        for col in columns:
            if col.lower() in lower_input:
                return col

        # Regex fallback: tokens matching column patterns
        tokens = re.split(r'[^a-zA-Z0-9_]+', lower_input)
        for token in tokens:
            for col in columns:
                if token == col.lower():
                    return col

        return None

    @staticmethod
    def detect_intent(user_input: str):
        """
        Identify high-level intent: distribution, summary, correlation, missing, report, etc.
        """
        u = user_input.lower()

        if "distribution" in u or "histogram" in u:
            return "distribution"

        if "missing" in u or "null" in u:
            return "missing_values"

        if "correlation" in u or "heatmap" in u:
            return "correlation"

        if "summary" in u or "describe" in u:
            return "summary"

        if "generate report" in u or "final report" in u:
            return "generate_report"

        if "skewness" in u:
            return "skewness"

        if "kurtosis" in u:
            return "kurtosis"

        return "general_query"

    @staticmethod
    def clean_text(text: str):
        """
        General cleaning for user queries before sending to LLM.
        """
        return " ".join(text.strip().split())

