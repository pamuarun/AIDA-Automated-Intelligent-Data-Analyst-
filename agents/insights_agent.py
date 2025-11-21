# agents/insights_agent.py
from dotenv import load_dotenv
import os
import json
from typing import Dict, Any

# Attempt to import Google GenAI client
try:
    from google import genai
except Exception:
    genai = None


class InsightsAgent:
    """
    Uses Gemini 2.0 Flash (Google GenAI) to generate insights from
    EDA outputs and cleaning summaries.
    """

    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise EnvironmentError("GOOGLE_API_KEY not found in .env")

        if genai is None:
            raise ImportError("Install google-genai: pip install google-genai")

        # initialize client
        self.client = genai.Client(api_key=self.api_key)

        # system prompt for structured outputs
        self.system_prompt = (
            "You are a concise, professional enterprise data analyst assistant. "
            "Given structured EDA statistics, cleaning summaries, and visual artifacts, "
            "produce a clear, actionable insights JSON containing: summary, anomalies, "
            "top correlations, recommended next analyses, business impact, and a short "
            "textual executive summary. Return only JSON parsable output."
        )

    def _compact_eda(self, eda_output: Dict[str, Any], max_chars: int = 4000) -> str:
        """Summarize EDA into a compact text for LLM."""
        parts = []
        parts.append(f"shape: {eda_output.get('shape')}")

        # missing values (top 10)
        missing = eda_output.get('missing_values', {})
        if missing:
            missing_sorted = sorted(missing.items(), key=lambda x: x[1], reverse=True)[:10]
            parts.append("missing_sample: " + ", ".join([f"{k}={v}" for k, v in missing_sorted]))

        # skewness & kurtosis
        skew = eda_output.get('skewness', {})
        if skew:
            skew_sorted = sorted(skew.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
            parts.append("skewness_sample: " + ", ".join([f"{k}={round(v,3)}" for k, v in skew_sorted]))

        # outliers
        outliers = eda_output.get('outliers', {})
        if outliers:
            outlier_sorted = sorted(outliers.items(), key=lambda x: x[1].get('outlier_count', 0), reverse=True)[:10]
            parts.append("outlier_sample: " + ", ".join([f"{k}={v.get('outlier_count')}" for k, v in outlier_sorted]))

        # correlations - top 10
        corr = eda_output.get('correlation_matrix', {})
        if corr:
            pairs = []
            cols = list(corr.keys())
            for i, c1 in enumerate(cols):
                for j, c2 in enumerate(cols):
                    if j <= i:
                        continue
                    val = corr[c1].get(c2)
                    if val is not None:
                        pairs.append((c1, c2, val))
            top_pairs = sorted(pairs, key=lambda x: abs(x[2]), reverse=True)[:10]
            parts.append("top_correlations: " + ", ".join([f"{a}-{b}={round(v,3)}" for a,b,v in top_pairs]))

        compacted = "\n".join(parts)
        if len(compacted) > max_chars:
            compacted = compacted[:max_chars]
        return compacted

    def generate_insights(
        self,
        original_df,
        cleaned_df,
        eda_output: Dict[str, Any],
        cleaning_summary: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate structured insights JSON using Gemini."""
        compacted = self._compact_eda(eda_output)

        prompt = (
            f"{self.system_prompt}\n\n"
            f"INPUT_SUMMARY:\n{compacted}\n\n"
            f"CLEANING_SUMMARY:\n{json.dumps(cleaning_summary, default=str, indent=2)}\n\n"
            "TASK: Provide a JSON object with keys: executive_summary, anomalies, "
            "top_correlations, recommendations, business_impact. Return only valid JSON."
        )

        try:
            response = self.client.generate(
                model="gemini-2.0-flash",
                prompt=prompt,
                temperature=0.0,
                max_output_tokens=1024,
            )
            if hasattr(response, 'candidates'):
                response_text = response.candidates[0].content
            else:
                response_text = str(response)
        except Exception as e:
            return {"executive_summary": f"LLM call failed: {e}"}

        # parse JSON
        try:
            parsed = json.loads(response_text)
        except Exception:
            start = response_text.find('{')
            end = response_text.rfind('}')
            if start != -1 and end != -1:
                try:
                    parsed = json.loads(response_text[start:end+1])
                except Exception:
                    parsed = {"raw": response_text}
            else:
                parsed = {"raw": response_text}

        if isinstance(parsed, dict) and 'executive_summary' not in parsed:
            parsed['executive_summary'] = parsed.get('raw', '')[:500]

        return parsed

    def ask_followup(self, question: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Answer follow-up questions in chatbot UI."""
        compacted = self._compact_eda(context.get('eda', {}), max_chars=2000)
        prior_insights = context.get('insights', {})

        prompt = (
            f"System: You are a helpful data analyst.\n"
            f"CONTEXT:\n{compacted}\n\n"
            f"PRIOR_INSIGHTS:\n{json.dumps(prior_insights, default=str)[:4000]}\n\n"
            f"QUESTION: {question}\n"
            "Answer in plain text. If requesting a plot, return 'PLOT_REQUEST: plot_name'."
        )

        try:
            resp = self.client.generate(
                model="gemini-2.0-flash",
                prompt=prompt,
                temperature=0.0,
                max_output_tokens=512,
            )
            if hasattr(resp, 'candidates'):
                text = resp.candidates[0].content
            else:
                text = str(resp)
        except Exception as e:
            text = f"LLM error: {e}"

        return {"text": text}
