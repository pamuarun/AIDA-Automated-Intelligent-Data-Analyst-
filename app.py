import os
import io
import streamlit as st
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from dotenv import load_dotenv
from feature_engine.outliers import Winsorizer
from agents.orchestrator import Orchestrator
from scipy.stats import skew, kurtosis

# --------------------------------
# Streamlit Setup
# --------------------------------
st.set_page_config(page_title="AI Data Analyst Assistant", layout="wide")
st.title("AI Data Analyst Assistant")

st.markdown("""
<style>
/* --------------------- Global Gradient Background --------------------- */
.stApp {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    background-size: 400% 400%;
    animation: gradientShift 12s ease infinite;
    color: white;
}
@keyframes gradientShift {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}

/* --------------------- Floating Title Animation --------------------- */
h1 {
    text-align: center !important;
    color: #ffffff !important;
    font-size: 42px !important;
    animation: float 3s ease-in-out infinite;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 10px;
}
@keyframes float {
    0% { transform: translateY(0px);}
    50% { transform: translateY(-12px);}
    100% { transform: translateY(0px);}
}

/* --------------------- Title Bot Icon --------------------- */
h1::before {
    content: '';
    display: inline-block;
    width: 60px;
    height: 60px;
    background-image: url('https://cdn-icons-png.flaticon.com/512/14958/14958350.png');
    background-size: cover;
    background-position: center;
    margin-right: 10px;
    vertical-align: middle;
}

/* --------------------- Floating Badge --------------------- */
.floating-badge {
    display: inline-flex;
    align-items: center;
    gap: 10px;
    text-align: center;
    margin-top: -10px;
    margin-bottom: 20px;
    font-size: 18px;
    font-weight: 600;
    padding: 8px 18px;
    border-radius: 50px;
    background: linear-gradient(90deg,#06b6d4,#7c3aed);
    color: white;
    animation: floatBadge 3s ease-in-out infinite;
    box-shadow: 0 6px 18px rgba(0,0,0,0.4);
    width: fit-content;
    margin-left: auto;
    margin-right: auto;
}
.floating-badge-icon {
    width: 28px;
    height: 28px;
    background-image: url('https://cdn-icons-png.flaticon.com/512/833/833472.png');
    background-size: cover;
    background-position: center;
    border-radius: 50%;
}
@keyframes floatBadge {
    0% { transform: translateY(0px);}
    50% { transform: translateY(-6px);}
    100% { transform: translateY(0px);}
}

/* --------------------- Metrics Cards --------------------- */
div[data-testid="metric-container"] {
    background: rgba(255, 255, 255, 0.1);
    padding: 18px;
    border-radius: 18px;
    box-shadow: 0 4px 20px rgba(255, 255, 255, 0.2);
    backdrop-filter: blur(10px);
    transition: 0.3s;
    color: white;
}
div[data-testid="metric-container"]:hover {
    transform: scale(1.05);
    box-shadow: 0 0 25px rgba(255, 255, 255, 0.4);
}

/* ------------------------------------------------------------------ */
/*     PREMIUM DARK VIOLET TABLE THEME (Applied to ALL Tables)        */
/* ------------------------------------------------------------------ */

/* Table Outer Container */
.table-container {
    max-height: 350px;
    overflow-y: auto;
    background: rgba(0, 0, 0, 0.25);
    border-radius: 16px;
    padding: 10px;
    margin-top: 10px;
    box-shadow: 0 4px 20px rgba(106, 17, 203, 0.35);
}

/* General Table */
table {
    width: 100%;
    border-collapse: collapse;
    color: #e8e8e8 !important;
    border-radius: 16px;
    overflow: hidden;
}

/* Premium Gradient Header */
th {
    background: linear-gradient(120deg, #6a11cb, #2575fc) !important;
    color: white !important;
    padding: 12px !important;
    font-size: 15px !important;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    border: none !important;
    position: sticky;
    top: 0;
    z-index: 9;
}

/* Table Cells */
td {
    padding: 10px !important;
    background: rgba(255,255,255,0.05) !important;
    border-bottom: 1px solid rgba(255,255,255,0.12) !important;
}

/* Row Hover */
tr:hover td {
    background: rgba(106, 17, 203, 0.25) !important;
    transition: 0.25s ease-in-out;
}

/* Scrollbar for Table Container */
.table-container::-webkit-scrollbar {
    width: 8px;
}
.table-container::-webkit-scrollbar-thumb {
    background: #6a11cb;
    border-radius: 8px;
}
.table-container::-webkit-scrollbar-track {
    background: rgba(255,255,255,0.1);
}

/* ------------------------------------------------------------------ */

/* --------------------- Plot Styling --------------------- */
img {
    border-radius: 16px !important;
    box-shadow: 0 0 25px rgba(0, 255, 255, 0.2);
    transition: 0.3s;
}
img:hover {
    transform: scale(1.02);
    box-shadow: 0 0 35px rgba(0, 255, 255, 0.4);
}

/* --------------------- Chat Input Fixed --------------------- */
div[data-testid="stTextInput"] {
    position: fixed;
    bottom: 25px;
    left: 50%;
    transform: translateX(-50%);
    width: 70%;
    z-index: 999999;
}
input[type="text"] {
    border-radius: 25px !important;
    padding: 12px 20px !important;
    background: rgba(255,255,255,0.15) !important;
    border: 2px solid rgba(255,255,255,0.4) !important;
    color: white !important;
    font-size: 16px !important;
}
input::placeholder {
    color: rgba(255,255,255,0.6);
}

/* --------------------- Chat Bubbles --------------------- */
.stMarkdown, .stMarkdown > div, .stChatMessageContent {
    display: flex !important;
    flex-direction: column;
    width: 100% !important;
}

/* User Bubble (Right) */
.user-msg {
    display: inline-block;
    background: linear-gradient(135deg, #6f4ef2, #a65ef7);
    padding: 12px 16px;
    border-radius: 18px 18px 4px 18px;
    margin-left: auto;
    margin-right: 70px;
    color: white;
    box-shadow: 0 0 15px rgba(0,0,0,0.25);
    position: relative;
    max-width: 70%;
    text-align: right;
    word-wrap: break-word;
}
.user-msg::after {
    content: '';
    position: absolute;
    right: -70px;
    top: 50%;
    transform: translateY(-50%);
    width: 48px;
    height: 48px;
    background-image: url('https://cdn-icons-png.flaticon.com/512/17446/17446833.png');
    background-size: cover;
    background-position: center;
    border-radius: 50%;
}

/* Bot Bubble (Left) */
.bot-msg {
    display: inline-block;
    background: rgba(255,255,255,0.18);
    padding: 12px 20px;
    border-radius: 18px 18px 18px 4px;
    margin-left: 70px;
    margin-right: auto;
    color: white;
    box-shadow: 0 0 20px rgba(255,255,255,0.25);
    position: relative;
    max-width: 70%;
    text-align: left;
}
.bot-msg::before {
    content: '';
    position: absolute;
    left: -70px;
    top: 50%;
    transform: translateY(-50%);
    width: 48px;
    height: 48px;
    background-image: url('https://cdn-icons-png.flaticon.com/512/14958/14958350.png');
    background-size: cover;
    background-position: center;
    border-radius: 50%;
}

/* --------------------- File Upload Styling --------------------- */
div[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.12);
    border: 2px solid rgba(255,255,255,0.18);
    padding: 25px;
    border-radius: 22px;
    backdrop-filter: blur(12px);
    box-shadow: 0 8px 28px rgba(0, 255, 255, 0.25);
    transition: all 0.35s ease;
    animation: fadeSlideIn 0.8s ease forwards;
    margin-top: 15px;
}
div[data-testid="stFileUploader"]:hover {
    transform: scale(1.03);
    box-shadow: 0 0 35px rgba(0, 255, 255, 0.45);
    border-color: rgba(0,255,255,0.5);
}
button[kind="secondary"] {
    background: linear-gradient(90deg,#06b6d4,#7c3aed) !important;
    color: white !important;
    border-radius: 14px !important;
    padding: 10px 20px !important;
    border: none !important;
    transition: 0.3s ease;
}
button[kind="secondary"]:hover {
    transform: scale(1.05);
    box-shadow: 0 0 18px rgba(124,58,237,0.6);
}
div[data-testid="stFileDropzone"] {
    border: 2px dashed rgba(255,255,255,0.35) !important;
    border-radius: 18px !important;
    background: rgba(255,255,255,0.08) !important;
    transition: 0.3s ease;
}
div[data-testid="stFileDropzone"]:hover {
    border-color: #00eaff !important;
    background: rgba(0, 238, 255, 0.18) !important;
}
.uploadedFileName, .uploadedFileSize {
    color: #00eaff !important;
    font-weight: 600 !important;
}

@keyframes fadeSlideIn {
    0% { opacity: 0; transform: translateY(20px); }
    100% { opacity: 1; transform: translateY(0px); }
}
</style>
""", unsafe_allow_html=True)

# Floating badge below title with files icon
st.markdown(
    '<div class="floating-badge"><div class="floating-badge-icon"></div>Transforming Data Chaos Into Clarity</div>',
    unsafe_allow_html=True
)

# Initialize Orchestrator
orch = Orchestrator()

# --------------------------------
# Load Google API Key
# --------------------------------
load_dotenv(r"D:\Capstone Project\.env")
API_KEY = os.getenv("GOOGLE_API_KEY")

# --------------------------------
# LangChain Setup
# --------------------------------
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import GoogleGenerativeAI
from langchain.chains import LLMChain
# -----------------------------
# Prompt Template
# -----------------------------
FULL_PROMPT_TEMPLATE = """
You are ANALYTICS_BOT — a senior AI Data Analyst with expert-level skills in:
Data Science, Machine Learning, Statistics, EDA, Data Cleaning, Feature Engineering, Business Insights, and Data Visualization.

Follow chat history and dataset context to answer user questions.
Always provide concise points, explanations (3–5 lines), visualization suggestions, and auto insights.

Special instructions:
1. If the user asks about "types" of anything (e.g., types of data analysis, machine learning models, charts), provide:
   - A numbered or bullet list of the types.
   - A brief 2–4 line description of each type.
   - Visualization suggestions if relevant.
2. Use Markdown tables only for relevant tabular content, such as:
   - Missing values
   - Duplicates
   - Skewness or distributions comparison
   - Categorical summaries
   - Any structured dataset summaries
3. Avoid creating tables for general definitions, purposes, or types — use points/lists instead.

Handle missing values, duplicates, skewness, distributions, categorical analysis, and business interpretations.
Forbidden: politics, personal questions, jokes, entertainment.

Chat History:
{chat_history}

Dataset + User Input:
{full_input}

Final Answer (Markdown + Points + Explanation + Visualization Suggestions):
"""

prompt = PromptTemplate(
    template=FULL_PROMPT_TEMPLATE,
    input_variables=["chat_history", "full_input"]
)

memory = ConversationBufferMemory(
    memory_key="chat_history",
    input_key="full_input",
    return_messages=True
)

llm = GoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=API_KEY,
    temperature=0.2,
    max_output_tokens=1200
)

qa_chain = LLMChain(llm=llm, prompt=prompt, memory=memory)

# --------------------------------
# Helper Functions
# --------------------------------
def plot_to_bytes(fig: plt.Figure) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    plt_bytes = buf.read()
    plt.close(fig)
    return plt_bytes

def safe_st_image(plot, caption=""):
    if plot is None:
        st.info(f"No plot available for {caption}")
    else:
        st.image(plot, caption=caption, use_container_width=True)

# -------------------------
# Numeric EDA
# -------------------------
def compute_numeric_eda(df: pd.DataFrame):
    stats = pd.DataFrame(index=df.columns)
    stats['mean'] = df.mean()
    stats['median'] = df.median()
    stats['mode'] = df.mode().iloc[0]
    stats['std'] = df.std()
    stats['variance'] = df.var()
    stats['range'] = df.max() - df.min()
    stats['skewness'] = df.apply(skew)
    stats['kurtosis'] = df.apply(kurtosis)
    
    # Compute potential outliers using IQR
    outliers_dict = {}
    for col in df.columns:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5*iqr, q3 + 1.5*iqr
        outliers = ((df[col] < lower) | (df[col] > upper)).sum()
        outliers_dict[col] = outliers
    stats['potential_outliers'] = pd.Series(outliers_dict)
    
    return stats

# -------------------------
# Missing Value Imputation
# -------------------------
def impute_missing(df: pd.DataFrame, numeric_stats: pd.DataFrame):
    df_imputed = df.copy()
    for col in df.select_dtypes(include=['float64','int64']).columns:
        skewness = numeric_stats.loc[col,'skewness']
        if abs(skewness) <= 0.5:
            df_imputed[col].fillna(numeric_stats.loc[col,'mean'], inplace=True)
        else:
            df_imputed[col].fillna(numeric_stats.loc[col,'median'], inplace=True)
    for col in df.select_dtypes(include='object').columns:
        df_imputed[col].fillna(df_imputed[col].mode()[0], inplace=True)
    return df_imputed

# -------------------------
# Auto EDA Insights
# -------------------------
def generate_eda_insights(df: pd.DataFrame, numeric_cols, cat_cols, numeric_stats):
    insights = []

    # Missing values
    missing_cols = [col for col in df.columns if df[col].isnull().sum() > 0]
    if missing_cols:
        insights.append(
            "Missing Values: " + ", ".join([f"{col}({df[col].isnull().sum()})" for col in missing_cols])
        )
    else:
        insights.append("No missing values present.")

    # Skewness insights
    skewed_cols = [col for col in numeric_cols if abs(numeric_stats.loc[col,'skewness']) > 1]
    if skewed_cols:
        insights.append("Highly skewed numeric columns: " + ", ".join(skewed_cols))
    else:
        insights.append("No highly skewed numeric columns.")

    # Categorical dominance
    for col in cat_cols:
        top_pct = df[col].value_counts(normalize=True).max()
        if top_pct > 0.5:
            top_cat = df[col].value_counts().idxmax()
            insights.append(f"{col}: dominated by '{top_cat}' ({top_pct:.0%})")

    return insights

# -------------------------
# Upload Dataset
# -------------------------
st.markdown("""
<style>
/* Header Styling — gradient wraps tightly around text + icon */
#upload-dataset-header {
    font-size: 32px !important;
    font-weight: 700;
    color: white;
    padding: 8px 20px;           /* padding inside the gradient */
    border-radius: 20px;
    background: linear-gradient(135deg, #06b6d4, #7c3aed);
    display: inline-flex;         /* hug text width and allow icon */
    align-items: center;
    gap: 10px;                    /* space between icon and text */
    box-shadow: 0 6px 18px rgba(0, 0, 0, 0.5);
    animation: floatHeader 3s ease-in-out infinite;
    cursor: default;
}

/* Floating animation */
@keyframes floatHeader {
    0% { transform: translateY(0px);}
    50% { transform: translateY(-6px);}
    100% { transform: translateY(0px);}
}

/* File icon styling */
#upload-header-icon {
    width: 36px;
    height: 36px;
    background-image: url('https://cdn-icons-png.flaticon.com/512/715/715676.png'); /* File/folder icon */
    background-size: cover;
    background-position: center;
    border-radius: 50%;
}

/* File uploader styling override */
div[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.12);
    border: 2px solid rgba(0,238,255,0.4);
    padding: 25px;
    border-radius: 22px;
    backdrop-filter: blur(12px);
    box-shadow: 0 8px 28px rgba(0, 238, 255, 0.4);
    transition: all 0.35s ease;
    margin-top: 15px;
}
div[data-testid="stFileUploader"]:hover {
    transform: scale(1.03);
    box-shadow: 0 0 35px rgba(0, 238, 255, 0.6);
    border-color: #7c3aed;
}
button[kind="secondary"] {
    background: linear-gradient(90deg,#06b6d4,#7c3aed) !important;
    color: white !important;
    border-radius: 14px !important;
    padding: 10px 20px !important;
    border: none !important;
    transition: 0.3s ease;
}
button[kind="secondary"]:hover {
    transform: scale(1.05);
    box-shadow: 0 0 18px rgba(124,58,237,0.6);
}
</style>
""", unsafe_allow_html=True)

# Styled header with file icon
st.markdown('<div id="upload-dataset-header"><div id="upload-header-icon"></div>Upload Dataset</div>', unsafe_allow_html=True)

# File uploader
uploaded = st.file_uploader("Upload CSV or Excel", type=["csv","xlsx"])


if uploaded:
    import time

    # Show temporary progress bar and loading message
    progress_placeholder = st.empty()
    message_placeholder = st.empty()
    
    message_placeholder.info("Loading this file...")
    progress_bar = progress_placeholder.progress(0)
    
    for percent_complete in range(0, 101, 20):
        progress_bar.progress(percent_complete)
        time.sleep(0.2)
    
    # Clear progress and message
    progress_placeholder.empty()
    message_placeholder.empty()

    if 'original_df' not in st.session_state or uploaded.name != st.session_state.get('filename'):
        try:
            df_original = pd.read_csv(uploaded) if uploaded.name.endswith(".csv") else pd.read_excel(uploaded)
        except Exception as e:
            st.error(f"Error reading file: {e}")
            st.stop()

        st.session_state['original_df'] = df_original
        st.session_state['filename'] = uploaded.name

        # Missing values
        missing_info = [{"Column": col, "Missing Before": df_original[col].isnull().sum()} 
                        for col in df_original.columns]
        st.session_state['missing_df'] = pd.DataFrame(missing_info)

        # Numeric EDA & Imputation
        numeric_cols = df_original.select_dtypes(include=['float64','int64']).columns.tolist()
        numeric_stats = compute_numeric_eda(df_original[numeric_cols]) if numeric_cols else pd.DataFrame()
        df_cleaned = impute_missing(df_original, numeric_stats)

        st.session_state['cleaned_df'] = df_cleaned
        st.session_state['numeric_stats'] = numeric_stats

        # Winsorization
        if numeric_cols:
            winsor = Winsorizer(capping_method='iqr', tail='both', fold=1.5, variables=numeric_cols)
            df_winsorized = df_cleaned.copy()
            df_winsorized[numeric_cols] = winsor.fit_transform(df_cleaned[numeric_cols])
            st.session_state['df_winsorized'] = df_winsorized
        else:
            st.session_state['df_winsorized'] = df_cleaned

        # Run Full EDA Pipeline
        with st.spinner("Running EDA + Cleaning + Insights..."):
            result = orch.run_full_pipeline(df_cleaned, run_cleaning_after_eda=False)
            st.session_state['eda_plots'] = result.get("eda", {}).get("visualizations_bytes", {})
            st.session_state['cleaning_report'] = result.get("cleaning_summary", {})

    else:
        df_original = st.session_state['original_df']
        df_cleaned = st.session_state['cleaned_df']
        numeric_stats = st.session_state.get('numeric_stats', pd.DataFrame())
        eda_plots = st.session_state.get('eda_plots', {})
        cleaning_report = st.session_state.get('cleaning_report', {})

    # Temporary "Loaded" message
    loaded_placeholder = st.empty()
    loaded_placeholder.success(f"Loaded {uploaded.name} → {df_original.shape[0]} rows × {df_original.shape[1]} columns")
    time.sleep(2)
    loaded_placeholder.empty()

    # -------------------------
    # Dashboard Overview (Dynamic KPI Cards)
    # -------------------------
    numeric_cols = df_original.select_dtypes(include=['float64','int64']).columns.tolist()
    cat_cols = df_original.select_dtypes(include='object').columns.tolist()
    rows = df_original.shape[0]
    columns_count = df_original.shape[1]
    numeric_count = len(numeric_cols)
    categorical_count = len(cat_cols)

    # Styled heading
    st.markdown('<h2 style="text-align:center; color:#00eaff; font-size:36px; margin-top:30px;">📊 Dataset Overview</h2>', unsafe_allow_html=True)

    # KPI Cards HTML + CSS
    kpi_html = f"""
    <style>
    .kpi-container {{
        display: flex;
        justify-content: center;
        gap: 25px;
        flex-wrap: wrap;
        margin-top: 20px;
    }}
    .kpi-card {{
        background: rgba(255,255,255,0.08);
        color: white;
        padding: 20px 25px;
        border-radius: 18px;
        text-align: center;
        width: 180px;
        box-shadow: 0 4px 15px rgba(255, 255, 255, 0.2);
        transition: all 0.3s ease;
        font-family: 'Arial', sans-serif;
        animation: floatCard 3s ease-in-out infinite;
    }}
    .kpi-card:hover {{
        transform: translateY(-12px) scale(1.05);
        box-shadow: 0 0 25px rgba(0, 238, 255, 0.6);
    }}
    .kpi-value {{
        font-size: 34px;
        font-weight: 700;
        color: #00eaff;
    }}
    .kpi-label {{
        font-size: 16px;
        font-weight: 500;
        margin-top: 5px;
        color: #ffffffcc;
    }}
    @keyframes floatCard {{
        0% {{ transform: translateY(0px); }}
        50% {{ transform: translateY(-8px); }}
        100% {{ transform: translateY(0px); }}
    }}
    </style>

    <div class="kpi-container">
        <div class="kpi-card">
            <div class="kpi-value">{rows}</div>
            <div class="kpi-label">Rows</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-value">{columns_count}</div>
            <div class="kpi-label">Columns</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-value">{numeric_count}</div>
            <div class="kpi-label">Numeric Columns</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-value">{categorical_count}</div>
            <div class="kpi-label">Categorical Columns</div>
        </div>
    </div>
    """
    st.markdown(kpi_html, unsafe_allow_html=True)

    # -------------------------
    # Columns & Data Types (Styled)
    # -------------------------
    st.markdown('<h3 style="text-align:center; color:#00eaff; margin-top:30px;">Columns & Data Types</h3>', unsafe_allow_html=True)

    columns_df = df_original.dtypes.to_frame(name="Type").reset_index().rename(columns={"index": "Column"})
    columns_table_html = columns_df.to_html(index=False, classes="styled-table", border=0)  
    st.markdown(f"""
<style>
.table-container {{
    max-height: 300px;
    overflow-y: auto;
    background: rgba(255,255,255,0.05);
    border-radius: 15px;
    padding: 10px;
    box-shadow: 0 4px 15px rgba(0, 238, 255, 0.2);
}}
.styled-table {{
    width: 100%;
    border-collapse: collapse;
    text-align: center;
    font-family: 'Arial', sans-serif;
}}
.styled-table th {{
    background: rgba(0, 238, 255, 0.2);
    color: #00eaff;
    padding: 8px;
    position: sticky;
    top: 0;
}}
.styled-table td {{
    padding: 8px;
    color: white;
    border-bottom: 1px solid rgba(255,255,255,0.1);
}}
.styled-table tr:hover {{
    background: rgba(0, 238, 255, 0.1);
}}
</style>
<div class="table-container">
{columns_table_html}
</div>
""", unsafe_allow_html=True)

    # -------------------------
    # Numeric EDA Insights (Styled)
    # -------------------------
    if not numeric_stats.empty:
        st.markdown('<h3 style="text-align:center; color:#00eaff; margin-top:30px;">Numeric EDA Insights</h3>', unsafe_allow_html=True)
        numeric_table_html = numeric_stats.reset_index().rename(columns={"index": "Column"}).to_html(index=False)
        st.markdown(f"""
        <div class="table-container">
        {numeric_table_html}
        </div>
        """, unsafe_allow_html=True)

    # -------------------------
    # Duplicates & Missing Values (Styled)
    # -------------------------
    st.markdown('<h3 style="text-align:center; color:#00eaff; margin-top:30px;">Data Cleaning Overview</h3>', unsafe_allow_html=True)

    duplicates_df = pd.DataFrame({
        "Type": ["Duplicates"],
        "Before": [df_original.duplicated().sum()],
        "After": [df_cleaned.duplicated().sum()]
    })
    st.markdown('<h5 style="text-align:center; color:#ffffffcc;">Duplicates</h5>', unsafe_allow_html=True)
    duplicates_html = duplicates_df.to_html(index=False)
    st.markdown(f"""
    <div class="table-container" style="max-height:150px;">
    {duplicates_html}
    </div>
    """, unsafe_allow_html=True)

    missing_df_full = st.session_state.get('missing_df', pd.DataFrame())
    missing_info_full = []
    for col in df_original.columns:
        missing_before = df_original[col].isnull().sum()
        missing_after = df_cleaned[col].isnull().sum()
        imputation_method = ""
        reason = ""
        if missing_before > 0 and col in numeric_cols:
            skewness = numeric_stats.loc[col,'skewness']
            if abs(skewness) <= 0.5:
                imputation_method = "Mean"
                reason = f"Skewness = {skewness:.2f} → symmetric distribution"
            else:
                imputation_method = "Median"
                reason = f"Skewness = {skewness:.2f} → skewed distribution"
        elif missing_before > 0 and df_original[col].dtype == 'object':
            imputation_method = "Mode"
            reason = "Categorical column → filled with mode"
        missing_info_full.append({
            "Column": col,
            "Missing Before": missing_before,
            "Missing After": missing_after,
            "Imputation Method": imputation_method,
            "Reason": reason
        })
    missing_df_full = pd.DataFrame(missing_info_full)
    st.markdown('<h5 style="text-align:center; color:#ffffffcc;">Missing Values & Imputation</h5>', unsafe_allow_html=True)
    missing_df_html = missing_df_full.to_html(index=False)
    st.markdown(f"""
    <div class="table-container" style="max-height:300px;">
    {missing_df_html}
    </div>
    """, unsafe_allow_html=True)

    # -------------------------
    # Auto Insights Section
    # -------------------------
    st.markdown('<h3 style="text-align:center; color:#00eaff; margin-top:30px;">💡 EDA Insights</h3>', unsafe_allow_html=True)

    eda_points = generate_eda_insights(df_cleaned, numeric_cols, cat_cols, numeric_stats)

    # Combine all cards into a single HTML container for proper flex layout
    eda_cards_html = """
    <style>
    .eda-container {
        display: flex;
        flex-wrap: wrap;
        gap: 20px;
        justify-content: center;
        margin-top: 15px;
    }
    .eda-card {
        background: rgba(255, 255, 255, 0.08);
        color: white;
        padding: 15px 20px;
        border-radius: 15px;
        min-width: 200px;
        max-width: 250px;
        box-shadow: 0 4px 15px rgba(255, 255, 255, 0.2);
        transition: all 0.3s ease;
        font-family: 'Arial', sans-serif;
        font-size: 14px;
        animation: floatCard 3s ease-in-out infinite;
    }
    .eda-card:hover {
        transform: translateY(-10px) scale(1.05);
        box-shadow: 0 0 25px rgba(0, 238, 255, 0.6);
    }
    @keyframes floatCard {
        0% { transform: translateY(0px); }
        50% { transform: translateY(-6px); }
        100% { transform: translateY(0px); }
    }
    </style>
    <div class="eda-container">
    """

    for point in eda_points:
        eda_cards_html += f'<div class="eda-card">{point}</div>'

    eda_cards_html += "</div>"

    st.markdown(eda_cards_html, unsafe_allow_html=True)


    # -------------------------
    # Boxplots, Histograms, Pie Charts
    # -------------------------
    if numeric_cols:
        st.subheader("Boxplots: Before Winsorization vs After Winsorization")
        for i in range(0, len(numeric_cols), 2):
            cols = st.columns(2)
            for j, col_name in enumerate(numeric_cols[i:i+2]):
                with cols[j]:
                    fig, axes = plt.subplots(1,2,figsize=(6,2.5))
                    sns.boxplot(x=df_cleaned[col_name], ax=axes[0], color='skyblue')
                    axes[0].set_title('Before Winsorization', fontsize=10)
                    sns.boxplot(x=st.session_state['df_winsorized'][col_name], ax=axes[1], color='orange')
                    axes[1].set_title('After Winsorization', fontsize=10)
                    plt.tight_layout()
                    st.pyplot(fig)

    if numeric_cols:
        st.subheader("Numeric Distributions")
        for i in range(0, len(numeric_cols), 2):
            cols = st.columns(2)
            for j, col_name in enumerate(numeric_cols[i:i+2]):
                with cols[j]:
                    fig, ax = plt.subplots(figsize=(4,2.5))
                    sns.histplot(df_cleaned[col_name], kde=True, ax=ax, color='skyblue')
                    ax.set_title(f"{col_name} Histogram", fontsize=10)
                    plt.tight_layout()
                    st.pyplot(fig)

    if cat_cols:
        st.subheader("Categorical Distributions")
        for i in range(0, len(cat_cols), 2):
            cols = st.columns(2)
            for j, col_name in enumerate(cat_cols[i:i+2]):
                with cols[j]:
                    fig, ax = plt.subplots(figsize=(4,2.5))
                    df_original[col_name].value_counts().plot.pie(autopct="%1.1f%%", ax=ax)
                    ax.set_ylabel("")
                    ax.set_title(f"{col_name} Distribution", fontsize=10)
                    plt.tight_layout()
                    st.pyplot(fig)

    # -------------------------
    # Other Visualizations
    # -------------------------
    st.subheader("Other Visualizations")
    all_plots = {**st.session_state.get('eda_plots', {}), 
                 **st.session_state.get('cleaning_report', {}).get("visualizations_bytes", {})}
    plot_keys = list(all_plots.keys())
    plots_per_row = 2

    for i in range(0, len(plot_keys), plots_per_row):
        cols = st.columns(plots_per_row)
        for j, key in enumerate(plot_keys[i:i+plots_per_row]):
            with cols[j]:
                safe_st_image(all_plots[key], caption=key.replace("_"," ").title())
                # -------------------------------
# Chat Section
# -------------------------------
st.markdown("---")

st.markdown("""
<style>
/* Chat header styling — gradient wraps tightly around text, includes icon */
#chat-header {
    font-size: 28px !important;
    font-weight: 700;
    color: white;
    padding: 8px 20px;           /* padding inside the gradient */
    border-radius: 20px;
    background: linear-gradient(135deg, #06b6d4, #7c3aed);
    display: inline-flex;        /* hug text width and allow icon */
    align-items: center;
    gap: 10px;
    box-shadow: 0 6px 18px rgba(0, 0, 0, 0.5);
    animation: floatHeader 3s ease-in-out infinite;
    cursor: default;
}

/* Floating animation */
@keyframes floatHeader {
    0% { transform: translateY(0px);}
    50% { transform: translateY(-6px);}
    100% { transform: translateY(0px);}
}

/* Icon styling */
#chat-header-icon {
    width: 32px;
    height: 32px;
    background-image: url('https://cdn-icons-png.flaticon.com/512/2910/2910763.png');
    background-size: cover;
    background-position: center;
    border-radius: 50%;
}
</style>
""", unsafe_allow_html=True)

# Styled header with icon
st.markdown('<div id="chat-header"><div id="chat-header-icon"></div>Ask Questions about your Dataset</div>', unsafe_allow_html=True)

# Custom CSS for chat bubbles and input box
st.markdown("""
<style>
/* Chat wrapper */
.chat-wrapper {
    display: flex;
    width: 100%;
    margin-bottom: 10px;
}

/* User message */
.user-msg {
    background: rgba(0, 238, 255, 0.2);
    color: #00eaff;
    padding: 10px 15px;
    border-radius: 15px;
    max-width: 75%;
    margin-left: auto;
    font-family: 'Arial', sans-serif;
    box-shadow: 0 4px 15px rgba(0, 238, 255, 0.2);
    animation: floatCard 3s ease-in-out infinite;
}

/* Bot message */
.bot-msg {
    background: rgba(255, 255, 255, 0.08);
    color: white;
    padding: 10px 15px;
    border-radius: 15px;
    max-width: 75%;
    margin-right: auto;
    font-family: 'Arial', sans-serif;
    box-shadow: 0 4px 15px rgba(255, 255, 255, 0.2);
    animation: floatCard 3s ease-in-out infinite;
}

@keyframes floatCard {
    0% { transform: translateY(0px); }
    50% { transform: translateY(-4px); }
    100% { transform: translateY(0px); }
}

/* Chat input box styling */
div.stTextInput > label {
    color: #00eaff;
    font-weight: 500;
}
div.stTextInput > div > input {
    background: rgba(255,255,255,0.05);
    color: white;
    border: 1px solid rgba(0, 238, 255, 0.4);
    border-radius: 12px;
    padding: 10px 15px;
    font-size: 16px;
    width: 100%;
    transition: all 0.3s ease;
}
div.stTextInput > div > input:focus {
    border: 1px solid #00eaff;
    box-shadow: 0 0 15px rgba(0, 238, 255, 0.5);
    outline: none;
}
</style>
""", unsafe_allow_html=True)

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

# Chat input
user_input = st.text_input(
    "",
    placeholder="Type your question here and press Enter",
    key="chat_input"
)

if user_input:
    st.session_state.chat_messages.append({"role": "user", "content": user_input})

    # Quick greetings
    if user_input.lower().strip() in ["hi", "hello", "hey"]:
        st.session_state.chat_messages.append({
            "role": "assistant",
            "content": "Hello! 👋 How can I help you today?"
        })
    else:
        df_original = st.session_state.get("original_df", pd.DataFrame())
        df_cleaned = st.session_state.get("cleaned_df", df_original)

        numeric_cols = df_original.select_dtypes(include="number").columns.tolist()
        cat_cols = df_original.select_dtypes(include="object").columns.tolist()

        missing_df = st.session_state.get("missing_df", pd.DataFrame())
        missing_info_text = (
            "No missing values present."
            if missing_df.empty or missing_df['Missing Before'].sum() == 0
            else missing_df.to_dict(orient="records")
        )

        numeric_stats = st.session_state.get("numeric_stats", pd.DataFrame())
        numeric_summary = numeric_stats[
            numeric_stats.index.isin(numeric_cols)
        ][['mean', 'median', 'skewness']].to_dict(orient='records')

        context_text = f"""
Dataset shape: {df_original.shape}
Columns: {list(df_original.columns)}
Numeric: {numeric_cols}
Categorical: {cat_cols}

Missing summary: {missing_info_text}
Numeric summary: {numeric_summary}

Instructions:
- Show concise insights.
- Do not recompute winsorization.
- Follow-up questions must use chat_history.
- You are an expert Data Analyst. Think step-by-step.
"""

        chain_input = {
            "full_input": context_text + "\nUser Question:\n" + user_input,
            "chat_history": st.session_state.chat_messages
        }

        with st.spinner("Generating insights..."):
            try:
                output = qa_chain(chain_input)
                reply = output.get("text") if isinstance(output, dict) else str(output)
            except Exception as e:
                reply = f"⚠️ Error generating response: {e}"

        st.session_state.chat_messages.append({"role": "assistant", "content": reply})

# Display chat messages
st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)
chat_container = st.container()

with chat_container:
    for msg in st.session_state.chat_messages:
        st.markdown("<div class='chat-wrapper'>", unsafe_allow_html=True)
        if msg["role"] == "user":
            st.markdown(f"<div class='user-msg'>{msg['content']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='bot-msg'>{msg['content']}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
