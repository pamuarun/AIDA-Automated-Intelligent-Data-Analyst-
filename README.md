# AI Data Analyst Assistant 🤖

An intelligent data analysis platform powered by Google Gemini AI that automates exploratory data analysis, data cleaning, and generates actionable insights from datasets with a conversational interface.

## 🌟 Key Features

- **Automated Data Analysis**: Upload CSV/Excel files and get instant EDA with visualizations
- **Intelligent Data Cleaning**: Automatic handling of missing values, duplicates, and outliers
- **AI-Powered Insights**: Conversational interface to ask questions about your data
- **Interactive Visualizations**: Beautiful charts and graphs for data exploration
- **Winsorization**: Advanced outlier treatment for numeric columns

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Google Gemini API Key
- Virtual environment (recommended)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Capstone-Project
```

2. Create and activate virtual environment:
```bash
python -m venv cap_env
source cap_env/bin/activate  # On Windows: cap_env\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
Create a `.env` file in the root directory:
```
GOOGLE_API_KEY=your_google_gemini_api_key_here
```

### Running the Application

```bash
streamlit run app.py
```

## 📁 Project Structure

```
Capstone-Project/
├── agents/
│   ├── orchestrator.py          # Main coordination logic
│   ├── eda_agent.py             # Exploratory Data Analysis
│   ├── cleaning_agent.py        # Data cleaning operations
│   ├── insights_agent.py        # AI-powered insights generation
│   └── memory/
│       ├── session_memory.py    # Session state management
│       └── long_term_memory.py  # Persistent memory storage
├── utils/
│   ├── chatbot_parser.py        # Chat message processing
│   ├── plot_utils.py            # Visualization helpers
│   └── file_utils.py            # File handling utilities
├── app.py                       # Main Streamlit application
├── requirements.txt             # Python dependencies
└── .env                         # Environment variables
```

## 💡 How It Works

1. **Upload Data**: Users upload CSV or Excel files through the intuitive interface
2. **Automatic Processing**: 
   - Dataset statistics and metadata extraction
   - Missing value detection and imputation (mean/median/mode based on distribution)
   - Duplicate identification and removal
   - Outlier detection using IQR method with Winsorization
3. **Exploratory Analysis**: 
   - Numeric distributions with histograms
   - Categorical distributions with pie charts
   - Before/after boxplots for outlier treatment
4. **AI Interaction**: 
   - Ask questions about the dataset using natural language
   - Get explanations, insights, and recommendations
   - Context-aware responses based on data characteristics

## 🎨 UI/UX Features

- **Modern Dark Theme**: Gradient backgrounds with animated elements
- **Responsive Design**: Works on various screen sizes
- **Interactive Components**: 
  - Animated KPI cards for dataset metrics
  - Floating chat interface with user/bot differentiation
  - Hover effects on all interactive elements
- **Real-time Feedback**: Progress indicators during data processing

## 🔧 Technical Stack

- **Frontend**: Streamlit for web interface
- **AI Engine**: Google Gemini 2.0 Flash model
- **Data Processing**: Pandas, NumPy, SciPy
- **Visualization**: Matplotlib, Seaborn
- **Outlier Treatment**: Feature Engine Winsorizer
- **Memory Management**: Custom session and long-term memory modules

## 📊 Data Processing Pipeline

1. **Data Ingestion**: Reads CSV/Excel files with error handling
2. **Quality Assessment**: 
   - Missing value analysis
   - Duplicate detection
   - Data type identification
3. **Statistical Analysis**: 
   - Central tendency measures (mean, median, mode)
   - Dispersion metrics (standard deviation, variance)
   - Distribution shape (skewness, kurtosis)
4. **Data Cleaning**: 
   - Intelligent missing value imputation
   - Duplicate removal
   - Outlier Winsorization
5. **Visualization Generation**: 
   - Distribution plots
   - Comparison charts
   - Cleaning effectiveness visualizations

## 🤖 AI Capabilities

- **Natural Language Processing**: Understands data-related queries
- **Context Awareness**: Remembers conversation history
- **Structured Responses**: Provides insights in organized formats
- **Visualization Recommendations**: Suggests appropriate chart types
- **Business Interpretation**: Connects statistical findings to business implications

## 🛡️ Security

- API keys stored in environment variables
- Client-side data processing (no external data transmission)
- Secure session management

## 📈 Future Enhancements

- Integration with cloud data sources (Snowflake, BigQuery)
- Advanced ML model recommendations
- Export capabilities (PDF reports, dashboards)
- Multi-language support
- Enhanced collaborative features

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Google Generative AI team for the Gemini API
- Streamlit community for the excellent framework
- Open-source data science libraries that made this possible