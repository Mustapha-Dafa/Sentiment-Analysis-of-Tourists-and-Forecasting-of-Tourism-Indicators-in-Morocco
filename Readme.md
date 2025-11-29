# PFA Project — Tourism Analysis (Ministry of Finance Internship)

Summary
-------
This project collects, cleans, and analyzes tourism-related data from Booking, Google Maps, Tripadvisor and Google Trends. It trains and evaluates forecasting and ML models (SARIMA, SARIMAX, Random Forest, Ridge, XGBoost) and provides a Streamlit dashboard for visualization and predictions.

Repository structure
--------------------
- src/
  - Booking/                # Booking scrapers and data
  - Google Maps/            # Google Maps scrapers and data
  - Google Trend/           # Pytrends scripts, raw JSON/CSV timelines
  - Tripadvisor/            # Tripadvisor scrapers and data
- notebooks/
  - chose_keyword.ipynb
  - models/                 # RF, Ridge, SARIMA, SARIMAX, XGBoost notebooks
- data/
  - reviews/                # raw & classified reviews
  - tourism/                # KPI, nights per destination, TES features, etc.
- images/ or assets/images/ # images used in reports or dashboard
- Requirements.txt          # pinned dependencies
- README.md                 # this file

Key dependencies
----------------
See Requirements.txt for pinned versions. Primary packages used:
- Python 3.10
- pandas, numpy, scipy
- scikit-learn, xgboost, statsmodels
- matplotlib, seaborn, plotly
- streamlit, streamlit-option-menu
- requests, beautifulsoup4, lxml, selenium, pytrends
- wordcloud, tqdm, joblib,openpyxl

