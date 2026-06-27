import os 
from pathlib import Path
from dotenv import load_dotenv 

load_dotenv()

# PATHS

BASE_PATH = Path(__file__).resolve().parent.parent 
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))
DB_PATH = Path(os.getenv("DB_PATH",BASE_DIR / "data" / "finance.db"))
LOG_DIR = Path(os.getenv("LOG_DIR",BASE_DIR / "logs"))

DATA_DIR.mkdir(parents=True,exist_ok=True)
LOG_DIR.mkdir(parents=True,exist_ok=True)

# API KEYS
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY","")
FRED_API_KEY = os.getenv("FED_API_KEY","")

#Universe
TICKERS = os.getenv("TICKERS",'AAPL,MSFT,TSLA,GOOGL,AMZN,NVDA').splits(",")
START_DATE = os.getenv("START_DATE",'2026-01-01')
END_DATE = os.getenv('END_DATE','2026-06-27')

#NEW SOURCE
NEWS_SOURCE      = os.getenv("NEWS_SOURCE", "yfinance_news")
KAGGLE_NEWS_CSV  = os.getenv("KAGGLE_NEWS_CSV", str(DATA_DIR / "news_kaggle.csv"))

# FinBERT 
FINBERT_MODEL      = os.getenv("FINBERT_MODEL", "ProsusAI/finbert")
FINBERT_BATCH_SIZE = int(os.getenv("FINBERT_BATCH_SIZE", 32))
FINBERT_MAX_LENGTH = int(os.getenv("FINBERT_MAX_LENGTH", 512))

# FRED macro series to pull ──────────────────────────────────────────────────
FRED_SERIES = {
    "fed_funds_rate": "FEDFUNDS",   # monthly
    "cpi":            "CPIAUCSL",   # monthly
    "unemployment":   "UNRATE",     # monthly
    "yield_10y":      "DGS10",      # daily
    "yield_2y":       "DGS2",       # daily
    "vix":            "VIXCLS",     # daily
    "sp500":          "SP500",      # daily
}

# ── Feature engineering 
RSI_WINDOW         = 14
MACD_FAST          = 12
MACD_SLOW          = 26
MACD_SIGNAL        = 9
BB_WINDOW          = 20
EMA_SHORT          = 20
EMA_LONG           = 50
ATR_WINDOW         = 14
PREDICTION_HORIZON = 5   # predict price direction N days ahead

# SQLite table names 
TABLE_PRICE      = "price_data"
TABLE_FEATURES   = "features"
TABLE_SENTIMENT  = "news_sentiment"
TABLE_MACRO      = "macro_data"
TABLE_MERGED     = "feature_store"

#  Source credibility weights for sentiment aggregation 
SOURCE_WEIGHTS = {
    "Reuters":       1.5,
    "Bloomberg":     1.5,
    "Financial Times": 1.4,
    "Wall Street Journal": 1.4,
    "CNBC":          1.2,
    "MarketWatch":   1.1,
    "Seeking Alpha": 0.9,
    "default":       1.0,
}
