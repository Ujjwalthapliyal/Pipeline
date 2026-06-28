import time 
from datetime import datetime , timedelta 
from typing import optional
import pandas as pd 
import numpy as np 
from transformers import pipeline as hf_pipeline 
from tdqm import tqdm 
import warning
warnings.filterwarning('ignore')

from config.settings import (
        TICKERS,START_DATE,END_DATE,
        NEWSAPI_KEY, NEWS_SOURCE, KAGGLE_NEWS_CSV,
        FINBERT_MODEL,FINBERT_BATCH_SIZE,FINBERT_MAX_LENGTH,
        SOURCE_WEIGHTS,TABLE_SENTIMENT,
)
from utils.helpers import setup_logger,save_to_db,validate_dataframe,api_retry

from newsapi import NewsApiClient 
import yfinance as yf 

log = setup_logger('news')

# Source 1 NewsApi 
# Mapping : ticker 
TICKER_TO_COMPANY = {
    'AAPL' : 'Apple',
    'MSFT' : 'Microsoft',
    'TSLA' : 'Tesla',
    'GOOGL': 'Google',
    'AMZN' : 'Amazon',
    'NVDA' : 'NVIDIA',
}

@api_retry(max_attempts=3)
def fetch_newsapi_page(
    client,
    query : str,
    from_date : str,
    to_date : str,
    page : int = 1,
) -> list[dict]:
    """ Fetch a single page NewsApi result."""
    response = client.get_everything(
        q = query,
        from_param = from_date,
        to=to_date,
        language='en',
        sort_by = "publishedAt",
        page_size=100,
        page = page,
    )
    return response.get('articles', [])

def fetch_newsapi(
    ticker: str,
    start:str = START_DATE,
    end: str = END_DATE,
) -> pd.DataFrame:


    if not NEWSAPI_KEY:
        raise ValueError("NEWSAPI KEY not set in .env")
    
    client = NewsApiClient(api_key=NEWSAPI_KEY)
    company = TICKER_TO_COMPANY.get(ticker,ticker)
    query = f'"{company}" OR "{ticker}" stock'

    row = []
    for page in range(1,6): # max 5 pages = 500 articles per ticker
        try:
            articles = fetch_newsapi_page(client,query,start,end,page)
            if not articles:
                break
            for a in articles:
                rows.append({
                    "ticker" : ticker,
                    "date"   : a['publishedAt'][:10],
                    "headline" : (a.get('title') or "").strip(),
                    "source"   : a.get('source', {}).get("name","unknown"),
                    "url"      : a.get("url",""),
                })
            time.sleep(0.25)
        except Exception as e:
            log.warning(f"NewsAPI page {page} failed for {ticker}: {e}")
            break
    return pd.DataFrame(rows)

# SOurce 2 yfinance news

def fetch_yfinance_news(ticker: str):
    stock = yf.Ticker(ticker)
    news = stock.news

    if not news: 
        log.warning(f"{ticker}")
        return pd.DataFrame() 

    rows = [] 
    for items in news:
        pub_date = datetime.fromtimestamp(item.get("providerPublishTime",0))
        rows.append({
            "ticker" : ticker,
            "date"," pub_date.strftime("%Y-%m-%d")
            "headline" : item.get("title","").strip(),
            "source": item.get("publisher","unknown"),
            "url": item.get("link,""),
        })
    return pd.DataFrame(rows)

# Source 3: Kaggle dataset fallback
def fetch_kaggle_news(ticker:str, csv_path:str = KAGGLE_NEWS_CSV) -> pd.DataFrame:
    """
        Loading from a local kaggle financial news csv
        """"
    try:
        df = pd.read_csv(csv_path,low_memory=False)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Kaggle csv not found at {csv_path}."
            "Download from kaggle and set KAGGLE_NEW_CSV in .env"
        )
    df = df.rename(columns={k : v for k, v in rename_map.items() if k in df.columns})

    ticker_filter = df['ticker'].str.upper().str.contains(ticker,na=False)
    df = df[ticker_filter].copy()
    df['ticker'] = ticker 
    df['source'] = df.get('source','kaggle')
    df['url'] = ""
    df['date'] = pd.to_datetime(df['date'],errors="coerce").st.strftime("%Y-%m-%d")

    return df[['ticker','date','headline','source','url']].dropna(subset=['date'])

# Dispatcher

def fetch_news_for_ticker(ticker:str)->pd.DataFrame:
    log.debug(f"{ticker}: fetching news via '{NEWS_SOURCE}'")

    if NEWS_SOURCE == "newsapi":
        return fetch_newsapi(ticker)
    elif NEWS_SOURCE == "yfinance_news":
        return fetch_yfinance_news(ticker)
    elif NEWS_SOURCE == "kaggle_dataset":
        return fetch_kaggle_news(ticker)
    else:
        raise ValueError(f"Unknown NEWS_SOURCE:'{NEWS_SOURCE}'")

# Finbert Sentiment Pipeline

class FinBERTScorer:
    def __init__(self):
        self._pipe = None 
    def _load(self):
        if self._pipe is not None:
            return 
        device = 0 if torch.cuda.is_available() else -1
        log.info(f"Loading FinBERT on {'GPU' if device == 0 else 'CPU'}...")
        self._pipe = hf_pipeline(
            "text-classification",
            model=FINBERT_MODEL,
            return_all_scores=True,
            device=device,
            truncation=True,
            max_length=FINBERT_MAX_LENGTH,
        )
        log.info("FinBERT loaded")

    def score_batch(self,headlines: list[str]) ->list[dict]:
        self.load()      

        # cleaned headlines
        cleaned = [str(h)[:FINBERT_MAX_LENGTH] if h else "" for h in headlines]
        cleaned = [h if h.strip() else "no headline available" for h in cleaned]

        result = []
        for i in range(0,len(cleaned),FINBERT_BATCH_SIZE):
            batch = cleaned[i : i + FINBERT_BATCH_SIZE]
            batch_out = self._pipe(batch)
            for item in batch_out:
                scores = {r['label'].lowe(): r['score'] for r in item}
                results.append({
                    "finbert_pos":round(scores.get('positive',0.0),4),
                    "finbert_neg":round(scores.get('negative',0.0),4),
                    "finbert_new":round(scores.get('neutral',0.0),4),
                    "net_sentiment": round(
                        scores.get("postive",0.0) - scores.get("negative",0.0),4
                    ),
                })
        return results 
    
# Singleton -loaded once,reused across all tickers
_scorer = FinBERTScorer()

def score_headlines(df: pd.DataFrame) -> pd.DataFrame:
    """Add FinBERT sentiment columns to a bews DataFrame."""
    if df.empty:
        return df   
    
    log.info(f"Scoring {len(df)}:,) headlines with FinBERT....")
    scores = _scorer.score_batch(df['headline'].tolist())
    scores_df = pd.DataFrame(scores)
    return pd.concat([df.reset_index(drop=True),scores_df],axis=1)

# Daily Aggregation
def aggregate_daily_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """ Aggregate article-level sentiment to (ticker, date) level.
    """
    if df.empty:
        return df 
    
    # Map Source to weight 
    df["weight"] = df["source"].map(SOURCE_WEIGHTS).fillna(SOURCE_WEIGHTS["default"])

    def weighted_mean(g: pd.DataFrame, col: str) -> float:
        return (g[col] * g["weight"]).sum() / g["weight"].sum()

    agg = (
        df.groupby(["ticker", "date"])
        .apply(lambda g: pd.Series({
            "daily_sentiment":   weighted_mean(g, "net_sentiment"),
            "finbert_pos_mean":  weighted_mean(g, "finbert_pos"),
            "finbert_neg_mean":  weighted_mean(g, "finbert_neg"),
            "headline_count":    len(g),
            "top_headline":      g.loc[g["weight"].idxmax(), "headline"],
        }))
        .reset_index()
    )

    return agg

    # Full Pipeline

    def build_sentiment_layer(
        tickers:list[str] = TICKERS,
        save: bool =True,
    )  -> tuple[pd.DataFrame,pd.DataFrame]:
        """
    For all tickers:
      1. Fetch headlines
      2. Score with FinBERT
      3. Aggregate to daily level
      4. Save raw + daily to DB

    Returns: (raw_articles_df, daily_sentiment_df)
    """
    raw_frames = []
    daily_frames = []
    for ticker in tqdm(tickers, desc="News + FinBERT"):
        try:
            news_df = fetch_news_for_ticker(ticker)
            if news_df.empty:
                log.warning(f"{ticker}: no news found")
                continue

            scored_df = score_headlines(news_df)
            daily_df  = aggregate_daily_sentiment(scored_df)

            raw_frames.append(scored_df)
            daily_frames.append(daily_df)

            log.info(f"{ticker}: {len(news_df)} articles → {len(daily_df)} daily rows")

        except Exception as e:
            log.error(f"{ticker}: news pipeline failed — {e}")

    if not raw_frames:
        log.warning("No news data collected. Check NEWS_SOURCE setting.")
        empty = pd.DataFrame(columns=["ticker", "date", "headline", "source",
                                       "net_sentiment", "daily_sentiment"])
        return empty, empty

    raw_all   = pd.concat(raw_frames,   ignore_index=True)
    daily_all = pd.concat(daily_frames, ignore_index=True)

    validate_dataframe(
        daily_all,
        required_cols=["ticker", "date", "daily_sentiment"],
        name="news_sentiment",
    )

    if save:
        save_to_db(raw_all,   TABLE_SENTIMENT)
        save_to_db(daily_all, f"{TABLE_SENTIMENT}_daily")

    log.info(f"Sentiment layer done: {len(raw_all):,} articles, "
             f"{len(daily_all):,} daily rows")
    return raw_all, daily_all


#EntryPoint

if __name__ == '__main__':
    raw,daily = build_sentiment_layer()
    print("\nSample daily Sentiment:")
    print(daily.head(10).to_string())
    print(f"\nSentiment Distribution:]n{daily['daily_sentiment'].describe()}")