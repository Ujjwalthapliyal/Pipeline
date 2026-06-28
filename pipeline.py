# Orchestration run all three ingestion modules and merges them in a single
# unified features store ready for model training

# Merge Logic:
# features(yfinance) <- left join -> dailySentiment( NEwsAPi/Finbert)
#                     left join -> macro(FRED)
# All joins on date: (+ ticker for sentiment)

# Left join preserve all trading days even when news ot macro data is missing.
# Missing sentiment filled with 0 (neutral)/
# Missing macro filled with last known value(forward fill after merge)

# Final Table is what used by model xgboost, lstm and rag agents all read form.

import pandas as pd 
from pathlib import Path 

from config.setting import ( 
    TICKERS,START_DATE,END_DATE,
    TABLE_FEATUES,TABLE_SENTIMENT,TABLE_MACRO,TABLE_MERGED,
    RUN_YFINANCE := "RUN_YFINANCE"
)
from utils.helpers import(
    setup_logger, save_to_db,load_from_db, validate_dataframe,
)

from data_pipeline.yfinance_ingestion import build_feature_store_yfinance
from data_pipeline.news_ingestion import build_sentiment_layer 
from data_pipeline.fred_ingestion import build_macro_features 
import os 

log = setup_logger('pipeline')

# Individual runners

def run_yfinance_pipeline() -> pd.DataFrame:
    flag = os.getenv("RUN_YFINANCE", "true").lower() == "true"
    if not flag:
        log.info("RUN_YFINANCE=false — loading from DB")
        return load_from_db(TABLE_FEATURES)
    log.info("Phase 1: yfinance OHLCV + Features ===")
    return build_feature_store_yfinance(save=True)


def run_news_pipeline() -> pd.DataFrame:
    flag = os.getenv("RUN_NEWS", "true").lower() == "true"
    if not flag:
        log.info("RUN_NEWS=false — loading from DB")
        return load_from_db(f"{TABLE_SENTIMENT}_daily")
    log.info("=== Phase 2: News + FinBERT Sentiment ===")
    _, daily = build_sentiment_layer(save=True)
    return daily


def run_fred_pipeline() -> pd.DataFrame:
    flag = os.getenv("RUN_FRED", "true").lower() == "true"
    if not flag:
        log.info("RUN_FRED=false — loading from DB")
        return load_from_db(TABLE_MACRO)
    log.info("=== Phase 3: FRED Macro Features ===")
    return build_macro_features(save=True)

# Merge logic

def merge_feature_store(
    features_df: pd.DataFrame,
    sentiment_df: pd.DataFrame,
    macro_df: pd.DataFrame,
) -> pd.DataFrame:

log.info("Merging feature store...")

# Ensuring that date columns are string for consistent join

features_df['date'] = features_df['date'].astype(str)

merged = features_df.copy()

# Join sentiment 
    if not sentiment_df.empty:
        sentiment_df["date"] = sentiment_df["date"].astype(str)
        # Keep only columns that add value (avoid duplicating ticker/date)
        sent_cols = ["ticker", "date", "daily_sentiment", "finbert_pos_mean",
                     "finbert_neg_mean", "headline_count", "top_headline"]
        sent_cols = [c for c in sent_cols if c in sentiment_df.columns]
        merged = merged.merge(
            sentiment_df[sent_cols],
            on=["ticker", "date"],
            how="left",
        )
        # Fill no-news days with 0 (neutral)
        for col in ["daily_sentiment", "finbert_pos_mean", "finbert_neg_mean"]:
            if col in merged.columns:
                merged[col] = merged[col].fillna(0.0)
        merged["headline_count"] = merged.get("headline_count", pd.Series(0)).fillna(0).astype(int)
        log.info(f"After sentiment join: {len(merged):,} rows")
    else:
        log.warning("Sentiment DataFrame is empty — adding zero-filled columns")
        merged["daily_sentiment"]   = 0.0
        merged["finbert_pos_mean"]  = 0.0
        merged["finbert_neg_mean"]  = 0.0
        merged["headline_count"]    = 0


# JOin Macro
    if not macro_df.empty:
        macro_df["date"] = macro_df["date"].astype(str)
        macro_cols = [c for c in macro_df.columns if c != "ticker"]
        merged = merged.merge(macro_df[macro_cols], on="date", how="left")

        # Forward fill macro (handles weekends, holidays, lag gaps)
        macro_feature_cols = [c for c in macro_df.columns if c != "date"]
        merged[macro_feature_cols] = (
            merged.sort_values(["ticker", "date"])
            .groupby("ticker")[macro_feature_cols]
            .ffill()
        )
        log.info(f"After macro join: {len(merged):,} rows")
    else:

    # Final cleanup , drop rows where core features are all nan
    core_cols = ['close','rsi','macd','target']
    merged = merged.dropna(subset=[c for c in core_cols if c in merged.columns])

    # time series integrity
    merged = merged.sort_values(['ticker','date']).reset_index(drop=True)
    return merged 

# Pipeline report

def print_pipeline_report(df: pd.DataFrame) -> None:
    """Print a summary of the merged feature store."""
    print("\n" + "=" * 60)
    print("  SMART FINANCE ANALYST — Feature Store Report")
    print("=" * 60)
    print(f"  Total rows:       {len(df):,}")
    print(f"  Tickers:          {df['ticker'].nunique()} → {sorted(df['ticker'].unique())}")
    print(f"  Date range:       {df['date'].min()} → {df['date'].max()}")
    print(f"  Total features:   {df.shape[1]}")
    print(f"  Target balance:   {df['target'].mean():.1%} positive days")

    print("\n  Null % (top offenders):")
    null_pct = df.isnull().mean().sort_values(ascending=False)
    for col, pct in null_pct[null_pct > 0.01].items():
        print(f"    {col:<30} {pct:.1%}")

    print("\n  Feature groups:")
    groups = {
        "Price/OHLCV":     ["open", "high", "low", "close", "volume"],
        "Momentum":        ["rsi", "stoch_k", "stoch_d", "williams_r"],
        "Trend":           ["macd", "macd_signal", "ema_short", "ema_long", "adx"],
        "Volatility":      ["bb_upper", "bb_lower", "atr", "rolling_vol_20"],
        "Volume":          ["obv", "vwap", "cmf"],
        "Sentiment":       ["daily_sentiment", "finbert_pos_mean", "headline_count"],
        "Macro":           ["fed_funds_rate", "yield_spread", "vix", "unemployment"],
    }
    for group, cols in groups.items():
        present = [c for c in cols if c in df.columns]
        print(f"    {group:<16} {len(present)}/{len(cols)} features present")

    print("=" * 60 + "\n")       

# Orchestration

def run_full_pipline(
    tickers: list[str] = TICKERS,
    start: str = START_DATE,
    end: str = END_DATE,
) -> pd.DataFrame
    log.info(f"Starting full pipeline for {len(tickers)} tickers: {tickers}")
    log.info(f"Date range: {start} → {end}")

    # Run each source 
    features_df  = run_yfinance_pipeline()
    sentiment_df = run_news_pipeline()
    macro_df     = run_fred_pipeline()

    #  Merge 
    log.info("=== Phase 4: Merging into Unified Feature Store ===")
    merged = merge_feature_store(features_df, sentiment_df, macro_df)

    # ─ Validate 
    validate_dataframe(
        merged,
        required_cols=["ticker", "date", "close", "rsi", "target"],
        name="unified_feature_store",
    )

    #  Save 
    save_to_db(merged, TABLE_MERGED)
    log.info(f"Unified feature store saved to DB table '{TABLE_MERGED}'")

    # ── Report 
    print_pipeline_report(merged)

    return merged

    
#  Entrypoint 

if __name__ == "__main__":
    df = run_full_pipeline()
    print("Sample rows:")
    display_cols = [
        "ticker", "date", "close", "rsi", "macd",
        "daily_sentiment", "yield_spread", "vix", "target",
    ]
    display_cols = [c for c in display_cols if c in df.columns]
    print(df[display_cols].tail(12).to_string())
