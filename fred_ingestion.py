import pandas as pd 
import numpy as np 
from fredapi import FRED 
from tdqm import tdqm 

from config.settings import (
    START_DATE,END_DATE,
    FRED_API_KEY, FRED_SERIES,
    TABLE_MACRO,
)
from utils.helpers import(
    setup_logger, save_to_db, validate_DataFrame,
    api_retry,business_days
)
import warnings
warnings.filterwarning('ignore')

log = setup_logger("fred")

# Single series pull

@api_retry(max_attempts=3)
def fetch_fred_series(
    fred_client : Fred,
    series_id : str,
    name: str,
    start: str,
    end:str,
) -> pd.Series:

    log.debug(f"Fetching FRED Series : {series_id} ({name})")
    s = fred_client.get_series(
        series_id,
        observation_start=start,
        observation_end=end,
    )
    s.name=name
    s.index = pd.to_datetime(s.index)
    return s 

    # Alignment to daily business-day index
    def align_to_daily(
        seires_dict: dict[str, pd.Series],
        start: str,
        end: str,
    ) -> pd.DataFrame:
   
   
    bday_idx = business_days(start,end)

    aligned = {}

    for name,s in seires_dict.items():
        # Reindex to business days
        s_daily = s.reindex(bday_idx)
        
        # Forward fill : hold fast known value forward
        s_daily = s_daily.ffill()

        # Lag by 1 day
        s_daily = s_daily.shift(1)

        aligned[name] = s_daily

    # Derived Features
    if "yield_10y" in df.columns and "yield2y" in df.columns:
        df['yield_spread'] = df['yield_10y'] - df['yield_2y']

    if "cpi" in df.columns:
        # Month-over-month CPI inflation rate
        df["cpi_mom"] = df["cpi"].pct_change()

    if "sp500" in df.columns:
        # SP500 daily return (market return context)
        df["sp500_return"] = df["sp500"].pct_change()

    if "vix" in df.columns:
        # Fear/greed regime: high VIX = fear
        df["high_vix"] = (df["vix"] > 25).astype(int)   # 25 = historical threshold

    return df   

# Full Fred Pipeline

def build_macro_features(
    start:str = START_DATE,
    end:str = END_DATE,
    save: bool = True,
) -> pd.DataFrame 

    if not FRED_API_KEY:
        log.warning(
            "FRED_API_KEY not set. Returning empty macro DataFrame. "
            "Get a free key at https://fred.stlouisfed.org/docs/api/api_key.html"
        )
        return pd.DataFrame()
    fred = Fred(api_key=FRED_API_KEY)
    series_dict: dict[str, pd.Series] = {}

    log.info(f"Fetching {len(FRED_SERIES)} FRED series...")

    for name, series_id in tqdm(FRED_SERIES.items(), desc="FRED API"):
        try:
            s = fetch_fred_series(fred, series_id, name, start, end)
            series_dict[name] = s
            log.info(f"{name} ({series_id}): {len(s)} observations")
        except Exception as e:
            log.error(f"{name} ({series_id}): FAILED — {e}")

    if not series_dict:
        raise RuntimeError("Failed to fetch any FRED series") 
    # Align everything to daily business days
    macro_df = align_to_daily(series_dict, start, end)

    # Flatten index for DB storage
    macro_df = macro_df.reset_index()
    macro_df["date"] = macro_df["date"].dt.strftime("%Y-%m-%d")

    # Drop rows where ALL macro values are NaN
    # (first few rows after lag will have NaN — that's expected)
    macro_df = macro_df.dropna(how="all", subset=[c for c in macro_df.columns if c != "date"])

    validate_dataframe(
        macro_df,
        required_cols=["date", "yield_spread"],
        name="macro_features",
    )

    if save:
        save_to_db(macro_df, TABLE_MACRO)

    log.info(f"Macro features built: {len(macro_df):,} rows, {macro_df.shape[1]} columns")
    log.info(f"Columns: {macro_df.columns.tolist()}")
    return macro_df


#  Descriptive stats helper 

def describe_macro(df: pd.DataFrame) -> None:
    """Print a quick summary of the macro feature DataFrame."""
    print("\n── FRED Macro Feature Summary ──────────────────")
    print(f"Date range: {df['date'].min()} → {df['date'].max()}")
    print(f"Shape: {df.shape}")
    print(f"\nNull % per column:")
    null_pct = df.isnull().mean().round(3) * 100
    print(null_pct.to_string())
    print(f"\nKey stats:")
    cols = [c for c in df.columns if c != "date"]
    print(df[cols].describe().round(3).to_string())

#  Entrypoint 

if __name__ == "__main__":
    macro = build_macro_features()
    describe_macro(macro)

    # Quick sanity check: yield curve plot (would be negative before 2022 rate hikes)
    if "yield_spread" in macro.columns:
        spread_series = macro.set_index("date")["yield_spread"]
        print(f"\nYield curve spread (10y-2y):")
        print(f"  Min (most inverted): {spread_series.min():.2f}%")
        print(f"  Max (most steep):    {spread_series.max():.2f}%")
        print(f"  Months inverted:     {(spread_series < 0).sum()} trading days")
