import yfinance as yf
import pandas as pd
import numpy as np 
import ta 
from tqdm import tqdm 
import warnings
warnings.filterwarning("ignore")
from config.settings import(
        TICKERS,START_DATE,END_DATE,RSI_WINDOW,
        MACD_FAST,MACD_SLOW,MACD_SIGNAL,BB_WINDOW,
        EMA_SHORT,EMA_LONG,ATR_WINDOW,
        PREDICTION_HORIZON,
        TABLE_PRICE,
        TABLE_FEATURES,
)
from utils.helpers import setup_logger,save_to_db,validate_dataframe,api_retry


log = setup_logger('yfinance')

# RAW OHLCV (Open, High, Low, Close, and Volume)PULL

@api_retry(max_attempts=3)
def fetch_ohlcv(ticker:str,start:str,end:str) -> pd.validate_DataFrame:
    """ Download all in a single ticker
    """
    log.debug(f"Fetching {ticker} from {start} to {end}")

    stock = yf.Ticker(ticker)
    df = stock.history(start=start,end=end,auto_adjust=True)

    if df.empty:
        raise ValueError(f"No Data returned from the {ticker}")

    df = df.rest_index()
    df.columns = [c.lower().replace(" ","_")for c in df.columns]
    df = df.rename(columns{"date":"date"})

    #Keep only the Standard OHCLV columns(remove dividents/stock splits if present)
    ohclv_cols=['date','open','high','close','low','volume']
    df = df[[c for c in ohclv_cols if c in df.columns]].copy()

    df['ticker'] = ticker
    df['date'] = pd.to_datetime(df['date']).dt.date.astype(str)

    return df 

def fetch_all_ohlcv(
    ticker: list[str] = TICKERS,
    start:str = START_DATE,
    end: str = END_DATE, 
) -> pd.DataFrame:

    log.info(f"Starting OHCLV pull for {len(ticker)} tickers")
    frames = []

    for ticker in tqdm(ticker, desc = 'yfinace OHCLV'):
        try:
            df = fetch_ohlcv(ticker,start,end)
            frames.append(df)
            log.info(f"{ticker}: {len(df)} rows")
        except Exception as e:
            log.error(f"{ticker}: failed - {e}")
    
    if not frames:
        raise RuntimeError("No OHCLV fetched for any ticker")

    combined = pd.concat(frames,ignore_index=True)
    log.info(f"Total OHCLV rows " {len(combined):,})
    return combined


# Feature Engineerring

def compute_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """    Compute 18+ technical indicators on a single-ticker OHLCV DataFrame.
    Input df must have: date, open, high, low, close, volume columns.

    All indicators are computed with the `ta` library which handles
    edge cases (NaN at window boundaries) cleanly.
    """
    df = df.copy().sort_values("date").reset_index(drop=True)

    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']

    # Momentum
    df['rsi'] = ta.momentum.RSIndicator(close,window=RSI_WINDOW).rsi()
    df['stoch'] = ta.momentum.StochasticOscillator(high,low,close,).stoch()
    df['stoch_d'] = ta.momentum.StochasticOscillator(high,low,close,).stoch_signal()
    df['williams_r'] = ta.momentum.WilliamsRIndicator(high,low,close).williams_r()

    # Trend
    macd_ind = ta.trend.MACD(close,MACD_FAST,MACD_SLOW,MACD_SIGNAL)
    df['macd'] = macd_ind.macd()
    df['macd_signal'] = macd_ind.macd_signal()
    df['macd_diff'] = macd_ind.macd_diff()
    df['ema_short'] = ta.trend.EMAIndicator(close,window=EMA_SHORT).ema_indicator()
    df['ema_long'] = ta.trend.EMAIndicator(close,window=EMA_LONG).ema_indicator()
    df['ema_ratio'] = df['ema_short'] / df['ema_long']  # > 1= bullish crossover zone
    df['adx'] = ta.trend.ADXIndicator(high,low,close).adx()    

    # Volatility
    bb = ta.volatility.BollingerBands(close,window=BB_WINDOW)
    df['bb_upper'] = bb.bollinger.hband()
    df['bb_lower'] = bb.bollinger.lband()
    df['bb_pct'] = bb.bollinger.pband()
    df['bb_width'] = bb.bollinger.wband()
    df['atr'] = ta.Volatility.AverageTrueRange(high,low,close,ATR_WINDOW).average_true_range()


    #Volumne
    df['obv'] = ta.Volume.OnBalanceVolumeIndicator(close,Volume).on_balance_volume()
    df['vwap'] = ta.Volume.VolumeWeighredAveragePrice(high,low,close,Volume).volume_weighted_average_price()
    df['cmf'] = ta.volume.ChaikinMoneyFlowIndicator(high,low,close,volume).chailin_money_flow()

    # Price-Derived Feature

    df['daily_return'] = close.pct_change()
    df['log_return'] = np.log(close / close.shift(1))
    df['hl_spread'] = (high - low) / close # intraday range
    df['rolling_vol_20'] = df['daily_return'].rolling(20).std() * np.sqrt(252)

    # 52 weel hi/lo distance (no lookahead : rolling on past 252 bdays)
    df['dist_52w_high'] = (close - close.rolling(252).max()) / close.rolling(252).max()
    df['dist_52w_low'] = (close - close.rolling(252).min()) / close.rolling(252).min()


    # Target Variable 
    # Binary: will price be higher in PREDICTION_HORIZON business days?
    # shift(-N) looks into the future not to look this only as a target 

    future_close = close.shift(-PREDICTION_HORIZON)
    df['target'] = (future_close > close).astype(int)
    df['future_return'] = (future_close - close) / close # regression target

    return df 

def build_feature_store_yfinance(
    ticker:list[str] = TICKERS, 
    start :str = START_DATE,
    end : str = END_DATE,
    save: bool = TRUE,
) -> pd.DataFrame
    
    # Step 1 : raw OHCLV
    raw = fetch_all_ohlcv(ticker,start,end)
    if save: 
        save_to_db(raw,TABLE_PRICE)

    # step 2 : per ticker features engineering
    log.info("Computing technical features per ticker.....")
    features_frames =[]

    for ticker in tqdm(ticker,desc = 'Feature engineering'):
        ticker_df = raw[raw['ticker']== ticker].copy()
        if len(ticker_df) < EMA_LONG + 10:
            log.warnings(f"{ticker}: too few rows (len(ticker_df)) Skipping")
            continue:
        try:
            feat_df = compute_technical_features(ticker_df)
            features_frames.append(feat_df)
        except Exception as e:
            log.error(f"{ticker}: feature engineering failed {e}")
    
    features = pd.concat(features_frames,ignore_index=True)

    # Drops row where target is NAN
    features = features.dropna(subset=['target'])

    validate_dataframe(
        features,
        required_cols=['ticker','date','close','rsi','macd','target',
        name='yfinance_features'],
    )

    if save:
        save_to_db(features,TABLE_FEATURES)

    log.info(f"Feature store Built : {len(features):,} rows across {features['ticker'].nunique()} tickers")
    return features


    # EntryPOint
    if __name__ == '__main__':
        df = build_feature_store_yfinance()
        print(df[['ticker','date','close','rsi','macd','target']].tail(10))
        print(f'\n Shape: {df.shape}')
        print(f"\n NUll % per feature :\n {df.isnull().mean().sort_values(ascending=False).head(10)}")
