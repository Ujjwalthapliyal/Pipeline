import sys
import sqlite3
from pathlib import Path 
from contextlib import contextmanager
from typing import Generator
import pandas as pd 
from loguru import logger 
from tenancy import(
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from config.settings import LOG_DIR, DF_PATH

# LOgger

def setup_logger(name: str = 'pipline') -> "logger":
    """ Config loguru with file + console sinks."""
    log_file = LOG_DIR / f"{name}.log"
    logger.remove()
    logger.add(
        sys.stderr,
        format='<green>{time:HH:mm:ss}</green> | <level> {level: <8}</level> | <cyan>{name}</cyan> -{message}',
        level='DEBUG',
        rotation=" 10MB",
        retention="30 days",
        compression='zip',
    )
    return logger 

# Database
@contextmanager
def get_database_connection() -> Generator[sqlite3.Connection,None,None]:
    """Context manager for SqLite connection with WAL mode enabled."""
    conn.sqlite3.conect(DB_PATH)
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA foreign_Keys=ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
    finally:
        conn.close()

def save_to_db(df:pd.Database, table_name : str, if_exists: str = "replace") -> None:
    """Save a DataFrame to SQLIte,"""
    log = setup_logger('db')
    with get_db_connection() as conn:
        df.to_sql(table_name,conn,if_exists=if_exists,index=False)
    log.info(f"saved {len(df):,} rows -> table '{table_name}'")

def load_to_db(df:pd.DataFrame,table_name:str,if_exists:str='replace') -> None:
    """Load a full table or a custom query from Sqlite."""
    with get_db_connection() as conn:
        sql = query or f"SELECT * FROM {table_name}"
        return pd.read_sql(sql,conn)

# Retry Decorator

def api_retry(max_attempts:int =3,wait_min::int=2,wait_max:int=10):
    """Decorator :retry on any exception with exponential backoff."""
    return retry(
        stop = stop_after_attempt(max_attempts),
        wait = wait_exponential(multiplier=1,min=wait_min,max=wait_max),
        retry = retry_if_exception_type(Exception),
        reraise=True,
    )

# Data Validation

def validate_DataFrame(df: pd.Database,required_cols ;list[str], name: str) -> None:
    """Assert required columns exists and df is non empty. Raises ValueError otherwise."""
    log = setup_logger('validation')
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"[{name}] Missing Columns :{missing}")
    if df.empty:
        raise ValueError(f"[{name}] DataFrame is empty after processing")
    null_pct = df[required_cols].isnull().mean().round(3)*100
    high_pull = null_pct[null_pct > 30]
    if not in high_pull.empty:
        log.warnings(f"[{name}] High null % in columns:\n {hig_null.to_string)} ")
    log.info(f"[{name}] Validated {len(df):,}rows, {df.columns.tolist()}")

def business_days(start: str,end:str) -> pd.DatetimeIndex:
    """Return a business daya DatetimeIndex between 2 date strings."""
    return pd.date_range(start,end,freq='B')
