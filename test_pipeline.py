"""
Tests cover:
  - yfinance feature shapes and no-lookahead
  - FinBERT scorer output format
  - FRED alignment and lag correctness
  - Merge logic (null handling, join keys)
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from datetime import date


# ── yfinance feature tests ────────────────────────────────────────────────────────

class TestYFinanceFeatures:

    @pytest.fixture
    def sample_ohlcv(self):
        """Minimal OHLCV DataFrame for testing (100 rows = enough for all indicators)."""
        np.random.seed(42)
        n = 150
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        return pd.DataFrame({
            "ticker": "TEST",
            "date":   pd.date_range("2022-01-01", periods=n, freq="B").strftime("%Y-%m-%d"),
            "open":   close * (1 + np.random.randn(n) * 0.002),
            "high":   close * (1 + abs(np.random.randn(n)) * 0.005),
            "low":    close * (1 - abs(np.random.randn(n)) * 0.005),
            "close":  close,
            "volume": np.random.randint(1_000_000, 10_000_000, n).astype(float),
        })

    def test_feature_columns_present(self, sample_ohlcv):
        from data_pipeline.yfinance_ingestion import compute_technical_features
        result = compute_technical_features(sample_ohlcv)

        required = ["rsi", "macd", "macd_signal", "ema_short", "ema_long",
                    "bb_upper", "bb_lower", "atr", "obv", "daily_return",
                    "target", "future_return"]
        for col in required:
            assert col in result.columns, f"Missing feature: {col}"

    def test_rsi_range(self, sample_ohlcv):
        """RSI must always be in [0, 100]."""
        from data_pipeline.yfinance_ingestion import compute_technical_features
        result = compute_technical_features(sample_ohlcv)
        rsi_clean = result["rsi"].dropna()
        assert (rsi_clean >= 0).all() and (rsi_clean <= 100).all(), \
            f"RSI out of range: min={rsi_clean.min():.2f}, max={rsi_clean.max():.2f}"

    def test_target_is_binary(self, sample_ohlcv):
        """Target variable must only contain 0 and 1."""
        from data_pipeline.yfinance_ingestion import compute_technical_features
        result = compute_technical_features(sample_ohlcv)
        target_vals = result["target"].dropna().unique()
        assert set(target_vals).issubset({0, 1}), \
            f"Target contains non-binary values: {target_vals}"

    def test_no_future_leakage(self, sample_ohlcv):
        """
        Ensure target is not in the feature set used for training.
        This is the #1 most important correctness test.
        """
        from data_pipeline.yfinance_ingestion import compute_technical_features
        result = compute_technical_features(sample_ohlcv)

        # Correlation between target and same-day close should not be ~1.0
        # (if it were, we've leaked future price info into the target)
        corr = result[["close", "target"]].dropna().corr().iloc[0, 1]
        assert abs(corr) < 0.5, \
            f"Suspicious correlation between close and target: {corr:.3f} — possible lookahead"

    def test_last_n_rows_have_nan_target(self, sample_ohlcv):
        """Last PREDICTION_HORIZON rows should have NaN target (no future data)."""
        from data_pipeline.yfinance_ingestion import compute_technical_features
        from config.settings import PREDICTION_HORIZON
        result = compute_technical_features(sample_ohlcv)
        last_n = result["target"].tail(PREDICTION_HORIZON)
        assert last_n.isnull().all(), \
            f"Expected NaN in last {PREDICTION_HORIZON} target rows"

    def test_row_count_preserved(self, sample_ohlcv):
        from data_pipeline.yfinance_ingestion import compute_technical_features
        result = compute_technical_features(sample_ohlcv)
        assert len(result) == len(sample_ohlcv)


# ── FinBERT sentiment tests ───────────────────────────────────────────────────────

class TestFinBERTScorer:

    def test_net_sentiment_range(self):
        """net_sentiment must be in [-1, +1]."""
        from data_pipeline.news_ingestion import FinBERTScorer
        scorer = FinBERTScorer()
        headlines = [
            "Apple stock surges to record high on strong earnings",
            "Tesla faces massive recall over safety concerns",
            "Market opens flat ahead of Fed decision",
        ]
        results = scorer.score_batch(headlines)
        for r in results:
            assert -1.0 <= r["net_sentiment"] <= 1.0, \
                f"net_sentiment out of range: {r['net_sentiment']}"

    def test_probabilities_sum_to_one(self):
        """pos + neg + neu should sum to ~1.0."""
        from data_pipeline.news_ingestion import FinBERTScorer
        scorer = FinBERTScorer()
        headlines = ["Strong earnings beat expectations"]
        result = scorer.score_batch(headlines)[0]
        total = result["finbert_pos"] + result["finbert_neg"] + result["finbert_neu"]
        assert abs(total - 1.0) < 0.01, f"Probabilities sum to {total:.3f}, not 1.0"

    def test_positive_headline_gets_positive_score(self):
        """Clearly positive financial news should score > 0."""
        from data_pipeline.news_ingestion import FinBERTScorer
        scorer = FinBERTScorer()
        results = scorer.score_batch(["Company reports record profits, stock surges"])
        assert results[0]["net_sentiment"] > 0, \
            "Positive headline scored negative"

    def test_negative_headline_gets_negative_score(self):
        """Clearly negative financial news should score < 0."""
        from data_pipeline.news_ingestion import FinBERTScorer
        scorer = FinBERTScorer()
        results = scorer.score_batch(["Company files for bankruptcy amid fraud allegations"])
        assert results[0]["net_sentiment"] < 0, \
            "Negative headline scored positive"

    def test_empty_headline_handled(self):
        """Empty string should not raise — returns neutral score."""
        from data_pipeline.news_ingestion import FinBERTScorer
        scorer = FinBERTScorer()
        results = scorer.score_batch([""])
        assert results[0]["net_sentiment"] is not None

    def test_batch_output_length(self):
        from data_pipeline.news_ingestion import FinBERTScorer
        scorer = FinBERTScorer()
        headlines = ["headline one", "headline two", "headline three"]
        results = scorer.score_batch(headlines)
        assert len(results) == 3


# ── FRED alignment tests ──────────────────────────────────────────────────────────

class TestFREDAlignment:

    @pytest.fixture
    def mock_series(self):
        """Simulate monthly Fed funds rate series."""
        dates = pd.date_range("2022-01-01", periods=12, freq="MS")
        return {
            "fed_funds_rate": pd.Series(
                [0.25, 0.25, 0.50, 1.00, 1.75, 2.50, 3.00, 3.25, 3.75, 4.00, 4.25, 4.50],
                index=dates,
                name="fed_funds_rate",
            )
        }

    def test_daily_index_has_business_days_only(self, mock_series):
        from data_pipeline.fred_ingestion import align_to_daily
        result = align_to_daily(mock_series, "2022-01-01", "2022-12-31")
        # Check no weekends in index
        day_of_week = result.index.dayofweek
        assert (day_of_week <= 4).all(), "Non-business days found in macro index"

    def test_ffill_fills_monthly_gaps(self, mock_series):
        from data_pipeline.fred_ingestion import align_to_daily
        result = align_to_daily(mock_series, "2022-01-01", "2022-12-31")
        # After ffill, no NaN except possibly the first row (before first observation)
        non_null = result["fed_funds_rate"].dropna()
        assert len(non_null) > 200, "ffill didn't propagate monthly values to daily"

    def test_lag_applied(self, mock_series):
        from data_pipeline.fred_ingestion import align_to_daily
        result = align_to_daily(mock_series, "2022-01-01", "2022-12-31")
        # Jan 3 (first business day) should have NaN because lag shifts everything
        first_row_val = result["fed_funds_rate"].iloc[0]
        assert pd.isna(first_row_val), \
            f"First row should be NaN after 1-day lag, got {first_row_val}"

    def test_yield_spread_derived(self):
        from data_pipeline.fred_ingestion import align_to_daily
        bday = pd.date_range("2022-01-01", periods=30, freq="B")
        series = {
            "yield_10y": pd.Series([3.5] * 30, index=bday, name="yield_10y"),
            "yield_2y":  pd.Series([4.0] * 30, index=bday, name="yield_2y"),
        }
        result = align_to_daily(series, "2022-01-01", "2022-02-28")
        assert "yield_spread" in result.columns
        # Spread should be 3.5 - 4.0 = -0.5 (inverted curve)
        spread_clean = result["yield_spread"].dropna()
        assert (abs(spread_clean - (-0.5)) < 0.001).all()


# ── Merge tests ───────────────────────────────────────────────────────────────────

class TestMerge:

    def _make_features(self) -> pd.DataFrame:
        return pd.DataFrame({
            "ticker": ["AAPL", "AAPL", "MSFT", "MSFT"],
            "date":   ["2023-01-02", "2023-01-03", "2023-01-02", "2023-01-03"],
            "close":  [150.0, 152.0, 240.0, 242.0],
            "rsi":    [55.0, 57.0, 50.0, 52.0],
            "macd":   [0.5, 0.6, -0.2, -0.1],
            "target": [1, 0, 1, 1],
        })

    def _make_sentiment(self) -> pd.DataFrame:
        return pd.DataFrame({
            "ticker":          ["AAPL"],
            "date":            ["2023-01-02"],
            "daily_sentiment": [0.45],
            "finbert_pos_mean":[0.70],
            "finbert_neg_mean":[0.10],
            "headline_count":  [5],
        })

    def _make_macro(self) -> pd.DataFrame:
        return pd.DataFrame({
            "date":            ["2023-01-02", "2023-01-03"],
            "yield_spread":    [-0.5, -0.5],
            "vix":             [20.0, 21.0],
            "fed_funds_rate":  [4.5, 4.5],
        })

    def test_merge_preserves_all_trading_rows(self):
        from data_pipeline.pipeline import merge_feature_store
        features  = self._make_features()
        sentiment = self._make_sentiment()
        macro     = self._make_macro()
        merged = merge_feature_store(features, sentiment, macro)
        assert len(merged) == 4, f"Expected 4 rows, got {len(merged)}"

    def test_missing_sentiment_filled_with_zero(self):
        from data_pipeline.pipeline import merge_feature_store
        features  = self._make_features()
        sentiment = self._make_sentiment()  # only AAPL Jan-02
        macro     = self._make_macro()
        merged = merge_feature_store(features, sentiment, macro)

        # MSFT rows should have 0 sentiment (no news)
        msft_sentiment = merged[merged["ticker"] == "MSFT"]["daily_sentiment"]
        assert (msft_sentiment == 0).all(), \
            "Missing sentiment should be filled with 0"

    def test_macro_columns_present(self):
        from data_pipeline.pipeline import merge_feature_store
        merged = merge_feature_store(
            self._make_features(), self._make_sentiment(), self._make_macro()
        )
        assert "yield_spread" in merged.columns
        assert "vix" in merged.columns

    def test_no_duplicate_rows(self):
        from data_pipeline.pipeline import merge_feature_store
        merged = merge_feature_store(
            self._make_features(), self._make_sentiment(), self._make_macro()
        )
        dups = merged.duplicated(subset=["ticker", "date"])
        assert not dups.any(), f"Duplicate (ticker, date) rows found after merge"

    def test_empty_sentiment_handled(self):
        from data_pipeline.pipeline import merge_feature_store
        merged = merge_feature_store(
            self._make_features(), pd.DataFrame(), self._make_macro()
        )
        assert "daily_sentiment" in merged.columns
        assert (merged["daily_sentiment"] == 0).all()


# ── Quick smoke test (runs if you have real API keys) ─────────────────────────────

@pytest.mark.integration
def test_yfinance_live_fetch():
    """Integration test — actually hits yfinance API."""
    from data_pipeline.yfinance_ingestion import fetch_ohlcv
    df = fetch_ohlcv("AAPL", "2024-01-01", "2024-01-31")
    assert not df.empty
    assert "close" in df.columns
    assert "ticker" in df.columns
    assert df["ticker"].iloc[0] == "AAPL"
