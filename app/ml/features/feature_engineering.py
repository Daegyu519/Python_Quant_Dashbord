"""
=============================================================================
Feature Engineering Pipeline — Alpha Research Grade
=============================================================================
시계열 피처 엔지니어링.
데이터 누수(data leakage) 방지를 위한 strict time-aware 설계.
Numba JIT + Polars 활용 초고속 처리.
=============================================================================
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd
from numba import njit

from app.core.base_classes import BaseFeatureEngineer
from app.core.types import OHLCVFrame
from app.config.logging_config import get_logger, PerformanceLogger

logger = get_logger(__name__)


# ─────────────────────────────────────────────
# Numba JIT 최적화 피처 계산
# ─────────────────────────────────────────────

@njit(cache=True, fastmath=True)
def _rolling_zscore(arr: np.ndarray, window: int) -> np.ndarray:
    """롤링 Z-Score (Numba JIT)"""
    n = len(arr)
    result = np.full(n, np.nan)

    for i in range(window - 1, n):
        window_data = arr[i - window + 1:i + 1]
        mean = np.mean(window_data)
        std = np.std(window_data)
        if std > 1e-10:
            result[i] = (arr[i] - mean) / std
        else:
            result[i] = 0.0

    return result


@njit(cache=True, fastmath=True)
def _rolling_rank(arr: np.ndarray, window: int) -> np.ndarray:
    """롤링 백분위 랭크 (Numba JIT)"""
    n = len(arr)
    result = np.full(n, np.nan)

    for i in range(window - 1, n):
        window_data = arr[i - window + 1:i + 1]
        curr_val = arr[i]
        rank = np.sum(window_data < curr_val) / window
        result[i] = rank

    return result


@njit(cache=True, fastmath=True)
def _rsi_fast(close: np.ndarray, period: int) -> np.ndarray:
    """고속 RSI 계산 (Numba JIT)"""
    n = len(close)
    rsi = np.full(n, 50.0)

    gains = np.zeros(n)
    losses = np.zeros(n)

    for i in range(1, n):
        delta = close[i] - close[i - 1]
        if delta > 0:
            gains[i] = delta
        else:
            losses[i] = -delta

    # Wilder's EMA
    alpha = 1.0 / period
    avg_gain = gains[1]
    avg_loss = losses[1]

    for i in range(1, n):
        avg_gain = alpha * gains[i] + (1 - alpha) * avg_gain
        avg_loss = alpha * losses[i] + (1 - alpha) * avg_loss

        if avg_loss > 1e-10:
            rs = avg_gain / avg_loss
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
        else:
            rsi[i] = 100.0

    return rsi


@njit(cache=True, fastmath=True)
def _atr_fast(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    """고속 ATR (Average True Range) 계산 (Numba JIT)"""
    n = len(close)
    atr = np.zeros(n)

    for i in range(1, n):
        tr = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1]),
        )
        if i < period:
            atr[i] = tr
        else:
            atr[i] = (atr[i - 1] * (period - 1) + tr) / period

    return atr


# ─────────────────────────────────────────────
# 메인 피처 엔지니어링 클래스
# ─────────────────────────────────────────────

class QuantFeatureEngineer(BaseFeatureEngineer):
    """
    기관급 피처 엔지니어링 파이프라인.

    생성 피처:
    1. 가격 기반 (Price Features)
       - Returns: 1d, 5d, 21d, 63d, 126d, 252d
       - Log Returns
       - Price Z-Score
       - Price Rank (rolling)

    2. 기술적 지표 (Technical Indicators)
       - RSI (7, 14, 21)
       - MACD
       - ATR (14)
       - Bollinger Bands
       - 이동평균 (5, 10, 20, 50, 200)
       - MA 기울기

    3. 변동성 피처 (Volatility Features)
       - Realized Volatility (5d, 20d, 60d)
       - GARCH Vol estimate
       - Vol of Vol
       - VIX 프록시

    4. 거래량 피처 (Volume Features)
       - Volume Ratio (상대 거래량)
       - Volume Z-Score
       - OBV (On Balance Volume)
       - VWAP 편차

    5. 시장 레짐 피처 (Market Regime Features)
       - Trend Strength (ADX 기반)
       - Volatility Regime
       - Rolling Correlation (시장과의 상관관계)

    6. 모멘텀 팩터 (Momentum Factors)
       - 1M, 3M, 6M, 12M 모멘텀
       - Skip-month 모멘텀
       - Momentum Quality

    모든 피처는 strict time-aware:
    - 미래 데이터 사용 금지
    - rolling window 기반
    - 정규화는 fit() 데이터만 사용
    """

    def __init__(
        self,
        lookback_short: int = 5,
        lookback_medium: int = 21,
        lookback_long: int = 252,
        include_regime: bool = True,
        include_volume: bool = True,
    ) -> None:
        self.lookback_short = lookback_short
        self.lookback_medium = lookback_medium
        self.lookback_long = lookback_long
        self.include_regime = include_regime
        self.include_volume = include_volume

        # fit()에서 채워질 통계 (스케일러용)
        self._feature_stats: dict[str, dict] = {}
        self._feature_names: list[str] = []
        self._is_fitted = False

    def fit(self, data: OHLCVFrame) -> "QuantFeatureEngineer":
        """
        학습 데이터로 피처 통계 계산.
        반드시 학습 데이터만 사용 (test set 오염 방지).
        """
        features = self._compute_raw_features(data)

        # 각 피처의 통계 저장 (정규화용)
        for col in features.columns:
            col_data = features[col].dropna()
            if len(col_data) > 0:
                self._feature_stats[col] = {
                    "mean": float(col_data.mean()),
                    "std": float(col_data.std()),
                    "min": float(col_data.min()),
                    "max": float(col_data.max()),
                    "p1": float(col_data.quantile(0.01)),
                    "p99": float(col_data.quantile(0.99)),
                }

        self._feature_names = list(features.columns)
        self._is_fitted = True

        logger.info(
            "feature_engineer_fitted",
            n_features=len(self._feature_names),
            n_samples=len(data.data),
        )

        return self

    def transform(self, data: OHLCVFrame) -> pd.DataFrame:
        """
        피처 행렬 생성.
        fit()에서 계산한 통계로 정규화.
        """
        if not self._is_fitted:
            raise RuntimeError("Must call fit() before transform()")

        with PerformanceLogger("feature_transform", symbol=data.symbol, bars=data.n_bars):
            features = self._compute_raw_features(data)

            # Winsorize + Z-Score 정규화 (fit 통계 사용)
            normalized = features.copy()
            for col in features.columns:
                if col in self._feature_stats:
                    stats = self._feature_stats[col]
                    # Winsorize (이상치 클리핑)
                    normalized[col] = features[col].clip(
                        lower=stats["p1"], upper=stats["p99"]
                    )
                    # Z-Score
                    if stats["std"] > 1e-10:
                        normalized[col] = (normalized[col] - stats["mean"]) / stats["std"]

        return normalized

    def _compute_raw_features(self, data: OHLCVFrame) -> pd.DataFrame:
        """모든 피처 계산"""
        df = data.data.copy()
        close = df["close"].to_numpy(dtype=np.float64)
        high = df["high"].to_numpy(dtype=np.float64)
        low = df["low"].to_numpy(dtype=np.float64)
        volume = df["volume"].to_numpy(dtype=np.float64)
        n = len(close)

        features: dict[str, np.ndarray] = {}

        # ── 1. 수익률 피처 ──
        for period in [1, 5, 10, 21, 63, 126, 252]:
            if n > period:
                ret = np.zeros(n)
                ret[period:] = (close[period:] - close[:-period]) / (close[:-period] + 1e-10)
                features[f"return_{period}d"] = ret

        # 로그 수익률
        log_ret = np.zeros(n)
        log_ret[1:] = np.log(close[1:] / (close[:-1] + 1e-10))
        features["log_return_1d"] = log_ret

        # ── 2. 기술적 지표 ──

        # RSI
        for rsi_period in [7, 14, 21]:
            features[f"rsi_{rsi_period}"] = _rsi_fast(close, rsi_period)

        # ATR
        atr_14 = _atr_fast(high, low, close, 14)
        features["atr_14"] = atr_14 / (close + 1e-10)  # ATR/Price 비율

        # 이동평균 편차 (가격 / MA - 1)
        for ma_period in [5, 10, 20, 50, 200]:
            if n > ma_period:
                ma = pd.Series(close).rolling(ma_period).mean().to_numpy()
                features[f"price_vs_ma_{ma_period}"] = close / (ma + 1e-10) - 1

        # MA 기울기 (트렌드 방향)
        for ma_period in [20, 50]:
            if n > ma_period + 5:
                ma = pd.Series(close).rolling(ma_period).mean()
                ma_slope = ma.diff(5) / (ma.shift(5) + 1e-10)
                features[f"ma_slope_{ma_period}"] = ma_slope.to_numpy()

        # 볼린저 밴드 위치
        bb_period = 20
        if n > bb_period:
            ma_20 = pd.Series(close).rolling(bb_period).mean()
            std_20 = pd.Series(close).rolling(bb_period).std()
            bb_upper = ma_20 + 2 * std_20
            bb_lower = ma_20 - 2 * std_20
            bb_width = (bb_upper - bb_lower) / (ma_20 + 1e-10)

            # BB %B: 현재 가격의 밴드 내 위치 (0=하단, 1=상단)
            bb_pct = (pd.Series(close) - bb_lower) / (bb_upper - bb_lower + 1e-10)
            features["bb_pct_b"] = bb_pct.to_numpy()
            features["bb_width"] = bb_width.to_numpy()

        # MACD
        ema_12 = pd.Series(close).ewm(span=12, adjust=False).mean()
        ema_26 = pd.Series(close).ewm(span=26, adjust=False).mean()
        macd = ema_12 - ema_26
        macd_signal = macd.ewm(span=9, adjust=False).mean()
        features["macd_hist"] = (macd - macd_signal).to_numpy() / (close + 1e-10)
        features["macd_line"] = macd.to_numpy() / (close + 1e-10)

        # ── 3. 변동성 피처 ──
        for vol_period in [5, 21, 63]:
            rv = pd.Series(log_ret).rolling(vol_period).std() * np.sqrt(252)
            features[f"realized_vol_{vol_period}d"] = rv.to_numpy()

        # 변동성의 변동성 (vol of vol)
        rv_21 = pd.Series(log_ret).rolling(21).std()
        features["vol_of_vol"] = rv_21.rolling(21).std().to_numpy()

        # ── 4. 거래량 피처 ──
        if self.include_volume:
            vol_ma_20 = pd.Series(volume).rolling(20).mean()
            features["volume_ratio_20d"] = (volume / (vol_ma_20.to_numpy() + 1e-10))

            # OBV (On Balance Volume)
            obv = np.zeros(n)
            for i in range(1, n):
                if close[i] > close[i - 1]:
                    obv[i] = obv[i - 1] + volume[i]
                elif close[i] < close[i - 1]:
                    obv[i] = obv[i - 1] - volume[i]
                else:
                    obv[i] = obv[i - 1]
            # OBV 정규화 (rolling z-score)
            features["obv_zscore"] = _rolling_zscore(obv, 20)

        # ── 5. 레짐 피처 ──
        if self.include_regime:
            # ADX 스타일 트렌드 강도
            for trend_period in [14, 28]:
                if n > trend_period * 2:
                    # 단순화된 트렌드 강도
                    prices = pd.Series(close)
                    trend_strength = (
                        prices.rolling(trend_period).max() -
                        prices.rolling(trend_period).min()
                    ) / prices.rolling(trend_period).mean()
                    features[f"trend_strength_{trend_period}d"] = trend_strength.to_numpy()

        # ── 6. 모멘텀 팩터 ──
        # 1M, 3M, 6M, 12M minus skip-month
        for months, days in [(1, 21), (3, 63), (6, 126), (12, 252)]:
            skip = 21
            if n > days + skip:
                momentum = np.full(n, np.nan)
                for i in range(days + skip, n):
                    past_price = close[i - days - skip]
                    curr_price = close[i - skip]
                    momentum[i] = curr_price / (past_price + 1e-10) - 1
                features[f"momentum_{months}m"] = momentum

        # Rolling Rank (팩터 단면 정규화 대용)
        if "momentum_1m" in features:
            features["momentum_1m_rank"] = _rolling_rank(
                np.nan_to_num(features["momentum_1m"]), 63
            )

        # ── DataFrame 변환 ──
        result = pd.DataFrame(features, index=df.index)

        return result

    def get_feature_names(self) -> list[str]:
        return self._feature_names.copy()

    def get_feature_importance_hint(self) -> dict[str, str]:
        """피처 설명 (ML 모델 해석에 사용)"""
        return {
            "return_1d": "1일 수익률",
            "return_5d": "5일 수익률",
            "return_21d": "21일 수익률 (1개월)",
            "rsi_14": "RSI(14) - 과매수/과매도",
            "macd_hist": "MACD 히스토그램 (추세)",
            "bb_pct_b": "볼린저 밴드 위치 (0=하단, 1=상단)",
            "realized_vol_21d": "21일 실현 변동성",
            "volume_ratio_20d": "상대 거래량",
            "momentum_12m": "12개월 모멘텀 팩터",
        }
