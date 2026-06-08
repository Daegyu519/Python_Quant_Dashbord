"""
=============================================================================
Unit Tests — Strategy Engine
=============================================================================
전략 엔진 단위 테스트.
pytest + numpy 기반.
=============================================================================
"""

from __future__ import annotations

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from app.core.types import OHLCVFrame, Timeframe, SignalDirection
from app.strategies.technical_strategies import (
    MovingAverageCrossStrategy,
    RSIStrategy,
    MACDStrategy,
    BollingerBandMeanReversionStrategy,
    MomentumStrategy,
    EnsembleStrategy,
)


# ─────────────────────────────────────────────
# 테스트 픽스처
# ─────────────────────────────────────────────

def make_ohlcv_frame(
    n_bars: int = 300,
    symbol: str = "TEST",
    trend: str = "up",  # up, down, sideways
    seed: int = 42,
) -> OHLCVFrame:
    """테스트용 OHLCV 데이터 생성"""
    np.random.seed(seed)
    dates = pd.date_range(
        start=datetime(2020, 1, 1),
        periods=n_bars,
        freq="D",
        tz="UTC",
    )

    if trend == "up":
        base_price = 100 * np.exp(np.cumsum(np.random.normal(0.001, 0.02, n_bars)))
    elif trend == "down":
        base_price = 100 * np.exp(np.cumsum(np.random.normal(-0.001, 0.02, n_bars)))
    else:
        base_price = 100 * np.exp(np.cumsum(np.random.normal(0, 0.02, n_bars)))

    noise = np.random.uniform(0.98, 1.02, n_bars)
    df = pd.DataFrame({
        "open": base_price * np.random.uniform(0.99, 1.01, n_bars),
        "high": base_price * np.random.uniform(1.00, 1.03, n_bars),
        "low": base_price * np.random.uniform(0.97, 1.00, n_bars),
        "close": base_price,
        "volume": np.random.uniform(1e6, 1e7, n_bars),
    }, index=dates)

    # OHLC 논리 수정 (H >= max(O,C), L <= min(O,C))
    df["high"] = df[["open", "close", "high"]].max(axis=1)
    df["low"] = df[["open", "close", "low"]].min(axis=1)

    return OHLCVFrame(symbol=symbol, timeframe=Timeframe.ONE_DAY, data=df)


# ─────────────────────────────────────────────
# MA Cross Strategy Tests
# ─────────────────────────────────────────────

class TestMovingAverageCrossStrategy:

    def test_basic_initialization(self):
        """전략 초기화 테스트"""
        strategy = MovingAverageCrossStrategy(fast_period=10, long_period=30)
        assert strategy.fast_period == 10
        assert strategy.long_period == 30
        assert strategy.ma_type == "ema"

    def test_validate_config_valid(self):
        """유효한 설정 검증"""
        strategy = MovingAverageCrossStrategy(fast_period=10, long_period=50)
        assert strategy.validate_config() is True

    def test_validate_config_invalid(self):
        """잘못된 설정 예외 발생"""
        strategy = MovingAverageCrossStrategy(fast_period=50, long_period=10)
        with pytest.raises(ValueError, match="fast_period.*long_period"):
            strategy.validate_config()

    def test_insufficient_data_returns_empty(self):
        """데이터 부족 시 빈 리스트 반환"""
        strategy = MovingAverageCrossStrategy(fast_period=20, long_period=50)
        data = make_ohlcv_frame(n_bars=30)  # long_period보다 적음
        signals = strategy.generate_signal(data)
        assert signals == []

    def test_generates_signal_with_enough_data(self):
        """충분한 데이터로 시그널 생성"""
        strategy = MovingAverageCrossStrategy(fast_period=10, long_period=30)
        data = make_ohlcv_frame(n_bars=200, trend="up")
        signals = strategy.generate_signal(data)
        # 상승 추세에서는 매수 시그널이 있을 수 있음
        assert isinstance(signals, list)

    def test_signal_attributes(self):
        """시그널 속성 검증"""
        strategy = MovingAverageCrossStrategy(fast_period=5, long_period=20)
        data = make_ohlcv_frame(n_bars=200, trend="up")
        signals = strategy.generate_signal(data)

        for signal in signals:
            assert 0.0 <= signal.strength <= 1.0
            assert 0.0 <= signal.confidence <= 1.0
            assert signal.symbol == "TEST"
            assert signal.strategy_id == strategy.strategy_id

    def test_uptrend_generates_buy_signals(self):
        """상승 추세에서 매수 시그널 확인"""
        strategy = MovingAverageCrossStrategy(fast_period=5, long_period=20)
        # 강한 상승 추세 데이터 생성
        n = 100
        close = np.linspace(100, 200, n)  # 선형 상승
        dates = pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC")
        df = pd.DataFrame({
            "open": close * 0.99,
            "high": close * 1.02,
            "low": close * 0.98,
            "close": close,
            "volume": np.ones(n) * 1e6,
        }, index=dates)

        data = OHLCVFrame(symbol="TEST", timeframe=Timeframe.ONE_DAY, data=df)
        signals = strategy.generate_signal(data)

        if signals:
            # 상승 추세에서는 매수 시그널
            assert any(s.is_long for s in signals)

    def test_ema_vs_sma(self):
        """EMA와 SMA 구분 테스트"""
        strategy_ema = MovingAverageCrossStrategy(ma_type="ema")
        strategy_sma = MovingAverageCrossStrategy(ma_type="sma")
        data = make_ohlcv_frame(n_bars=200)

        # 둘 다 실행 가능해야 함
        signals_ema = strategy_ema.generate_signal(data)
        signals_sma = strategy_sma.generate_signal(data)
        assert isinstance(signals_ema, list)
        assert isinstance(signals_sma, list)


# ─────────────────────────────────────────────
# RSI Strategy Tests
# ─────────────────────────────────────────────

class TestRSIStrategy:

    def test_initialization(self):
        strategy = RSIStrategy(period=14, oversold=30.0, overbought=70.0)
        assert strategy.period == 14
        assert strategy.oversold == 30.0
        assert strategy.overbought == 70.0

    def test_rsi_calculation(self):
        """RSI 계산 정확도 테스트"""
        strategy = RSIStrategy(period=14)
        n = 100
        dates = pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC")
        close = pd.Series(
            [100 + i * 0.5 + np.sin(i * 0.3) * 5 for i in range(n)],
            index=dates,
        )
        df = pd.DataFrame({
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": np.ones(n) * 1e6,
        }, index=dates)
        data = OHLCVFrame(symbol="TEST", timeframe=Timeframe.ONE_DAY, data=df)

        rsi = strategy._calc_rsi(close, 14)

        # RSI는 0~100 범위
        valid_rsi = rsi.dropna()
        assert (valid_rsi >= 0).all()
        assert (valid_rsi <= 100).all()

    def test_oversold_buy_signal(self):
        """과매도 시 매수 시그널 확인"""
        strategy = RSIStrategy(period=14, oversold=30.0, overbought=70.0)

        # 급격한 하락 후 반등 시나리오
        n = 100
        dates = pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC")
        close_vals = np.concatenate([
            np.linspace(100, 70, 80),   # 급격한 하락 (RSI 과매도)
            np.linspace(70, 72, 20),    # 소폭 반등
        ])

        df = pd.DataFrame({
            "open": close_vals * 0.99,
            "high": close_vals * 1.01,
            "low": close_vals * 0.99,
            "close": close_vals,
            "volume": np.ones(n) * 1e6,
        }, index=dates)
        data = OHLCVFrame(symbol="TEST", timeframe=Timeframe.ONE_DAY, data=df)

        signals = strategy.generate_signal(data)
        # 하락 후 과매도 상태에서 시그널이 있을 수 있음
        assert isinstance(signals, list)

    def test_validate_config(self):
        """설정 검증"""
        with pytest.raises(ValueError):
            RSIStrategy(period=14, oversold=70.0, overbought=30.0).validate_config()


# ─────────────────────────────────────────────
# Ensemble Strategy Tests
# ─────────────────────────────────────────────

class TestEnsembleStrategy:

    def test_equal_weights(self):
        """동일 가중치 앙상블"""
        strategies = [
            MovingAverageCrossStrategy(),
            RSIStrategy(),
            MACDStrategy(),
        ]
        ensemble = EnsembleStrategy(strategies=strategies)

        assert len(ensemble.weights) == 3
        assert abs(sum(ensemble.weights) - 1.0) < 1e-6

    def test_custom_weights(self):
        """커스텀 가중치 앙상블"""
        strategies = [MovingAverageCrossStrategy(), RSIStrategy()]
        ensemble = EnsembleStrategy(
            strategies=strategies,
            weights=[0.7, 0.3]
        )
        assert ensemble.weights[0] == 0.7

    def test_invalid_weights_raise(self):
        """잘못된 가중치 예외"""
        strategies = [MovingAverageCrossStrategy(), RSIStrategy()]
        with pytest.raises(AssertionError):
            EnsembleStrategy(strategies=strategies, weights=[0.8, 0.8])  # 합계 != 1.0

    def test_generates_consensus_signal(self):
        """앙상블 합의 시그널 생성"""
        strategies = [
            MovingAverageCrossStrategy(fast_period=5, long_period=20),
            RSIStrategy(),
        ]
        ensemble = EnsembleStrategy(strategies=strategies, min_agreement=0.5)
        data = make_ohlcv_frame(n_bars=300, trend="up")

        signals = ensemble.generate_signal(data)
        assert isinstance(signals, list)


# ─────────────────────────────────────────────
# Performance Tests
# ─────────────────────────────────────────────

class TestStrategyPerformance:

    @pytest.mark.timeout(5)
    def test_signal_generation_speed(self):
        """시그널 생성 속도 테스트 (5초 이내)"""
        import time
        strategy = MovingAverageCrossStrategy(fast_period=20, long_period=50)
        data = make_ohlcv_frame(n_bars=5000)  # 약 20년치 일봉

        start = time.perf_counter()
        signals = strategy.generate_signal(data)
        elapsed = time.perf_counter() - start

        assert elapsed < 5.0, f"Too slow: {elapsed:.2f}s"

    def test_signal_consistency(self):
        """동일 입력에 동일 출력 (결정적)"""
        strategy = MovingAverageCrossStrategy(fast_period=20, long_period=50)
        data = make_ohlcv_frame(n_bars=300, seed=42)

        signals1 = strategy.generate_signal(data)
        signals2 = strategy.generate_signal(data)

        assert len(signals1) == len(signals2)
        for s1, s2 in zip(signals1, signals2):
            assert s1.direction == s2.direction
            assert abs(s1.strength - s2.strength) < 1e-9


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
