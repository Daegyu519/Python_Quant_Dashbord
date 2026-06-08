"""백테스팅 엔진 단위 테스트"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from datetime import datetime

from app.core.types import BacktestMetrics, OHLCVFrame, Timeframe
from app.backtesting.vectorized_engine import (
    VectorizedBacktestEngine,
    calculate_metrics,
    _calc_max_drawdown,
    _calc_equity_curve,
)


class TestVectorizedEngine:

    @pytest.fixture
    def engine(self):
        return VectorizedBacktestEngine(
            commission_bps=5.0,
            slippage_bps=2.0,
            initial_capital=100_000_000.0,
        )

    def test_calc_max_drawdown_no_drawdown(self):
        """낙폭 없는 경우"""
        equity = np.array([100.0, 110.0, 120.0, 130.0])
        mdd, duration = _calc_max_drawdown(equity)
        assert mdd == 0.0
        assert duration == 0

    def test_calc_max_drawdown_known_case(self):
        """알려진 낙폭 계산"""
        equity = np.array([100.0, 120.0, 80.0, 90.0, 100.0])
        mdd, duration = _calc_max_drawdown(equity)
        # Peak=120, Trough=80, MDD = (80-120)/120 = -33.33%
        assert abs(mdd - (-0.3333)) < 0.01

    def test_equity_curve_long_position(self):
        """롱 포지션 자산 곡선"""
        close = np.array([100.0, 105.0, 110.0, 108.0, 115.0])
        signals = np.array([1, 1, 1, 1, 1], dtype=np.int8)
        equity, returns, commission = _calc_equity_curve(
            signals, close, 1_000_000.0, 5.0, 2.0
        )
        # 상승 추세에서 롱 포지션 → 수익
        assert equity[-1] > equity[0]

    def test_equity_curve_neutral_no_change(self):
        """중립 포지션 자산 곡선 변화 없음"""
        close = np.array([100.0, 105.0, 110.0, 108.0, 115.0])
        signals = np.zeros(5, dtype=np.int8)
        equity, returns, commission = _calc_equity_curve(
            signals, close, 1_000_000.0, 0.0, 0.0
        )
        # 포지션 없으면 자산 불변
        np.testing.assert_allclose(equity, 1_000_000.0)

    def test_calculate_metrics_basic(self):
        """기본 지표 계산"""
        n = 252
        equity = np.linspace(100_000.0, 120_000.0, n)
        returns = np.diff(equity) / equity[:-1]
        returns = np.insert(returns, 0, 0.0)

        metrics = calculate_metrics(
            equity=equity,
            returns=returns,
            initial_capital=100_000.0,
            start_date=datetime(2020, 1, 1),
            end_date=datetime(2021, 1, 1),
        )

        assert metrics.total_return > 0
        assert metrics.cagr > 0
        assert -1.0 <= metrics.max_drawdown <= 0.0
        assert metrics.sharpe_ratio is not None

    def test_calculate_metrics_negative_returns(self):
        """손실 케이스 지표 계산"""
        n = 252
        equity = np.linspace(100_000.0, 80_000.0, n)
        returns = np.diff(equity) / equity[:-1]
        returns = np.insert(returns, 0, 0.0)

        metrics = calculate_metrics(
            equity=equity,
            returns=returns,
            initial_capital=100_000.0,
        )

        assert metrics.total_return < 0
        assert metrics.cagr < 0
        assert metrics.max_drawdown < 0

    def test_calculate_metrics_sharpe_positive_for_uptrend(self):
        """상승 추세에서 Sharpe > 0"""
        n = 252
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.01, n)  # 양의 평균 수익률
        equity = 100_000.0 * np.cumprod(1 + returns)

        metrics = calculate_metrics(
            equity=equity,
            returns=returns,
            initial_capital=100_000.0,
        )

        assert metrics.sharpe_ratio > 0

    def test_engine_initialization_warms_up_numba(self):
        """엔진 초기화 시 Numba 워밍업"""
        engine = VectorizedBacktestEngine()
        # 초기화가 오류 없이 완료되어야 함
        assert engine is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
