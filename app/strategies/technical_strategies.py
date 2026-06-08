"""
=============================================================================
Technical Analysis Strategies — Production-Grade Implementation
=============================================================================
기술적 분석 기반 전략 구현.
모든 전략은 BaseStrategy를 구현하며 plug-in 방식으로 동작.
=============================================================================
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

import numpy as np
import pandas as pd

from app.core.base_classes import BaseStrategy
from app.core.types import (
    OHLCVFrame, Position, Signal, SignalDirection, SignalSource,
    Symbol, Timeframe,
)
from app.config.logging_config import get_logger

logger = get_logger(__name__)


def _make_signal(
    strategy_id: str,
    symbol: Symbol,
    direction: SignalDirection,
    strength: float,
    confidence: float,
    timeframe: Timeframe,
    metadata: dict = None,
) -> Signal:
    """시그널 팩토리 함수"""
    return Signal(
        signal_id=str(uuid.uuid4()),
        strategy_id=strategy_id,
        symbol=symbol,
        direction=direction,
        strength=strength,
        confidence=confidence,
        timestamp=datetime.utcnow(),
        source=SignalSource.TECHNICAL,
        timeframe=timeframe,
        metadata=metadata or {},
    )


# ─────────────────────────────────────────────
# 이동평균 교차 전략
# ─────────────────────────────────────────────
class MovingAverageCrossStrategy(BaseStrategy):
    """
    이중 이동평균 교차 전략 (Golden/Dead Cross).

    진입: 단기 MA > 장기 MA → 매수
          단기 MA < 장기 MA → 매도/공매도
    청산: 반대 신호 발생 시

    파라미터:
        fast_period: 단기 이동평균 기간 (기본: 20)
        long_period: 장기 이동평균 기간 (기본: 50)
        ma_type: 이동평균 유형 ('sma', 'ema', 'wma')
    """

    def __init__(
        self,
        strategy_id: str = None,
        fast_period: int = 20,
        long_period: int = 50,
        ma_type: str = "ema",
        allow_short: bool = False,
    ) -> None:
        super().__init__(
            strategy_id=strategy_id or f"ma_cross_{fast_period}_{long_period}",
            name=f"MA Cross ({fast_period}/{long_period} {ma_type.upper()})",
            config={
                "fast_period": fast_period,
                "long_period": long_period,
                "ma_type": ma_type,
                "allow_short": allow_short,
            },
        )
        self.fast_period = fast_period
        self.long_period = long_period
        self.ma_type = ma_type
        self.allow_short = allow_short

    def _calc_ma(self, prices: pd.Series, period: int) -> pd.Series:
        """이동평균 계산"""
        if self.ma_type == "ema":
            return prices.ewm(span=period, adjust=False).mean()
        elif self.ma_type == "wma":
            weights = np.arange(1, period + 1)
            return prices.rolling(period).apply(
                lambda x: np.dot(x, weights) / weights.sum(), raw=True
            )
        else:  # sma
            return prices.rolling(period).mean()

    def generate_signal(
        self,
        data: OHLCVFrame,
        current_positions: dict[Symbol, Position] = None,
    ) -> list[Signal]:
        """골든/데드 크로스 시그널 생성"""
        if data.n_bars < self.long_period + 2:
            return []

        close = data.data["close"]
        fast_ma = self._calc_ma(close, self.fast_period)
        slow_ma = self._calc_ma(close, self.long_period)

        # 교차 탐지
        prev_fast = fast_ma.iloc[-2]
        prev_slow = slow_ma.iloc[-2]
        curr_fast = fast_ma.iloc[-1]
        curr_slow = slow_ma.iloc[-1]

        signals = []

        # 골든 크로스 (단기 > 장기 돌파)
        if prev_fast <= prev_slow and curr_fast > curr_slow:
            strength = min((curr_fast - curr_slow) / curr_slow * 100, 1.0)
            signals.append(_make_signal(
                strategy_id=self.strategy_id,
                symbol=data.symbol,
                direction=SignalDirection.BUY,
                strength=abs(strength),
                confidence=0.7,
                timeframe=data.timeframe,
                metadata={
                    "fast_ma": curr_fast,
                    "slow_ma": curr_slow,
                    "cross_type": "golden_cross",
                },
            ))

        # 데드 크로스 (단기 < 장기 돌파)
        elif prev_fast >= prev_slow and curr_fast < curr_slow:
            strength = min((curr_slow - curr_fast) / curr_slow * 100, 1.0)
            direction = SignalDirection.SELL if self.allow_short else SignalDirection.SELL
            signals.append(_make_signal(
                strategy_id=self.strategy_id,
                symbol=data.symbol,
                direction=direction,
                strength=abs(strength),
                confidence=0.7,
                timeframe=data.timeframe,
                metadata={
                    "fast_ma": curr_fast,
                    "slow_ma": curr_slow,
                    "cross_type": "dead_cross",
                },
            ))

        return signals

    def calculate_position_size(
        self,
        signal: Signal,
        portfolio_value: float,
        current_price: float,
        risk_per_trade: float = 0.01,
    ) -> float:
        """ATR 기반 포지션 크기 계산"""
        risk_amount = portfolio_value * risk_per_trade
        # 가격의 2% ATR 가정 (실제로는 데이터에서 계산)
        atr_estimate = current_price * 0.02
        quantity = risk_amount / atr_estimate
        return max(0.0, quantity)

    def validate_config(self) -> bool:
        if self.fast_period >= self.long_period:
            raise ValueError(f"fast_period({self.fast_period}) >= long_period({self.long_period})")
        if self.fast_period < 2:
            raise ValueError("fast_period must be >= 2")
        return True


# ─────────────────────────────────────────────
# RSI 전략
# ─────────────────────────────────────────────
class RSIStrategy(BaseStrategy):
    """
    RSI 과매수/과매도 전략.

    진입: RSI < oversold_threshold → 매수 (과매도)
          RSI > overbought_threshold → 매도 (과매수)
    청산: RSI 중립 구간 진입 시

    파라미터:
        period: RSI 기간 (기본: 14)
        oversold: 과매도 기준 (기본: 30)
        overbought: 과매수 기준 (기본: 70)
    """

    def __init__(
        self,
        strategy_id: str = None,
        period: int = 14,
        oversold: float = 30.0,
        overbought: float = 70.0,
        allow_short: bool = True,
    ) -> None:
        super().__init__(
            strategy_id=strategy_id or f"rsi_{period}",
            name=f"RSI Strategy (period={period}, OS={oversold}, OB={overbought})",
            config={
                "period": period,
                "oversold": oversold,
                "overbought": overbought,
                "allow_short": allow_short,
            },
        )
        self.period = period
        self.oversold = oversold
        self.overbought = overbought
        self.allow_short = allow_short

    @staticmethod
    def _calc_rsi(close: pd.Series, period: int) -> pd.Series:
        """
        RSI 계산 (Wilder's Smoothing Method).
        pandas_ta 없이 순수 pandas/numpy로 구현.
        """
        delta = close.diff()
        gains = delta.where(delta > 0, 0.0)
        losses = -delta.where(delta < 0, 0.0)

        avg_gain = gains.ewm(com=period - 1, adjust=False).mean()
        avg_loss = losses.ewm(com=period - 1, adjust=False).mean()

        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)  # NaN → 중립값

    def generate_signal(
        self,
        data: OHLCVFrame,
        current_positions: dict[Symbol, Position] = None,
    ) -> list[Signal]:
        """RSI 기반 시그널 생성"""
        if data.n_bars < self.period + 5:
            return []

        close = data.data["close"]
        rsi = self._calc_rsi(close, self.period)

        curr_rsi = rsi.iloc[-1]
        prev_rsi = rsi.iloc[-2]

        signals = []

        # 과매도 → 매수 시그널
        if curr_rsi < self.oversold:
            # RSI가 과매도 구간에서 반등 중일 때 더 강한 시그널
            is_bouncing = curr_rsi > prev_rsi
            strength = (self.oversold - curr_rsi) / self.oversold
            confidence = 0.75 if is_bouncing else 0.55

            signals.append(_make_signal(
                strategy_id=self.strategy_id,
                symbol=data.symbol,
                direction=SignalDirection.BUY,
                strength=min(strength, 1.0),
                confidence=confidence,
                timeframe=data.timeframe,
                metadata={
                    "rsi": round(curr_rsi, 2),
                    "rsi_prev": round(prev_rsi, 2),
                    "condition": "oversold",
                    "bouncing": is_bouncing,
                },
            ))

        # 과매수 → 매도 시그널
        elif curr_rsi > self.overbought:
            is_declining = curr_rsi < prev_rsi
            strength = (curr_rsi - self.overbought) / (100 - self.overbought)
            confidence = 0.75 if is_declining else 0.55

            direction = SignalDirection.SELL if self.allow_short else SignalDirection.SELL
            signals.append(_make_signal(
                strategy_id=self.strategy_id,
                symbol=data.symbol,
                direction=direction,
                strength=min(strength, 1.0),
                confidence=confidence,
                timeframe=data.timeframe,
                metadata={
                    "rsi": round(curr_rsi, 2),
                    "rsi_prev": round(prev_rsi, 2),
                    "condition": "overbought",
                    "declining": is_declining,
                },
            ))

        return signals

    def calculate_position_size(
        self,
        signal: Signal,
        portfolio_value: float,
        current_price: float,
        risk_per_trade: float = 0.01,
    ) -> float:
        """시그널 강도 가중 포지션 크기"""
        base_risk = portfolio_value * risk_per_trade
        adjusted_risk = base_risk * signal.strength * signal.confidence
        # 단순 고정 달러 리스크
        return adjusted_risk / (current_price * 0.02)

    def validate_config(self) -> bool:
        if not 0 < self.oversold < self.overbought < 100:
            raise ValueError("Invalid RSI thresholds")
        if self.period < 2:
            raise ValueError("period must be >= 2")
        return True


# ─────────────────────────────────────────────
# MACD 전략
# ─────────────────────────────────────────────
class MACDStrategy(BaseStrategy):
    """
    MACD (Moving Average Convergence Divergence) 전략.

    시그널:
    - MACD 라인이 시그널 라인 상향 돌파 → 매수
    - MACD 라인이 시그널 라인 하향 돌파 → 매도
    - 히스토그램 방향 변화로 확인
    """

    def __init__(
        self,
        strategy_id: str = None,
        fast: int = 12,
        slow: int = 26,
        signal_period: int = 9,
    ) -> None:
        super().__init__(
            strategy_id=strategy_id or f"macd_{fast}_{slow}_{signal_period}",
            name=f"MACD ({fast}/{slow}/{signal_period})",
            config={"fast": fast, "slow": slow, "signal_period": signal_period},
        )
        self.fast = fast
        self.slow = slow
        self.signal_period = signal_period

    def _calc_macd(self, close: pd.Series):
        """MACD 계산"""
        ema_fast = close.ewm(span=self.fast, adjust=False).mean()
        ema_slow = close.ewm(span=self.slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=self.signal_period, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    def generate_signal(
        self,
        data: OHLCVFrame,
        current_positions: dict[Symbol, Position] = None,
    ) -> list[Signal]:
        """MACD 시그널 생성"""
        min_bars = self.slow + self.signal_period + 5
        if data.n_bars < min_bars:
            return []

        close = data.data["close"]
        macd_line, signal_line, histogram = self._calc_macd(close)

        # 최근 2봉 비교
        curr_macd = macd_line.iloc[-1]
        prev_macd = macd_line.iloc[-2]
        curr_signal = signal_line.iloc[-1]
        prev_signal = signal_line.iloc[-2]
        curr_hist = histogram.iloc[-1]
        prev_hist = histogram.iloc[-2]

        signals = []

        # MACD 상향 돌파
        if prev_macd <= prev_signal and curr_macd > curr_signal:
            # 0 라인 위 돌파 시 더 강한 시그널
            is_above_zero = curr_macd > 0
            strength = min(abs(curr_hist) / (close.iloc[-1] * 0.001), 1.0)
            confidence = 0.80 if is_above_zero else 0.65

            signals.append(_make_signal(
                strategy_id=self.strategy_id,
                symbol=data.symbol,
                direction=SignalDirection.BUY,
                strength=strength,
                confidence=confidence,
                timeframe=data.timeframe,
                metadata={
                    "macd": round(curr_macd, 6),
                    "signal": round(curr_signal, 6),
                    "histogram": round(curr_hist, 6),
                    "above_zero": is_above_zero,
                },
            ))

        # MACD 하향 돌파
        elif prev_macd >= prev_signal and curr_macd < curr_signal:
            is_below_zero = curr_macd < 0
            strength = min(abs(curr_hist) / (close.iloc[-1] * 0.001), 1.0)
            confidence = 0.80 if is_below_zero else 0.65

            signals.append(_make_signal(
                strategy_id=self.strategy_id,
                symbol=data.symbol,
                direction=SignalDirection.SELL,
                strength=strength,
                confidence=confidence,
                timeframe=data.timeframe,
                metadata={
                    "macd": round(curr_macd, 6),
                    "signal": round(curr_signal, 6),
                    "histogram": round(curr_hist, 6),
                    "below_zero": is_below_zero,
                },
            ))

        return signals

    def calculate_position_size(
        self,
        signal: Signal,
        portfolio_value: float,
        current_price: float,
        risk_per_trade: float = 0.01,
    ) -> float:
        return (portfolio_value * risk_per_trade * signal.confidence) / (current_price * 0.02)

    def validate_config(self) -> bool:
        if self.fast >= self.slow:
            raise ValueError("fast must be < slow")
        return True


# ─────────────────────────────────────────────
# 볼린저 밴드 평균 회귀 전략
# ─────────────────────────────────────────────
class BollingerBandMeanReversionStrategy(BaseStrategy):
    """
    볼린저 밴드 기반 평균 회귀 전략.

    진입: 가격이 하단 밴드 아래 → 매수 (과매도)
          가격이 상단 밴드 위 → 매도 (과매수)
    청산: 중간 밴드 (SMA) 도달 시

    볼린저 밴드 스퀴즈 감지로 돌파 회피.
    """

    def __init__(
        self,
        strategy_id: str = None,
        period: int = 20,
        std_dev: float = 2.0,
    ) -> None:
        super().__init__(
            strategy_id=strategy_id or f"bb_mr_{period}_{std_dev}",
            name=f"Bollinger Band Mean Reversion ({period}, {std_dev}σ)",
            config={"period": period, "std_dev": std_dev},
        )
        self.period = period
        self.std_dev = std_dev

    def _calc_bands(self, close: pd.Series):
        """볼린저 밴드 계산"""
        sma = close.rolling(self.period).mean()
        std = close.rolling(self.period).std()
        upper = sma + self.std_dev * std
        lower = sma - self.std_dev * std
        bandwidth = (upper - lower) / sma  # 밴드 폭
        return sma, upper, lower, bandwidth

    def generate_signal(
        self,
        data: OHLCVFrame,
        current_positions: dict[Symbol, Position] = None,
    ) -> list[Signal]:
        """볼린저 밴드 시그널"""
        if data.n_bars < self.period + 5:
            return []

        close = data.data["close"]
        sma, upper, lower, bandwidth = self._calc_bands(close)

        curr_price = close.iloc[-1]
        curr_upper = upper.iloc[-1]
        curr_lower = lower.iloc[-1]
        curr_bw = bandwidth.iloc[-1]
        avg_bw = bandwidth.rolling(20).mean().iloc[-1]

        # 볼린저 스퀴즈 감지 (밴드 폭이 평균보다 좁을 때)
        is_squeeze = curr_bw < avg_bw * 0.8 if pd.notna(avg_bw) else False

        signals = []

        # 스퀴즈 상태에서는 평균 회귀 전략 비활성화 (돌파 가능성 높음)
        if is_squeeze:
            return signals

        # 하단 밴드 이탈 → 매수
        if curr_price < curr_lower:
            penetration = (curr_lower - curr_price) / curr_lower
            strength = min(penetration * 10, 1.0)

            signals.append(_make_signal(
                strategy_id=self.strategy_id,
                symbol=data.symbol,
                direction=SignalDirection.BUY,
                strength=strength,
                confidence=0.7,
                timeframe=data.timeframe,
                metadata={
                    "price": curr_price,
                    "lower_band": round(curr_lower, 4),
                    "upper_band": round(curr_upper, 4),
                    "sma": round(sma.iloc[-1], 4),
                    "bandwidth": round(curr_bw, 4),
                    "penetration_pct": round(penetration * 100, 2),
                },
            ))

        # 상단 밴드 이탈 → 매도
        elif curr_price > curr_upper:
            penetration = (curr_price - curr_upper) / curr_upper
            strength = min(penetration * 10, 1.0)

            signals.append(_make_signal(
                strategy_id=self.strategy_id,
                symbol=data.symbol,
                direction=SignalDirection.SELL,
                strength=strength,
                confidence=0.7,
                timeframe=data.timeframe,
                metadata={
                    "price": curr_price,
                    "upper_band": round(curr_upper, 4),
                    "lower_band": round(curr_lower, 4),
                    "penetration_pct": round(penetration * 100, 2),
                },
            ))

        return signals

    def calculate_position_size(
        self,
        signal: Signal,
        portfolio_value: float,
        current_price: float,
        risk_per_trade: float = 0.01,
    ) -> float:
        return (portfolio_value * risk_per_trade) / (current_price * 0.02)

    def validate_config(self) -> bool:
        if self.period < 5:
            raise ValueError("period must be >= 5")
        if self.std_dev <= 0:
            raise ValueError("std_dev must be positive")
        return True


# ─────────────────────────────────────────────
# 모멘텀 전략
# ─────────────────────────────────────────────
class MomentumStrategy(BaseStrategy):
    """
    가격 모멘텀 전략.

    Fama-French 스타일의 시계열 모멘텀.
    최근 성과가 좋은 종목 매수, 나쁜 종목 매도.

    파라미터:
        lookback: 모멘텀 측정 기간 (기본: 12개월 = 252일)
        skip: 최근 제외 기간 (반전 효과 방지, 기본: 21일)
    """

    def __init__(
        self,
        strategy_id: str = None,
        lookback: int = 252,
        skip: int = 21,
        threshold: float = 0.05,  # 최소 모멘텀 임계값
    ) -> None:
        super().__init__(
            strategy_id=strategy_id or f"momentum_{lookback}",
            name=f"Momentum Strategy ({lookback}d lookback)",
            config={"lookback": lookback, "skip": skip, "threshold": threshold},
        )
        self.lookback = lookback
        self.skip = skip
        self.threshold = threshold

    def generate_signal(
        self,
        data: OHLCVFrame,
        current_positions: dict[Symbol, Position] = None,
    ) -> list[Signal]:
        """모멘텀 시그널"""
        min_bars = self.lookback + self.skip + 10
        if data.n_bars < min_bars:
            return []

        close = data.data["close"]

        # 모멘텀 계산: (skip일 전 가격) / (lookback+skip일 전 가격) - 1
        curr_price = close.iloc[-self.skip - 1]
        past_price = close.iloc[-(self.lookback + self.skip)]

        momentum = (curr_price / past_price) - 1

        signals = []

        if momentum > self.threshold:
            # 강한 양의 모멘텀
            strength = min(momentum / 0.5, 1.0)
            signals.append(_make_signal(
                strategy_id=self.strategy_id,
                symbol=data.symbol,
                direction=SignalDirection.BUY,
                strength=strength,
                confidence=0.65,
                timeframe=data.timeframe,
                metadata={
                    "momentum": round(momentum * 100, 2),
                    "lookback": self.lookback,
                    "curr_price": curr_price,
                    "past_price": past_price,
                },
            ))
        elif momentum < -self.threshold:
            # 강한 음의 모멘텀
            strength = min(abs(momentum) / 0.5, 1.0)
            signals.append(_make_signal(
                strategy_id=self.strategy_id,
                symbol=data.symbol,
                direction=SignalDirection.SELL,
                strength=strength,
                confidence=0.65,
                timeframe=data.timeframe,
                metadata={
                    "momentum": round(momentum * 100, 2),
                    "lookback": self.lookback,
                },
            ))

        return signals

    def calculate_position_size(
        self,
        signal: Signal,
        portfolio_value: float,
        current_price: float,
        risk_per_trade: float = 0.01,
    ) -> float:
        return (portfolio_value * risk_per_trade * signal.strength) / (current_price * 0.02)

    def validate_config(self) -> bool:
        if self.lookback <= self.skip:
            raise ValueError("lookback must be > skip")
        return True


# ─────────────────────────────────────────────
# 앙상블 전략
# ─────────────────────────────────────────────
class EnsembleStrategy(BaseStrategy):
    """
    여러 전략의 시그널을 앙상블.

    가중 투표 방식으로 최종 시그널 결정.
    각 전략의 신뢰도(confidence) 기반 가중치 적용.
    """

    def __init__(
        self,
        strategies: list[BaseStrategy],
        weights: Optional[list[float]] = None,
        min_agreement: float = 0.5,  # 최소 동의 비율
        strategy_id: str = None,
    ) -> None:
        super().__init__(
            strategy_id=strategy_id or "ensemble",
            name=f"Ensemble ({len(strategies)} strategies)",
            config={"n_strategies": len(strategies), "min_agreement": min_agreement},
        )
        self.strategies = strategies
        self.weights = weights or [1.0 / len(strategies)] * len(strategies)
        self.min_agreement = min_agreement

        assert len(self.weights) == len(strategies), "weights length must match strategies"
        assert abs(sum(self.weights) - 1.0) < 1e-6, "weights must sum to 1.0"

    def generate_signal(
        self,
        data: OHLCVFrame,
        current_positions: dict[Symbol, Position] = None,
    ) -> list[Signal]:
        """앙상블 시그널 생성"""
        all_signals = []

        for strategy, weight in zip(self.strategies, self.weights):
            try:
                sub_signals = strategy.generate_signal(data, current_positions)
                for sig in sub_signals:
                    all_signals.append((sig, weight))
            except Exception as e:
                logger.error(
                    "ensemble_strategy_failed",
                    strategy=strategy.name,
                    error=str(e),
                )

        if not all_signals:
            return []

        # 가중 투표
        buy_weight = sum(w for s, w in all_signals if s.is_long)
        sell_weight = sum(w for s, w in all_signals if s.is_short)
        total_weight = buy_weight + sell_weight

        if total_weight == 0:
            return []

        buy_ratio = buy_weight / total_weight

        # 최소 동의 비율 미달 시 중립
        if buy_ratio < self.min_agreement and buy_ratio > (1 - self.min_agreement):
            return []

        direction = SignalDirection.BUY if buy_ratio > 0.5 else SignalDirection.SELL
        consensus_strength = abs(buy_ratio - 0.5) * 2  # 0 ~ 1

        avg_confidence = np.mean([s.confidence for s, _ in all_signals])

        return [_make_signal(
            strategy_id=self.strategy_id,
            symbol=data.symbol,
            direction=direction,
            strength=consensus_strength,
            confidence=avg_confidence,
            timeframe=data.timeframe,
            metadata={
                "buy_weight": buy_weight,
                "sell_weight": sell_weight,
                "consensus_ratio": buy_ratio,
                "n_strategies": len(self.strategies),
            },
        )]

    def calculate_position_size(
        self,
        signal: Signal,
        portfolio_value: float,
        current_price: float,
        risk_per_trade: float = 0.01,
    ) -> float:
        return (portfolio_value * risk_per_trade * signal.confidence) / (current_price * 0.02)

    def validate_config(self) -> bool:
        for strategy in self.strategies:
            strategy.validate_config()
        return True
