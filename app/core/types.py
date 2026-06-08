"""
=============================================================================
Core Type Definitions — Institution-Grade Type System
=============================================================================
모든 모듈이 공유하는 핵심 타입 정의.
타입 안전성을 극대화하여 런타임 오류를 컴파일 타임에 잡는다.
=============================================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum, auto
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────
# 기본 수치 타입
# ─────────────────────────────────────────────
Price = float           # 가격
Volume = float          # 거래량
Timestamp = int         # Unix 타임스탬프 (ms)
Symbol = str            # 종목 코드 (예: "AAPL", "BTC/USDT")
StrategyID = str        # 전략 고유 ID
SignalID = str          # 시그널 고유 ID

# NumPy 배열 타입 별칭
PriceArray = np.ndarray     # float64 배열
VolumeArray = np.ndarray    # float64 배열
DatetimeIndex = pd.DatetimeIndex


# ─────────────────────────────────────────────
# 시장 열거형
# ─────────────────────────────────────────────
class Market(str, Enum):
    """지원 시장"""
    KRX = "KRX"             # 한국거래소
    NYSE = "NYSE"           # 뉴욕증권거래소
    NASDAQ = "NASDAQ"       # 나스닥
    BINANCE = "BINANCE"     # 바이낸스 (암호화폐)
    BYBIT = "BYBIT"         # 바이빗
    FX = "FX"               # 외환
    FUTURES = "FUTURES"     # 선물


class AssetClass(str, Enum):
    """자산 유형"""
    EQUITY = "EQUITY"           # 주식
    CRYPTO = "CRYPTO"           # 암호화폐
    FUTURES = "FUTURES"         # 선물
    OPTIONS = "OPTIONS"         # 옵션
    FOREX = "FOREX"             # 외환
    FIXED_INCOME = "FIXED_INCOME"  # 채권
    COMMODITY = "COMMODITY"     # 원자재


class Timeframe(str, Enum):
    """
    타임프레임.
    분 단위로 정규화 가능.
    """
    TICK = "1T"
    ONE_MIN = "1m"
    THREE_MIN = "3m"
    FIVE_MIN = "5m"
    FIFTEEN_MIN = "15m"
    THIRTY_MIN = "30m"
    ONE_HOUR = "1h"
    TWO_HOUR = "2h"
    FOUR_HOUR = "4h"
    SIX_HOUR = "6h"
    TWELVE_HOUR = "12h"
    ONE_DAY = "1d"
    THREE_DAY = "3d"
    ONE_WEEK = "1w"
    ONE_MONTH = "1M"

    @property
    def minutes(self) -> int:
        """분 단위 변환"""
        mapping = {
            "1T": 0, "1m": 1, "3m": 3, "5m": 5, "15m": 15,
            "30m": 30, "1h": 60, "2h": 120, "4h": 240,
            "6h": 360, "12h": 720, "1d": 1440,
            "3d": 4320, "1w": 10080, "1M": 43200,
        }
        return mapping.get(self.value, 0)


class OrderSide(str, Enum):
    """주문 방향"""
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    """주문 유형"""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"
    TRAILING_STOP = "TRAILING_STOP"


class OrderStatus(str, Enum):
    """주문 상태"""
    PENDING = "PENDING"
    OPEN = "OPEN"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


class PositionSide(str, Enum):
    """포지션 방향"""
    LONG = "LONG"
    SHORT = "SHORT"
    FLAT = "FLAT"


class SignalDirection(int, Enum):
    """시그널 방향"""
    STRONG_BUY = 2
    BUY = 1
    NEUTRAL = 0
    SELL = -1
    STRONG_SELL = -2


class SignalSource(str, Enum):
    """시그널 소스"""
    TECHNICAL = "TECHNICAL"
    ML_MODEL = "ML_MODEL"
    FUNDAMENTAL = "FUNDAMENTAL"
    SENTIMENT = "SENTIMENT"
    ENSEMBLE = "ENSEMBLE"


class RegimeType(str, Enum):
    """시장 레짐"""
    BULL_TREND = "BULL_TREND"
    BEAR_TREND = "BEAR_TREND"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    LOW_VOLATILITY = "LOW_VOLATILITY"
    SIDEWAYS = "SIDEWAYS"
    CRISIS = "CRISIS"


# ─────────────────────────────────────────────
# OHLCV 데이터 구조
# ─────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class OHLCV:
    """
    단일 OHLCV 캔들.
    immutable + slots → 최소 메모리, 최대 속도.
    """
    symbol: Symbol
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    timeframe: Timeframe = Timeframe.ONE_DAY

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "timeframe": self.timeframe.value,
        }

    @property
    def typical_price(self) -> float:
        """전형적 가격 = (H+L+C)/3"""
        return (self.high + self.low + self.close) / 3.0

    @property
    def range(self) -> float:
        return self.high - self.low

    @property
    def body_size(self) -> float:
        return abs(self.close - self.open)


@dataclass
class OHLCVFrame:
    """
    복수 OHLCV 데이터 (DataFrame 래퍼).
    Polars DataFrame 우선, pandas 호환.
    """
    symbol: Symbol
    timeframe: Timeframe
    data: pd.DataFrame  # columns: [open, high, low, close, volume]

    def __post_init__(self) -> None:
        required_cols = {"open", "high", "low", "close", "volume"}
        actual_cols = set(self.data.columns)
        if not required_cols.issubset(actual_cols):
            missing = required_cols - actual_cols
            raise ValueError(f"OHLCVFrame missing columns: {missing}")

    @property
    def close_prices(self) -> np.ndarray:
        return self.data["close"].to_numpy(dtype=np.float64)

    @property
    def volume_data(self) -> np.ndarray:
        return self.data["volume"].to_numpy(dtype=np.float64)

    @property
    def returns(self) -> np.ndarray:
        """로그 수익률"""
        close = self.close_prices
        return np.log(close[1:] / close[:-1])

    @property
    def n_bars(self) -> int:
        return len(self.data)


# ─────────────────────────────────────────────
# 틱 데이터
# ─────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class Tick:
    """실시간 틱 데이터"""
    symbol: Symbol
    timestamp: datetime
    price: float
    volume: float
    side: OrderSide  # BUYER_MAKER 여부
    trade_id: str = ""


@dataclass(frozen=True, slots=True)
class OrderBookLevel:
    """호가창 단일 레벨"""
    price: float
    quantity: float


@dataclass
class OrderBook:
    """호가창 스냅샷"""
    symbol: Symbol
    timestamp: datetime
    bids: list[OrderBookLevel]  # 매수 호가 (내림차순)
    asks: list[OrderBookLevel]  # 매도 호가 (오름차순)

    @property
    def best_bid(self) -> float:
        return self.bids[0].price if self.bids else 0.0

    @property
    def best_ask(self) -> float:
        return self.asks[0].price if self.asks else 0.0

    @property
    def mid_price(self) -> float:
        return (self.best_bid + self.best_ask) / 2.0

    @property
    def spread(self) -> float:
        return self.best_ask - self.best_bid

    @property
    def spread_bps(self) -> float:
        """스프레드 (bps)"""
        if self.mid_price == 0:
            return 0.0
        return (self.spread / self.mid_price) * 10_000


# ─────────────────────────────────────────────
# 시그널 데이터 구조
# ─────────────────────────────────────────────
@dataclass
class Signal:
    """
    트레이딩 시그널.
    전략/ML 모델이 생성하며 signal_engine이 처리.
    """
    signal_id: SignalID
    strategy_id: StrategyID
    symbol: Symbol
    direction: SignalDirection
    strength: float         # 0.0 ~ 1.0
    confidence: float       # 0.0 ~ 1.0 (모델 확신도)
    timestamp: datetime
    source: SignalSource
    timeframe: Timeframe
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    expiry: Optional[datetime] = None

    @property
    def is_long(self) -> bool:
        return self.direction in (SignalDirection.BUY, SignalDirection.STRONG_BUY)

    @property
    def is_short(self) -> bool:
        return self.direction in (SignalDirection.SELL, SignalDirection.STRONG_SELL)

    @property
    def is_valid(self) -> bool:
        if self.expiry and datetime.utcnow() > self.expiry:
            return False
        return 0.0 <= self.strength <= 1.0 and 0.0 <= self.confidence <= 1.0


# ─────────────────────────────────────────────
# 포지션 & 주문
# ─────────────────────────────────────────────
@dataclass
class Position:
    """현재 포지션 상태"""
    symbol: Symbol
    side: PositionSide
    quantity: float
    entry_price: float
    current_price: float
    entry_time: datetime
    strategy_id: StrategyID = ""
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    commission_paid: float = 0.0

    @property
    def market_value(self) -> float:
        return self.quantity * self.current_price

    @property
    def pnl_pct(self) -> float:
        if self.entry_price == 0:
            return 0.0
        raw_pnl = (self.current_price - self.entry_price) / self.entry_price
        return raw_pnl if self.side == PositionSide.LONG else -raw_pnl

    def update_price(self, price: float) -> None:
        self.current_price = price
        if self.side == PositionSide.LONG:
            self.unrealized_pnl = (price - self.entry_price) * self.quantity
        else:
            self.unrealized_pnl = (self.entry_price - price) * self.quantity


@dataclass
class Order:
    """주문 객체"""
    order_id: str
    symbol: Symbol
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float]  # MARKET 주문은 None
    strategy_id: StrategyID
    status: OrderStatus = OrderStatus.PENDING
    timestamp: datetime = field(default_factory=datetime.utcnow)
    filled_quantity: float = 0.0
    filled_price: float = 0.0
    commission: float = 0.0
    slippage: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_filled(self) -> bool:
        return self.status == OrderStatus.FILLED

    @property
    def fill_ratio(self) -> float:
        if self.quantity == 0:
            return 0.0
        return self.filled_quantity / self.quantity


# ─────────────────────────────────────────────
# 백테스트 결과
# ─────────────────────────────────────────────
@dataclass
class BacktestMetrics:
    """
    백테스트 성과 지표 전체 세트.
    기관급 리포팅에 필요한 모든 지표 포함.
    """
    # 수익 지표
    total_return: float = 0.0
    cagr: float = 0.0
    annualized_return: float = 0.0

    # 리스크 지표
    volatility: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_duration_days: int = 0
    calmar_ratio: float = 0.0

    # 리스크 조정 수익률
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    omega_ratio: float = 0.0

    # 거래 통계
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    avg_holding_days: float = 0.0

    # 시장 대비
    alpha: float = 0.0
    beta: float = 0.0
    information_ratio: float = 0.0
    treynor_ratio: float = 0.0

    # VaR
    var_95: float = 0.0
    cvar_95: float = 0.0
    var_99: float = 0.0

    # 기타
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    initial_capital: float = 0.0
    final_capital: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict
        return asdict(self)

    def to_report(self) -> str:
        """사람이 읽기 좋은 성과 리포트"""
        return f"""
╔══════════════════════════════════════════════════════╗
║              BACKTEST PERFORMANCE REPORT              ║
╠══════════════════════════════════════════════════════╣
║  기간: {self.start_date} ~ {self.end_date}
║  초기자본: {self.initial_capital:,.0f}  →  최종자본: {self.final_capital:,.0f}
╠══════════════════════════════════════════════════════╣
║  [수익률]
║  총 수익률:      {self.total_return*100:+.2f}%
║  연환산 수익률:  {self.cagr*100:+.2f}%
╠══════════════════════════════════════════════════════╣
║  [리스크 조정 수익률]
║  Sharpe Ratio:   {self.sharpe_ratio:.4f}
║  Sortino Ratio:  {self.sortino_ratio:.4f}
║  Calmar Ratio:   {self.calmar_ratio:.4f}
╠══════════════════════════════════════════════════════╣
║  [리스크]
║  연간 변동성:    {self.volatility*100:.2f}%
║  최대낙폭(MDD):  {self.max_drawdown*100:.2f}%
║  VaR (95%):      {self.var_95*100:.2f}%
║  CVaR (95%):     {self.cvar_95*100:.2f}%
╠══════════════════════════════════════════════════════╣
║  [거래 통계]
║  총 거래수:      {self.total_trades}
║  승률:           {self.win_rate*100:.1f}%
║  손익비:         {self.profit_factor:.2f}
║  평균 보유일:    {self.avg_holding_days:.1f}일
╠══════════════════════════════════════════════════════╣
║  [시장 대비]
║  Alpha:          {self.alpha*100:+.2f}%
║  Beta:           {self.beta:.4f}
╚══════════════════════════════════════════════════════╝
"""
