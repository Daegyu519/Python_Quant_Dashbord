"""
=============================================================================
Abstract Base Classes — SOLID Principles + Clean Architecture
=============================================================================
모든 핵심 컴포넌트의 인터페이스 정의.
의존성 역전 원칙(DIP)을 따라 구체 구현에서 분리.
=============================================================================
"""

from __future__ import annotations

import abc
from datetime import datetime
from typing import Any, AsyncGenerator, Dict, Generic, List, Optional, TypeVar

import pandas as pd

from app.core.types import (
    OHLCV, BacktestMetrics, Order, OrderBook, OHLCVFrame,
    Position, Signal, Symbol, Tick, Timeframe, SignalDirection,
)

# ─────────────────────────────────────────────
# 제네릭 타입 변수
# ─────────────────────────────────────────────
T = TypeVar("T")
ConfigType = TypeVar("ConfigType")


# ─────────────────────────────────────────────
# 데이터 수집기 인터페이스
# ─────────────────────────────────────────────
class BaseDataCollector(abc.ABC):
    """
    모든 데이터 소스 수집기의 추상 기반 클래스.

    구현 요구사항:
    - 비동기 OHLCV 수집
    - 실시간 WebSocket 스트리밍
    - Rate limit 준수
    - 자동 재연결
    """

    def __init__(self, source_name: str) -> None:
        self.source_name = source_name
        self._is_connected: bool = False

    @abc.abstractmethod
    async def fetch_ohlcv(
        self,
        symbol: Symbol,
        timeframe: Timeframe,
        start: datetime,
        end: Optional[datetime] = None,
        limit: int = 1000,
    ) -> OHLCVFrame:
        """OHLCV 데이터 비동기 수집"""
        ...

    @abc.abstractmethod
    async def fetch_ticker(self, symbol: Symbol) -> dict[str, Any]:
        """현재 가격 정보 조회"""
        ...

    @abc.abstractmethod
    async def fetch_orderbook(self, symbol: Symbol, depth: int = 20) -> OrderBook:
        """호가창 조회"""
        ...

    @abc.abstractmethod
    async def stream_ticks(
        self, symbols: list[Symbol]
    ) -> AsyncGenerator[Tick, None]:
        """실시간 틱 스트리밍"""
        ...

    @abc.abstractmethod
    async def stream_ohlcv(
        self, symbol: Symbol, timeframe: Timeframe
    ) -> AsyncGenerator[OHLCV, None]:
        """실시간 OHLCV 스트리밍"""
        ...

    @abc.abstractmethod
    def get_supported_symbols(self) -> list[Symbol]:
        """지원 종목 목록"""
        ...

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, *args):
        await self.disconnect()

    async def connect(self) -> None:
        """연결 초기화 (서브클래스에서 override 가능)"""
        self._is_connected = True

    async def disconnect(self) -> None:
        """연결 해제"""
        self._is_connected = False


# ─────────────────────────────────────────────
# 스토리지 인터페이스
# ─────────────────────────────────────────────
class BaseStorage(abc.ABC):
    """
    데이터 저장소 추상 인터페이스.
    TimescaleDB, ClickHouse, DuckDB 모두 이 인터페이스를 구현.
    """

    @abc.abstractmethod
    async def save_ohlcv(self, frames: list[OHLCVFrame]) -> int:
        """OHLCV 배치 저장. 저장된 레코드 수 반환."""
        ...

    @abc.abstractmethod
    async def load_ohlcv(
        self,
        symbol: Symbol,
        timeframe: Timeframe,
        start: datetime,
        end: Optional[datetime] = None,
    ) -> OHLCVFrame:
        """OHLCV 로드"""
        ...

    @abc.abstractmethod
    async def save_ticks(self, ticks: list[Tick]) -> int:
        """틱 데이터 배치 저장"""
        ...

    @abc.abstractmethod
    async def get_latest_timestamp(
        self, symbol: Symbol, timeframe: Timeframe
    ) -> Optional[datetime]:
        """특정 심볼의 최신 데이터 타임스탬프"""
        ...

    @abc.abstractmethod
    async def health_check(self) -> bool:
        """스토리지 헬스 체크"""
        ...


# ─────────────────────────────────────────────
# 전략 인터페이스
# ─────────────────────────────────────────────
class BaseStrategy(abc.ABC):
    """
    모든 트레이딩 전략의 추상 기반 클래스.

    Plugin Architecture:
    - 각 전략은 독립적으로 실행 가능
    - Ensemble에 포함 가능
    - 백테스트와 실시간 실행 동일 인터페이스

    SOLID 원칙:
    - SRP: 각 전략은 하나의 알파 소스만 담당
    - OCP: 새 전략 추가 시 기존 코드 수정 불필요
    - LSP: 모든 구현체는 이 인터페이스로 대체 가능
    """

    def __init__(self, strategy_id: str, name: str, config: dict[str, Any] = None) -> None:
        self.strategy_id = strategy_id
        self.name = name
        self.config = config or {}
        self._is_initialized: bool = False

    @abc.abstractmethod
    def generate_signal(
        self,
        data: OHLCVFrame,
        current_positions: dict[Symbol, Position] = None,
    ) -> list[Signal]:
        """
        가격 데이터와 현재 포지션을 받아 시그널 생성.

        Returns:
            시그널 리스트 (비어있으면 아무 행동 안 함)
        """
        ...

    @abc.abstractmethod
    def calculate_position_size(
        self,
        signal: Signal,
        portfolio_value: float,
        current_price: float,
        risk_per_trade: float = 0.01,
    ) -> float:
        """
        포지션 크기 계산.

        Args:
            signal: 매매 시그널
            portfolio_value: 현재 포트폴리오 가치
            current_price: 현재 가격
            risk_per_trade: 거래당 리스크 비율 (0.01 = 1%)

        Returns:
            매수할 수량
        """
        ...

    def get_stop_loss(
        self,
        signal: Signal,
        entry_price: float,
        data: OHLCVFrame,
    ) -> Optional[float]:
        """손절 가격 계산 (기본값: None = 손절 없음)"""
        return None

    def get_take_profit(
        self,
        signal: Signal,
        entry_price: float,
        data: OHLCVFrame,
    ) -> Optional[float]:
        """익절 가격 계산 (기본값: None = 익절 없음)"""
        return None

    def on_order_filled(self, order: Order) -> None:
        """주문 체결 이벤트 핸들러"""
        pass

    def on_position_closed(self, position: Position) -> None:
        """포지션 종료 이벤트 핸들러"""
        pass

    @abc.abstractmethod
    def validate_config(self) -> bool:
        """설정값 유효성 검증"""
        ...

    @property
    def metadata(self) -> dict[str, Any]:
        """전략 메타데이터"""
        return {
            "strategy_id": self.strategy_id,
            "name": self.name,
            "config": self.config,
        }


# ─────────────────────────────────────────────
# 백테스팅 엔진 인터페이스
# ─────────────────────────────────────────────
class BaseBacktestEngine(abc.ABC):
    """
    백테스팅 엔진 추상 인터페이스.
    벡터화 엔진과 이벤트 드리븐 엔진 모두 구현.
    """

    @abc.abstractmethod
    def run(
        self,
        strategy: BaseStrategy,
        data: OHLCVFrame,
        initial_capital: float = 100_000_000.0,
        commission_bps: float = 5.0,
        slippage_bps: float = 2.0,
    ) -> BacktestMetrics:
        """단일 전략 백테스트 실행"""
        ...

    @abc.abstractmethod
    def run_multi_asset(
        self,
        strategy: BaseStrategy,
        data: dict[Symbol, OHLCVFrame],
        initial_capital: float = 100_000_000.0,
    ) -> BacktestMetrics:
        """멀티 자산 백테스트"""
        ...

    def run_walk_forward(
        self,
        strategy: BaseStrategy,
        data: OHLCVFrame,
        n_splits: int = 5,
        train_ratio: float = 0.7,
    ) -> list[BacktestMetrics]:
        """Walk-Forward Analysis"""
        raise NotImplementedError("Walk-forward not implemented for this engine")

    def run_monte_carlo(
        self,
        returns: pd.Series,
        n_simulations: int = 10_000,
        initial_capital: float = 100_000_000.0,
    ) -> dict[str, Any]:
        """Monte Carlo Simulation"""
        raise NotImplementedError("Monte Carlo not implemented for this engine")


# ─────────────────────────────────────────────
# ML 모델 인터페이스
# ─────────────────────────────────────────────
class BaseMLModel(abc.ABC):
    """
    모든 ML/AI 예측 모델의 추상 기반 클래스.
    scikit-learn 스타일 인터페이스 채택.
    """

    def __init__(self, model_id: str, name: str) -> None:
        self.model_id = model_id
        self.name = name
        self._is_fitted: bool = False
        self._feature_names: list[str] = []

    @abc.abstractmethod
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        eval_set: Optional[tuple[pd.DataFrame, pd.Series]] = None,
    ) -> "BaseMLModel":
        """모델 학습"""
        ...

    @abc.abstractmethod
    def predict(self, X: pd.DataFrame) -> pd.Series:
        """예측 (분류: 클래스, 회귀: 수치)"""
        ...

    @abc.abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        """확률 예측 (분류 모델용)"""
        ...

    @abc.abstractmethod
    def get_feature_importance(self) -> pd.Series:
        """피처 중요도"""
        ...

    def save(self, path: str) -> None:
        """모델 직렬화 저장"""
        import joblib
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> "BaseMLModel":
        """모델 로드"""
        import joblib
        return joblib.load(path)

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted


# ─────────────────────────────────────────────
# 리스크 관리자 인터페이스
# ─────────────────────────────────────────────
class BaseRiskManager(abc.ABC):
    """
    리스크 관리 시스템 인터페이스.
    모든 주문은 리스크 관리자를 통과해야 한다.
    """

    @abc.abstractmethod
    def check_order(
        self,
        order: Order,
        portfolio: dict[Symbol, Position],
        portfolio_value: float,
    ) -> tuple[bool, str]:
        """
        주문 승인/거부 결정.

        Returns:
            (approved: bool, reason: str)
        """
        ...

    @abc.abstractmethod
    def calculate_var(
        self,
        returns: pd.Series,
        confidence: float = 0.95,
    ) -> float:
        """Value at Risk 계산"""
        ...

    @abc.abstractmethod
    def calculate_portfolio_var(
        self,
        positions: dict[Symbol, Position],
        returns_matrix: pd.DataFrame,
        confidence: float = 0.95,
    ) -> float:
        """포트폴리오 VaR"""
        ...

    @abc.abstractmethod
    def check_drawdown_limit(
        self,
        current_value: float,
        peak_value: float,
        limit: float = 0.15,
    ) -> bool:
        """낙폭 한도 초과 여부"""
        ...


# ─────────────────────────────────────────────
# 브로커 인터페이스 (실거래 연동)
# ─────────────────────────────────────────────
class BaseBroker(abc.ABC):
    """
    실거래/Paper Trading 브로커 추상 인터페이스.
    실거래와 백테스트가 동일한 인터페이스 사용.
    """

    @abc.abstractmethod
    async def submit_order(self, order: Order) -> Order:
        """주문 제출"""
        ...

    @abc.abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """주문 취소"""
        ...

    @abc.abstractmethod
    async def get_positions(self) -> dict[Symbol, Position]:
        """현재 포지션 조회"""
        ...

    @abc.abstractmethod
    async def get_balance(self) -> dict[str, float]:
        """잔고 조회"""
        ...

    @abc.abstractmethod
    async def get_order_status(self, order_id: str) -> Order:
        """주문 상태 조회"""
        ...


# ─────────────────────────────────────────────
# Feature Engineer 인터페이스
# ─────────────────────────────────────────────
class BaseFeatureEngineer(abc.ABC):
    """
    피처 엔지니어링 파이프라인 인터페이스.
    데이터 누수 방지를 위한 strict time-aware 설계.
    """

    @abc.abstractmethod
    def fit(self, data: OHLCVFrame) -> "BaseFeatureEngineer":
        """학습 데이터로 피처 통계 계산 (스케일러 등)"""
        ...

    @abc.abstractmethod
    def transform(self, data: OHLCVFrame) -> pd.DataFrame:
        """피처 행렬 생성"""
        ...

    def fit_transform(self, data: OHLCVFrame) -> pd.DataFrame:
        return self.fit(data).transform(data)

    @abc.abstractmethod
    def get_feature_names(self) -> list[str]:
        """피처 이름 목록"""
        ...


# ─────────────────────────────────────────────
# 포트폴리오 최적화 인터페이스
# ─────────────────────────────────────────────
class BasePortfolioOptimizer(abc.ABC):
    """포트폴리오 최적화 추상 인터페이스"""

    @abc.abstractmethod
    def optimize(
        self,
        expected_returns: pd.Series,
        covariance_matrix: pd.DataFrame,
        constraints: dict[str, Any] = None,
    ) -> pd.Series:
        """
        최적 포트폴리오 비중 계산.

        Returns:
            각 자산별 비중 (합계 = 1.0)
        """
        ...

    @abc.abstractmethod
    def efficient_frontier(
        self,
        expected_returns: pd.Series,
        covariance_matrix: pd.DataFrame,
        n_points: int = 100,
    ) -> pd.DataFrame:
        """효율적 프론티어 계산"""
        ...
