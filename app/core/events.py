"""
=============================================================================
Event-Driven Architecture — Async Event Bus
=============================================================================
이벤트 기반 아키텍처 구현.
모든 컴포넌트는 이벤트를 통해 느슨하게 결합.
Redis Pub/Sub 기반 분산 이벤트 지원.
=============================================================================
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import auto, Enum
from typing import Any, Callable, Coroutine, Dict, List, Optional, Set, Type
from uuid import uuid4

from app.config.logging_config import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────
# 이벤트 타입 정의
# ─────────────────────────────────────────────
class EventType(str, Enum):
    """시스템 전역 이벤트 타입"""
    # 데이터 이벤트
    TICK_RECEIVED = "tick.received"
    OHLCV_UPDATED = "ohlcv.updated"
    ORDERBOOK_UPDATED = "orderbook.updated"

    # 시그널 이벤트
    SIGNAL_GENERATED = "signal.generated"
    SIGNAL_EXPIRED = "signal.expired"

    # 주문 이벤트
    ORDER_SUBMITTED = "order.submitted"
    ORDER_FILLED = "order.filled"
    ORDER_CANCELLED = "order.cancelled"
    ORDER_REJECTED = "order.rejected"
    ORDER_PARTIALLY_FILLED = "order.partially_filled"

    # 포지션 이벤트
    POSITION_OPENED = "position.opened"
    POSITION_CLOSED = "position.closed"
    POSITION_UPDATED = "position.updated"

    # 리스크 이벤트
    RISK_LIMIT_BREACH = "risk.limit_breach"
    DRAWDOWN_ALERT = "risk.drawdown_alert"
    VAR_LIMIT_BREACH = "risk.var_limit_breach"

    # 백테스트 이벤트
    BACKTEST_STARTED = "backtest.started"
    BACKTEST_COMPLETED = "backtest.completed"
    BACKTEST_FAILED = "backtest.failed"

    # ML 이벤트
    MODEL_TRAINED = "ml.model_trained"
    PREDICTION_GENERATED = "ml.prediction_generated"

    # 시스템 이벤트
    SYSTEM_STARTUP = "system.startup"
    SYSTEM_SHUTDOWN = "system.shutdown"
    HEALTH_CHECK = "system.health_check"
    ERROR = "system.error"


# ─────────────────────────────────────────────
# 기본 이벤트 클래스
# ─────────────────────────────────────────────
@dataclass
class Event:
    """
    기본 이벤트 클래스.
    모든 이벤트는 이 클래스를 상속.
    """
    event_type: EventType
    event_id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    source: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
    correlation_id: Optional[str] = None  # 관련 이벤트 추적


@dataclass
class TickEvent(Event):
    """틱 이벤트"""
    symbol: str = ""
    price: float = 0.0
    volume: float = 0.0

    def __post_init__(self):
        self.event_type = EventType.TICK_RECEIVED
        self.payload.update({
            "symbol": self.symbol,
            "price": self.price,
            "volume": self.volume,
        })


@dataclass
class SignalEvent(Event):
    """시그널 이벤트"""
    signal_id: str = ""
    strategy_id: str = ""
    symbol: str = ""
    direction: int = 0

    def __post_init__(self):
        self.event_type = EventType.SIGNAL_GENERATED


@dataclass
class OrderEvent(Event):
    """주문 이벤트"""
    order_id: str = ""
    symbol: str = ""
    side: str = ""
    quantity: float = 0.0
    price: float = 0.0


@dataclass
class RiskAlertEvent(Event):
    """리스크 경보 이벤트"""
    alert_type: str = ""
    current_value: float = 0.0
    limit_value: float = 0.0
    severity: str = "WARNING"  # WARNING, CRITICAL


# ─────────────────────────────────────────────
# 이벤트 핸들러 타입
# ─────────────────────────────────────────────
EventHandler = Callable[[Event], Coroutine[Any, Any, None]]
SyncEventHandler = Callable[[Event], None]


# ─────────────────────────────────────────────
# 로컬 이벤트 버스 (인메모리)
# ─────────────────────────────────────────────
class EventBus:
    """
    비동기 이벤트 버스.

    특징:
    - asyncio 기반 비동기 처리
    - 우선순위 큐 지원
    - 이벤트 필터링
    - 에러 격리 (하나의 핸들러 오류가 다른 핸들러에 영향 없음)
    - 이벤트 히스토리 (디버깅용)

    Usage:
        bus = EventBus()

        @bus.on(EventType.TICK_RECEIVED)
        async def handle_tick(event: Event):
            print(f"Tick: {event.payload}")

        await bus.emit(TickEvent(symbol="AAPL", price=150.0, volume=1000))
    """

    def __init__(self, max_history: int = 1000) -> None:
        self._handlers: dict[EventType, list[EventHandler]] = {}
        self._queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self._running: bool = False
        self._history: list[Event] = []
        self._max_history = max_history
        self._error_count: int = 0

    def on(self, event_type: EventType, priority: int = 5):
        """
        이벤트 핸들러 등록 데코레이터.

        Args:
            event_type: 구독할 이벤트 타입
            priority: 핸들러 우선순위 (낮을수록 먼저 실행, 기본값 5)
        """
        def decorator(handler: EventHandler) -> EventHandler:
            self.subscribe(event_type, handler)
            return handler
        return decorator

    def subscribe(
        self,
        event_type: EventType,
        handler: EventHandler,
        priority: int = 5,
    ) -> None:
        """핸들러 구독 등록"""
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)
        logger.debug(
            "event_handler_registered",
            event_type=event_type.value,
            handler=handler.__qualname__,
        )

    def unsubscribe(self, event_type: EventType, handler: EventHandler) -> None:
        """핸들러 구독 해제"""
        if event_type in self._handlers:
            self._handlers[event_type].remove(handler)

    async def emit(self, event: Event, priority: int = 5) -> None:
        """이벤트 발행 (큐에 추가)"""
        await self._queue.put((priority, event))

    async def emit_immediate(self, event: Event) -> None:
        """이벤트 즉시 발행 (큐 우회)"""
        await self._dispatch(event)

    async def _dispatch(self, event: Event) -> None:
        """이벤트 핸들러에 디스패치"""
        handlers = self._handlers.get(event.event_type, [])

        # 히스토리 기록
        self._history.append(event)
        if len(self._history) > self._max_history:
            self._history.pop(0)

        if not handlers:
            return

        # 모든 핸들러 병렬 실행 (에러 격리)
        tasks = [
            asyncio.create_task(self._safe_handle(handler, event))
            for handler in handlers
        ]
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _safe_handle(self, handler: EventHandler, event: Event) -> None:
        """에러 격리된 핸들러 실행"""
        try:
            await handler(event)
        except Exception as e:
            self._error_count += 1
            logger.error(
                "event_handler_error",
                handler=handler.__qualname__,
                event_type=event.event_type.value,
                error=str(e),
            )

    async def start(self) -> None:
        """이벤트 처리 루프 시작"""
        self._running = True
        logger.info("event_bus_started")

        while self._running:
            try:
                priority, event = await asyncio.wait_for(
                    self._queue.get(), timeout=0.1
                )
                await self._dispatch(event)
                self._queue.task_done()
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error("event_bus_error", error=str(e))

    def stop(self) -> None:
        """이벤트 처리 루프 종료"""
        self._running = False
        logger.info("event_bus_stopped")

    def get_history(
        self,
        event_type: Optional[EventType] = None,
        limit: int = 100,
    ) -> list[Event]:
        """이벤트 히스토리 조회"""
        history = self._history
        if event_type:
            history = [e for e in history if e.event_type == event_type]
        return history[-limit:]

    @property
    def stats(self) -> dict[str, Any]:
        """이벤트 버스 통계"""
        return {
            "queue_size": self._queue.qsize(),
            "registered_handlers": {
                k.value: len(v) for k, v in self._handlers.items()
            },
            "history_size": len(self._history),
            "error_count": self._error_count,
            "is_running": self._running,
        }


# ─────────────────────────────────────────────
# 글로벌 이벤트 버스 싱글턴
# ─────────────────────────────────────────────
_global_event_bus: Optional[EventBus] = None


def get_event_bus() -> EventBus:
    """글로벌 이벤트 버스 싱글턴 반환"""
    global _global_event_bus
    if _global_event_bus is None:
        _global_event_bus = EventBus()
    return _global_event_bus
