"""
=============================================================================
SQLAlchemy ORM Models — MySQL 호환 스키마
=============================================================================
MySQL 9.x 기준으로 재작성.
- UUID → CHAR(36)
- JSONB → JSON
- ARRAY → JSON (MySQL은 배열 타입 미지원)
- DateTime(timezone=True) → DateTime
=============================================================================
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    JSON, BigInteger, Boolean, Column, DateTime, Float,
    ForeignKey, Index, Integer, Numeric, String, Text,
    UniqueConstraint, func,
)
from sqlalchemy.ext.asyncio import AsyncAttrs
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


# ─────────────────────────────────────────────
# Base 클래스
# ─────────────────────────────────────────────
class Base(AsyncAttrs, DeclarativeBase):
    """SQLAlchemy Base (async 지원)"""
    pass


class TimestampMixin:
    """생성/수정 타임스탬프 자동 관리"""
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


# ─────────────────────────────────────────────
# 사용자 관리
# ─────────────────────────────────────────────
class User(Base, TimestampMixin):
    """사용자 테이블"""
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    username: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    is_superuser: Mapped[bool] = mapped_column(Boolean, default=False)
    api_key: Mapped[Optional[str]] = mapped_column(String(64), unique=True, index=True)

    strategies: Mapped[list["Strategy"]] = relationship(back_populates="user")
    portfolios: Mapped[list["Portfolio"]] = relationship(back_populates="user")


# ─────────────────────────────────────────────
# OHLCV 주가 데이터
# ─────────────────────────────────────────────
class OHLCVData(Base):
    """
    주가 OHLCV 데이터 테이블.
    symbol + timeframe + time 복합 인덱스로 빠른 조회.
    """
    __tablename__ = "ohlcv_data"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    time: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    symbol: Mapped[str] = mapped_column(String(30), nullable=False)
    timeframe: Mapped[str] = mapped_column(String(5), nullable=False)  # 1m, 5m, 1h, 1d 등

    open: Mapped[float] = mapped_column(Numeric(20, 8), nullable=False)
    high: Mapped[float] = mapped_column(Numeric(20, 8), nullable=False)
    low: Mapped[float] = mapped_column(Numeric(20, 8), nullable=False)
    close: Mapped[float] = mapped_column(Numeric(20, 8), nullable=False)
    volume: Mapped[float] = mapped_column(Numeric(30, 8), nullable=False, default=0)

    source: Mapped[str] = mapped_column(String(20), nullable=False, default="yahoo")
    # yahoo, krx, binance, coingecko

    __table_args__ = (
        UniqueConstraint("symbol", "timeframe", "time", name="uq_ohlcv_symbol_tf_time"),
        Index("idx_ohlcv_symbol_time", "symbol", "time"),
        Index("idx_ohlcv_symbol_tf_time", "symbol", "timeframe", "time"),
    )


# ─────────────────────────────────────────────
# 전략 관리
# ─────────────────────────────────────────────
class Strategy(Base, TimestampMixin):
    """트레이딩 전략 메타데이터"""
    __tablename__ = "strategies"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    strategy_id: Mapped[str] = mapped_column(String(100), unique=True, nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)
    strategy_type: Mapped[str] = mapped_column(String(50), nullable=False)
    # technical, ml, ensemble

    # JSON으로 설정 저장 (MySQL 5.7.8+ 지원)
    config: Mapped[Optional[dict]] = mapped_column(JSON, default=dict)
    parameters: Mapped[Optional[dict]] = mapped_column(JSON, default=dict)
    universe: Mapped[Optional[list]] = mapped_column(JSON, default=list)
    # universe = ["AAPL", "005930", "BTCUSDT"] 형식

    is_active: Mapped[bool] = mapped_column(Boolean, default=False)
    is_paper_trading: Mapped[bool] = mapped_column(Boolean, default=True)

    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), nullable=False)
    user: Mapped["User"] = relationship(back_populates="strategies")

    last_sharpe: Mapped[Optional[float]] = mapped_column(Float)
    last_cagr: Mapped[Optional[float]] = mapped_column(Float)
    last_max_drawdown: Mapped[Optional[float]] = mapped_column(Float)

    __table_args__ = (
        Index("idx_strategy_type_active", "strategy_type", "is_active"),
    )


# ─────────────────────────────────────────────
# 포트폴리오
# ─────────────────────────────────────────────
class Portfolio(Base, TimestampMixin):
    """포트폴리오 테이블"""
    __tablename__ = "portfolios"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)

    initial_capital: Mapped[float] = mapped_column(Numeric(20, 2), nullable=False, default=100_000_000.0)
    current_value: Mapped[float] = mapped_column(Numeric(20, 2), default=0.0)
    cash: Mapped[float] = mapped_column(Numeric(20, 2), default=0.0)

    currency: Mapped[str] = mapped_column(String(10), default="KRW")
    risk_profile: Mapped[str] = mapped_column(String(50), default="moderate")
    rebalance_frequency: Mapped[str] = mapped_column(String(20), default="monthly")

    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), nullable=False)
    user: Mapped["User"] = relationship(back_populates="portfolios")

    positions: Mapped[list["PortfolioPosition"]] = relationship(back_populates="portfolio")
    allocations: Mapped[list["PortfolioAllocation"]] = relationship(back_populates="portfolio")


class PortfolioPosition(Base, TimestampMixin):
    """현재 포트폴리오 포지션"""
    __tablename__ = "portfolio_positions"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    portfolio_id: Mapped[str] = mapped_column(String(36), ForeignKey("portfolios.id"), nullable=False)
    symbol: Mapped[str] = mapped_column(String(30), nullable=False, index=True)
    side: Mapped[str] = mapped_column(String(10), nullable=False)  # LONG, SHORT

    quantity: Mapped[float] = mapped_column(Numeric(30, 8), nullable=False)
    entry_price: Mapped[float] = mapped_column(Numeric(20, 8), nullable=False)
    current_price: Mapped[float] = mapped_column(Numeric(20, 8), default=0.0)
    market_value: Mapped[float] = mapped_column(Numeric(20, 2), default=0.0)

    unrealized_pnl: Mapped[float] = mapped_column(Numeric(20, 2), default=0.0)
    realized_pnl: Mapped[float] = mapped_column(Numeric(20, 2), default=0.0)
    commission_paid: Mapped[float] = mapped_column(Numeric(20, 2), default=0.0)

    strategy_id: Mapped[Optional[str]] = mapped_column(String(100), index=True)
    portfolio: Mapped["Portfolio"] = relationship(back_populates="positions")

    __table_args__ = (
        UniqueConstraint("portfolio_id", "symbol", "side", name="uq_portfolio_position"),
        Index("idx_position_portfolio_symbol", "portfolio_id", "symbol"),
    )


class PortfolioAllocation(Base, TimestampMixin):
    """목표 포트폴리오 배분"""
    __tablename__ = "portfolio_allocations"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    portfolio_id: Mapped[str] = mapped_column(String(36), ForeignKey("portfolios.id"), nullable=False)
    symbol: Mapped[str] = mapped_column(String(30), nullable=False)
    target_weight: Mapped[float] = mapped_column(Float, nullable=False)
    actual_weight: Mapped[float] = mapped_column(Float, default=0.0)
    rebalance_date: Mapped[Optional[datetime]] = mapped_column(DateTime)

    portfolio: Mapped["Portfolio"] = relationship(back_populates="allocations")


# ─────────────────────────────────────────────
# 주문 & 거래 내역
# ─────────────────────────────────────────────
class OrderRecord(Base):
    """주문 기록"""
    __tablename__ = "orders"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    order_id: Mapped[str] = mapped_column(String(100), unique=True, nullable=False, index=True)
    portfolio_id: Mapped[str] = mapped_column(String(36), ForeignKey("portfolios.id"), nullable=False)
    strategy_id: Mapped[Optional[str]] = mapped_column(String(100), index=True)
    symbol: Mapped[str] = mapped_column(String(30), nullable=False, index=True)

    side: Mapped[str] = mapped_column(String(10), nullable=False)        # BUY, SELL
    order_type: Mapped[str] = mapped_column(String(20), nullable=False)  # MARKET, LIMIT
    status: Mapped[str] = mapped_column(String(30), nullable=False, index=True)

    quantity: Mapped[float] = mapped_column(Numeric(30, 8), nullable=False)
    price: Mapped[Optional[float]] = mapped_column(Numeric(20, 8))
    filled_quantity: Mapped[float] = mapped_column(Numeric(30, 8), default=0.0)
    filled_price: Mapped[float] = mapped_column(Numeric(20, 8), default=0.0)
    commission: Mapped[float] = mapped_column(Numeric(20, 8), default=0.0)

    submitted_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, index=True, server_default=func.now()
    )
    filled_at: Mapped[Optional[datetime]] = mapped_column(DateTime)

    metadata_json: Mapped[Optional[dict]] = mapped_column(JSON, default=dict)

    __table_args__ = (
        Index("idx_order_symbol_time", "symbol", "submitted_at"),
        Index("idx_order_strategy_time", "strategy_id", "submitted_at"),
    )


# ─────────────────────────────────────────────
# 백테스트 결과
# ─────────────────────────────────────────────
class BacktestResult(Base, TimestampMixin):
    """백테스트 결과"""
    __tablename__ = "backtest_results"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    strategy_id: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    run_id: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)

    start_date: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    end_date: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    timeframe: Mapped[str] = mapped_column(String(10), nullable=False)

    # 핵심 성과 지표
    total_return: Mapped[Optional[float]] = mapped_column(Float)
    cagr: Mapped[Optional[float]] = mapped_column(Float)
    sharpe_ratio: Mapped[Optional[float]] = mapped_column(Float, index=True)
    sortino_ratio: Mapped[Optional[float]] = mapped_column(Float)
    max_drawdown: Mapped[Optional[float]] = mapped_column(Float)
    win_rate: Mapped[Optional[float]] = mapped_column(Float)
    profit_factor: Mapped[Optional[float]] = mapped_column(Float)
    total_trades: Mapped[Optional[int]] = mapped_column(Integer)
    volatility: Mapped[Optional[float]] = mapped_column(Float)
    var_95: Mapped[Optional[float]] = mapped_column(Float)

    # 전체 지표 JSON
    full_metrics: Mapped[Optional[dict]] = mapped_column(JSON, default=dict)

    initial_capital: Mapped[Optional[float]] = mapped_column(Numeric(20, 2))
    commission_bps: Mapped[float] = mapped_column(Float, default=5.0)
    slippage_bps: Mapped[float] = mapped_column(Float, default=2.0)
    parameters: Mapped[Optional[dict]] = mapped_column(JSON, default=dict)

    status: Mapped[str] = mapped_column(String(20), default="completed")
    error_message: Mapped[Optional[str]] = mapped_column(Text)

    __table_args__ = (
        Index("idx_backtest_strategy_sharpe", "strategy_id", "sharpe_ratio"),
        Index("idx_backtest_cagr", "cagr"),
    )


# ─────────────────────────────────────────────
# 매매 시그널 기록
# ─────────────────────────────────────────────
class SignalRecord(Base):
    """생성된 매매 시그널 기록"""
    __tablename__ = "signals"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    signal_id: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    strategy_id: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    symbol: Mapped[str] = mapped_column(String(30), nullable=False, index=True)

    direction: Mapped[int] = mapped_column(Integer, nullable=False)  # 1=매수, -1=매도
    strength: Mapped[float] = mapped_column(Float, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    source: Mapped[str] = mapped_column(String(50), nullable=False)
    timeframe: Mapped[str] = mapped_column(String(10), nullable=False)

    target_price: Mapped[Optional[float]] = mapped_column(Float)
    stop_loss: Mapped[Optional[float]] = mapped_column(Float)
    take_profit: Mapped[Optional[float]] = mapped_column(Float)

    generated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, index=True, server_default=func.now()
    )
    expiry: Mapped[Optional[datetime]] = mapped_column(DateTime)
    signal_metadata: Mapped[Optional[dict]] = mapped_column(JSON, default=dict)

    __table_args__ = (
        Index("idx_signal_symbol_time", "symbol", "generated_at"),
        Index("idx_signal_strategy_time", "strategy_id", "generated_at"),
    )


# ─────────────────────────────────────────────
# ML 모델 레지스트리
# ─────────────────────────────────────────────
class MLModel(Base, TimestampMixin):
    """ML 모델 메타데이터 레지스트리"""
    __tablename__ = "ml_models"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    model_id: Mapped[str] = mapped_column(String(100), unique=True, nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    model_type: Mapped[str] = mapped_column(String(50), nullable=False)
    version: Mapped[str] = mapped_column(String(20), nullable=False)

    train_start: Mapped[Optional[datetime]] = mapped_column(DateTime)
    train_end: Mapped[Optional[datetime]] = mapped_column(DateTime)
    symbols: Mapped[Optional[list]] = mapped_column(JSON, default=list)
    timeframe: Mapped[Optional[str]] = mapped_column(String(10))

    val_accuracy: Mapped[Optional[float]] = mapped_column(Float)
    val_auc: Mapped[Optional[float]] = mapped_column(Float)
    val_sharpe: Mapped[Optional[float]] = mapped_column(Float)

    hyperparameters: Mapped[Optional[dict]] = mapped_column(JSON, default=dict)
    feature_names: Mapped[Optional[list]] = mapped_column(JSON, default=list)
    feature_importance: Mapped[Optional[dict]] = mapped_column(JSON, default=dict)

    model_path: Mapped[Optional[str]] = mapped_column(String(500))
    is_deployed: Mapped[bool] = mapped_column(Boolean, default=False)

    __table_args__ = (
        Index("idx_ml_model_type_deployed", "model_type", "is_deployed"),
    )


# ─────────────────────────────────────────────
# 알림 & 이벤트 로그
# ─────────────────────────────────────────────
class Alert(Base):
    """시스템 알림 및 이벤트 로그"""
    __tablename__ = "alerts"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    alert_type: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    severity: Mapped[str] = mapped_column(String(20), nullable=False)  # INFO, WARNING, CRITICAL
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    metadata_json: Mapped[Optional[dict]] = mapped_column(JSON, default=dict)
    is_resolved: Mapped[bool] = mapped_column(Boolean, default=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now(), index=True
    )
    resolved_at: Mapped[Optional[datetime]] = mapped_column(DateTime)

    __table_args__ = (
        Index("idx_alert_type_time", "alert_type", "created_at"),
        Index("idx_alert_unresolved", "is_resolved", "created_at"),
    )
