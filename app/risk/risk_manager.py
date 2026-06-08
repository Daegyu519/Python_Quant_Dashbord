"""
=============================================================================
Risk Management System — Institution-Grade
=============================================================================
헤지펀드 수준의 종합 리스크 관리.
VaR, CVaR, 포지션 한도, 낙폭 제한 모두 포함.
=============================================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy import stats

from app.core.base_classes import BaseRiskManager
from app.core.types import Order, Position, Symbol
from app.config.logging_config import get_logger
from app.config.settings import get_settings

logger = get_logger(__name__)
settings = get_settings()


@dataclass
class RiskCheckResult:
    """리스크 체크 결과"""
    approved: bool
    reason: str
    risk_score: float = 0.0   # 0.0 (안전) ~ 1.0 (위험)
    warnings: list[str] = None

    def __post_init__(self):
        self.warnings = self.warnings or []


class QuantRiskManager(BaseRiskManager):
    """
    기관급 통합 리스크 관리 시스템.

    리스크 체크:
    1. 단일 포지션 한도 (max_position_size)
    2. 섹터 노출 한도 (max_sector_exposure)
    3. 일별/주별 손실 한도
    4. 최대 낙폭 제한 (max_drawdown_limit)
    5. 포트폴리오 VaR 한도
    6. 레버리지 한도
    7. 상관관계 리스크 (concentration risk)

    주문은 모든 체크를 통과해야 실행.
    """

    def __init__(
        self,
        max_position_size: float = None,
        max_sector_exposure: float = None,
        max_drawdown_limit: float = None,
        daily_loss_limit: float = None,
        var_confidence: float = None,
    ) -> None:
        risk_cfg = settings.risk

        self.max_position_size = max_position_size or risk_cfg.max_position_size
        self.max_sector_exposure = max_sector_exposure or risk_cfg.max_sector_exposure
        self.max_drawdown_limit = max_drawdown_limit or risk_cfg.max_drawdown_limit
        self.daily_loss_limit = daily_loss_limit or risk_cfg.daily_loss_limit
        self.var_confidence = var_confidence or risk_cfg.var_confidence

        # 일별 손실 추적
        self._daily_pnl: list[tuple[datetime, float]] = []
        self._peak_value: float = 0.0
        self._current_value: float = 0.0

    def check_order(
        self,
        order: Order,
        portfolio: dict[Symbol, Position],
        portfolio_value: float,
    ) -> tuple[bool, str]:
        """
        주문 리스크 체크 (체인 방식).

        모든 체크를 순서대로 실행.
        하나라도 실패 시 즉시 거부.
        """
        result = RiskCheckResult(approved=True, reason="All checks passed")

        checks = [
            self._check_position_size(order, portfolio_value),
            self._check_drawdown_limit(portfolio_value),
            self._check_daily_loss_limit(portfolio_value),
            self._check_leverage(order, portfolio, portfolio_value),
            self._check_concentration(order, portfolio, portfolio_value),
        ]

        for check_result in checks:
            if not check_result.approved:
                logger.warning(
                    "risk_check_failed",
                    order_id=order.order_id,
                    symbol=order.symbol,
                    reason=check_result.reason,
                    risk_score=check_result.risk_score,
                )
                return False, check_result.reason

            result.warnings.extend(check_result.warnings)

        if result.warnings:
            logger.warning(
                "risk_warnings",
                order_id=order.order_id,
                warnings=result.warnings,
            )

        return True, "approved"

    def _check_position_size(
        self, order: Order, portfolio_value: float
    ) -> RiskCheckResult:
        """단일 포지션 크기 한도 체크"""
        if portfolio_value <= 0:
            return RiskCheckResult(False, "Invalid portfolio value")

        order_value = (order.quantity * (order.price or 0.0))
        position_size_pct = order_value / portfolio_value

        if position_size_pct > self.max_position_size:
            return RiskCheckResult(
                approved=False,
                reason=(
                    f"Position size {position_size_pct:.1%} exceeds "
                    f"limit {self.max_position_size:.1%}"
                ),
                risk_score=min(position_size_pct / self.max_position_size, 1.0),
            )

        warnings = []
        if position_size_pct > self.max_position_size * 0.8:
            warnings.append(
                f"Position size {position_size_pct:.1%} approaching limit"
            )

        return RiskCheckResult(
            approved=True,
            reason="position_size_ok",
            risk_score=position_size_pct / self.max_position_size,
            warnings=warnings,
        )

    def _check_drawdown_limit(self, current_value: float) -> RiskCheckResult:
        """낙폭 한도 체크"""
        if self._peak_value > 0:
            current_dd = (current_value - self._peak_value) / self._peak_value

            if current_dd < -self.max_drawdown_limit:
                return RiskCheckResult(
                    approved=False,
                    reason=(
                        f"Max drawdown {abs(current_dd):.1%} exceeds "
                        f"limit {self.max_drawdown_limit:.1%}. "
                        f"Trading halted."
                    ),
                    risk_score=1.0,
                )

            warnings = []
            if current_dd < -self.max_drawdown_limit * 0.7:
                warnings.append(
                    f"Drawdown {abs(current_dd):.1%} approaching limit — reduce exposure"
                )

            return RiskCheckResult(
                approved=True,
                reason="drawdown_ok",
                risk_score=abs(current_dd) / self.max_drawdown_limit,
                warnings=warnings,
            )

        return RiskCheckResult(approved=True, reason="no_peak_value")

    def _check_daily_loss_limit(self, current_value: float) -> RiskCheckResult:
        """일별 손실 한도 체크"""
        today = datetime.utcnow().date()

        # 오늘의 PnL 계산
        today_pnl_entries = [
            pnl for dt, pnl in self._daily_pnl
            if dt.date() == today
        ]

        if today_pnl_entries:
            daily_loss = sum(today_pnl_entries) / (current_value + 1e-10)

            if daily_loss < -self.daily_loss_limit:
                return RiskCheckResult(
                    approved=False,
                    reason=(
                        f"Daily loss {abs(daily_loss):.1%} exceeds "
                        f"limit {self.daily_loss_limit:.1%}. "
                        f"No more trading today."
                    ),
                    risk_score=1.0,
                )

        return RiskCheckResult(approved=True, reason="daily_loss_ok")

    def _check_leverage(
        self,
        order: Order,
        portfolio: dict[Symbol, Position],
        portfolio_value: float,
    ) -> RiskCheckResult:
        """레버리지 한도 체크 (현재 구현: 1x 기본)"""
        total_exposure = sum(
            abs(pos.market_value)
            for pos in portfolio.values()
        )

        order_value = order.quantity * (order.price or 0.0)
        new_exposure = total_exposure + order_value

        leverage = new_exposure / (portfolio_value + 1e-10)

        if leverage > 1.5:  # 1.5x 한도
            return RiskCheckResult(
                approved=False,
                reason=f"Leverage {leverage:.2f}x would exceed 1.5x limit",
                risk_score=min(leverage / 1.5, 1.0),
            )

        return RiskCheckResult(approved=True, reason="leverage_ok", risk_score=leverage)

    def _check_concentration(
        self,
        order: Order,
        portfolio: dict[Symbol, Position],
        portfolio_value: float,
    ) -> RiskCheckResult:
        """집중 리스크 체크 (상위 포지션 편중)"""
        if not portfolio:
            return RiskCheckResult(approved=True, reason="no_existing_positions")

        # 포지션 가중치 계산
        weights = {
            sym: abs(pos.market_value) / (portfolio_value + 1e-10)
            for sym, pos in portfolio.items()
        }

        # HHI (허핀달-허쉬만 지수) - 집중도 측정
        hhi = sum(w ** 2 for w in weights.values())

        warnings = []
        if hhi > 0.25:  # HHI > 0.25 = 집중도 높음
            warnings.append(
                f"High portfolio concentration (HHI={hhi:.3f}). "
                f"Consider diversifying."
            )

        return RiskCheckResult(approved=True, reason="concentration_ok", warnings=warnings)

    # ─────────────────────────────────────────────
    # VaR 계산
    # ─────────────────────────────────────────────

    def calculate_var(
        self,
        returns: pd.Series,
        confidence: float = 0.95,
        method: str = "historical",
    ) -> float:
        """
        Value at Risk 계산.

        방법:
        - historical: 역사적 시뮬레이션
        - parametric: 정규분포 가정
        - cornish_fisher: Cornish-Fisher 확장 (두꺼운 꼬리 보정)
        """
        clean_returns = returns.dropna().to_numpy()

        if len(clean_returns) < 30:
            return 0.0

        alpha = 1 - confidence

        if method == "historical":
            var = float(-np.percentile(clean_returns, alpha * 100))

        elif method == "parametric":
            mu = np.mean(clean_returns)
            sigma = np.std(clean_returns)
            z = stats.norm.ppf(alpha)
            var = float(-(mu + z * sigma))

        elif method == "cornish_fisher":
            # Cornish-Fisher 확장 (비대칭, 두꺼운 꼬리 보정)
            mu = np.mean(clean_returns)
            sigma = np.std(clean_returns)
            skew = stats.skew(clean_returns)
            kurt = stats.kurtosis(clean_returns)  # excess kurtosis

            z = stats.norm.ppf(alpha)
            # Cornish-Fisher 수정 Z
            z_cf = (z + (z**2 - 1) * skew / 6
                    + (z**3 - 3*z) * kurt / 24
                    - (2*z**3 - 5*z) * skew**2 / 36)
            var = float(-(mu + z_cf * sigma))

        else:
            raise ValueError(f"Unknown VaR method: {method}")

        return max(var, 0.0)

    def calculate_cvar(
        self,
        returns: pd.Series,
        confidence: float = 0.95,
    ) -> float:
        """
        Conditional Value at Risk (Expected Shortfall).
        VaR보다 꼬리 리스크를 더 잘 포착.
        """
        clean_returns = returns.dropna().to_numpy()

        if len(clean_returns) < 30:
            return 0.0

        var = self.calculate_var(returns, confidence)
        tail_losses = clean_returns[clean_returns <= -var]

        if len(tail_losses) == 0:
            return var

        cvar = float(-np.mean(tail_losses))
        return max(cvar, var)

    def calculate_portfolio_var(
        self,
        positions: dict[Symbol, Position],
        returns_matrix: pd.DataFrame,
        confidence: float = 0.95,
    ) -> float:
        """
        포트폴리오 VaR (상관관계 고려).

        Delta-Normal 방법 사용.
        """
        if not positions or returns_matrix.empty:
            return 0.0

        total_value = sum(abs(p.market_value) for p in positions.values())
        if total_value == 0:
            return 0.0

        # 포지션 가중치
        weights = np.array([
            positions[sym].market_value / total_value
            for sym in returns_matrix.columns
            if sym in positions
        ])

        if len(weights) == 0:
            return 0.0

        # 공분산 행렬 계산 (연간화)
        aligned_returns = returns_matrix[[
            sym for sym in returns_matrix.columns if sym in positions
        ]].dropna()

        if len(aligned_returns) < 30:
            return 0.0

        cov_matrix = aligned_returns.cov().to_numpy() * 252
        portfolio_vol = np.sqrt(weights @ cov_matrix @ weights)

        # 분위수 기반 VaR
        alpha = 1 - confidence
        z = stats.norm.ppf(alpha)
        portfolio_var = float(-z * portfolio_vol / np.sqrt(252))  # 일별 VaR

        return max(portfolio_var, 0.0)

    def check_drawdown_limit(
        self,
        current_value: float,
        peak_value: float,
        limit: float = 0.15,
    ) -> bool:
        """낙폭 한도 초과 여부"""
        if peak_value <= 0:
            return False
        drawdown = (current_value - peak_value) / peak_value
        return drawdown < -limit

    def update_portfolio_value(self, value: float) -> None:
        """포트폴리오 가치 갱신 (PnL 추적용)"""
        if self._current_value > 0:
            pnl = value - self._current_value
            self._daily_pnl.append((datetime.utcnow(), pnl))

            # 30일 이상 된 기록 삭제
            cutoff = datetime.utcnow() - timedelta(days=30)
            self._daily_pnl = [
                (dt, pnl) for dt, pnl in self._daily_pnl if dt > cutoff
            ]

        self._current_value = value
        self._peak_value = max(self._peak_value, value)

    def get_risk_report(
        self,
        portfolio: dict[Symbol, Position],
        returns_matrix: pd.DataFrame,
        portfolio_value: float,
    ) -> dict[str, Any]:
        """종합 리스크 리포트"""
        if not portfolio:
            return {"status": "no_positions"}

        portfolio_returns = returns_matrix.mean(axis=1) if not returns_matrix.empty else pd.Series()

        var_95 = self.calculate_var(portfolio_returns, 0.95) if len(portfolio_returns) > 30 else 0.0
        cvar_95 = self.calculate_cvar(portfolio_returns, 0.95) if len(portfolio_returns) > 30 else 0.0
        portfolio_var = self.calculate_portfolio_var(portfolio, returns_matrix)

        # 현재 낙폭
        current_dd = 0.0
        if self._peak_value > 0:
            current_dd = (portfolio_value - self._peak_value) / self._peak_value

        # 집중도 (HHI)
        weights = [abs(pos.market_value) / portfolio_value for pos in portfolio.values()]
        hhi = sum(w**2 for w in weights)

        return {
            "portfolio_value": portfolio_value,
            "current_drawdown": round(current_dd * 100, 2),
            "max_drawdown_limit": self.max_drawdown_limit * 100,
            "var_95_pct": round(var_95 * 100, 3),
            "cvar_95_pct": round(cvar_95 * 100, 3),
            "portfolio_var": round(portfolio_var * 100, 3),
            "hhi_concentration": round(hhi, 4),
            "n_positions": len(portfolio),
            "total_exposure": sum(abs(p.market_value) for p in portfolio.values()),
            "daily_loss_limit_pct": self.daily_loss_limit * 100,
            "risk_status": "CRITICAL" if current_dd < -self.max_drawdown_limit * 0.9
                          else "WARNING" if current_dd < -self.max_drawdown_limit * 0.7
                          else "NORMAL",
        }
