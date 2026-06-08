"""
=============================================================================
Portfolio Optimizer — Mean-Variance, Black-Litterman, Risk Parity
=============================================================================
기관급 포트폴리오 최적화 엔진.
효율적 프론티어, 리스크 패리티, 블랙-리터만 모델 구현.
=============================================================================
"""

from __future__ import annotations

from typing import Any, Optional
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize, LinearConstraint, Bounds

from app.core.base_classes import BasePortfolioOptimizer
from app.config.logging_config import get_logger

logger = get_logger(__name__)
warnings.filterwarnings("ignore", category=RuntimeWarning)


class MeanVarianceOptimizer(BasePortfolioOptimizer):
    """
    Markowitz 평균-분산 포트폴리오 최적화.

    목표:
    - Max Sharpe Ratio
    - Min Variance
    - Max Return (주어진 리스크 하에서)
    - Target Volatility
    """

    def __init__(
        self,
        risk_free_rate: float = 0.03,
        trading_days: int = 252,
        min_weight: float = 0.0,
        max_weight: float = 0.3,  # 단일 자산 최대 30%
    ) -> None:
        self.risk_free_rate = risk_free_rate
        self.trading_days = trading_days
        self.min_weight = min_weight
        self.max_weight = max_weight

    def optimize(
        self,
        expected_returns: pd.Series,
        covariance_matrix: pd.DataFrame,
        constraints: dict[str, Any] = None,
        objective: str = "max_sharpe",
    ) -> pd.Series:
        """
        최적 포트폴리오 비중 계산.

        Args:
            expected_returns: 연간화 기대 수익률 (종목별)
            covariance_matrix: 연간화 공분산 행렬
            constraints: 추가 제약 조건
            objective: 최적화 목표
                - max_sharpe: 최대 샤프 비율
                - min_variance: 최소 분산
                - risk_parity: 리스크 패리티

        Returns:
            종목별 최적 비중 (합계 = 1.0)
        """
        n = len(expected_returns)
        mu = expected_returns.to_numpy()
        Sigma = covariance_matrix.to_numpy()

        # 초기 가중치 (균등 배분)
        w0 = np.ones(n) / n

        # 제약 조건
        bounds = Bounds(
            lb=self.min_weight,
            ub=self.max_weight,
        )
        weight_sum_constraint = {
            "type": "eq",
            "fun": lambda w: np.sum(w) - 1.0,
        }
        constraint_list = [weight_sum_constraint]

        if constraints:
            for name, val in constraints.items():
                constraint_list.append(val)

        # ── 목적 함수 선택 ──
        if objective == "max_sharpe":
            def neg_sharpe(w):
                port_return = w @ mu
                port_vol = np.sqrt(w @ Sigma @ w)
                if port_vol < 1e-10:
                    return 0.0
                return -(port_return - self.risk_free_rate) / port_vol

            result = minimize(
                neg_sharpe, w0,
                method="SLSQP",
                bounds=bounds,
                constraints=constraint_list,
                options={"maxiter": 1000, "ftol": 1e-9},
            )

        elif objective == "min_variance":
            def portfolio_variance(w):
                return w @ Sigma @ w

            result = minimize(
                portfolio_variance, w0,
                method="SLSQP",
                bounds=bounds,
                constraints=constraint_list,
                options={"maxiter": 1000},
            )

        elif objective == "risk_parity":
            return self._risk_parity_weights(Sigma, expected_returns.index)

        else:
            raise ValueError(f"Unknown objective: {objective}")

        if not result.success:
            logger.warning(
                "optimization_failed",
                objective=objective,
                message=result.message,
            )
            # 최적화 실패 시 균등 배분 반환
            return pd.Series(w0, index=expected_returns.index)

        weights = pd.Series(result.x, index=expected_returns.index)

        # 수치 오류로 인한 음수 가중치 클리핑
        weights = weights.clip(lower=0.0)
        weights = weights / weights.sum()  # 재정규화

        logger.info(
            "portfolio_optimized",
            objective=objective,
            n_assets=n,
            sharpe=round(self._calc_sharpe(weights, mu, Sigma), 3),
        )

        return weights

    def _risk_parity_weights(
        self,
        Sigma: np.ndarray,
        symbols: pd.Index,
    ) -> pd.Series:
        """
        리스크 패리티 (Equal Risk Contribution).
        각 자산이 포트폴리오 리스크에 동등하게 기여하도록 배분.
        """
        n = len(symbols)
        w0 = np.ones(n) / n

        def risk_parity_objective(w):
            port_vol = np.sqrt(w @ Sigma @ w)
            # 한계 리스크 기여도
            mrc = Sigma @ w / port_vol
            # 리스크 기여도
            rc = w * mrc
            # 동등한 리스크 기여도에서의 편차 최소화
            target_rc = port_vol / n
            return np.sum((rc - target_rc) ** 2)

        bounds = Bounds(lb=0.001, ub=0.5)
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

        result = minimize(
            risk_parity_objective, w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )

        weights = pd.Series(
            result.x / result.x.sum(),
            index=symbols,
        )
        return weights

    def _calc_sharpe(
        self,
        weights: pd.Series,
        mu: np.ndarray,
        Sigma: np.ndarray,
    ) -> float:
        """포트폴리오 샤프 비율 계산"""
        w = weights.to_numpy()
        port_return = w @ mu
        port_vol = np.sqrt(w @ Sigma @ w)
        if port_vol < 1e-10:
            return 0.0
        return (port_return - self.risk_free_rate) / port_vol

    def efficient_frontier(
        self,
        expected_returns: pd.Series,
        covariance_matrix: pd.DataFrame,
        n_points: int = 100,
    ) -> pd.DataFrame:
        """
        효율적 프론티어 계산.

        Returns:
            DataFrame with columns: [return, volatility, sharpe, weights...]
        """
        mu = expected_returns.to_numpy()
        Sigma = covariance_matrix.to_numpy()
        n = len(mu)

        min_ret = mu.min() * 0.8
        max_ret = mu.max() * 1.2
        target_returns = np.linspace(min_ret, max_ret, n_points)

        frontier_points = []

        for target_ret in target_returns:
            bounds = Bounds(lb=0.0, ub=1.0)
            constraints = [
                {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
                {"type": "eq", "fun": lambda w, r=target_ret: w @ mu - r},
            ]

            result = minimize(
                lambda w: w @ Sigma @ w,
                np.ones(n) / n,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
            )

            if result.success:
                w = result.x
                port_vol = np.sqrt(w @ Sigma @ w)
                sharpe = (target_ret - self.risk_free_rate) / max(port_vol, 1e-10)

                point = {
                    "return": target_ret,
                    "volatility": port_vol,
                    "sharpe": sharpe,
                }
                for sym, wi in zip(expected_returns.index, w):
                    point[f"weight_{sym}"] = wi

                frontier_points.append(point)

        return pd.DataFrame(frontier_points)


class BlackLittermanOptimizer(BasePortfolioOptimizer):
    """
    Black-Litterman 모델.
    시장 균형 수익률 + 투자자 뷰를 결합하여
    더 안정적인 포트폴리오 추정.
    """

    def __init__(
        self,
        risk_free_rate: float = 0.03,
        risk_aversion: float = 2.5,  # 위험 회피 계수
        tau: float = 0.05,  # 불확실성 스케일
    ) -> None:
        self.risk_free_rate = risk_free_rate
        self.risk_aversion = risk_aversion
        self.tau = tau

    def optimize(
        self,
        expected_returns: pd.Series,
        covariance_matrix: pd.DataFrame,
        constraints: dict[str, Any] = None,
        market_weights: Optional[pd.Series] = None,
        views: Optional[list[dict]] = None,
    ) -> pd.Series:
        """
        Black-Litterman 최적화.

        Args:
            market_weights: 시장 자본화 비중 (균등 배분 사용 가능)
            views: 투자자 뷰 리스트
                [{"assets": ["AAPL"], "outperform": ["MSFT"], "confidence": 0.5}]
        """
        symbols = expected_returns.index
        n = len(symbols)
        Sigma = covariance_matrix.to_numpy()

        # 시장 균형 비중
        if market_weights is None:
            market_weights = pd.Series(np.ones(n) / n, index=symbols)

        w_mkt = market_weights.to_numpy()

        # 시장 균형 수익률 (역최적화)
        pi = self.risk_aversion * (Sigma @ w_mkt)

        if views is None or len(views) == 0:
            # 뷰 없으면 시장 균형 수익률로 최적화
            bl_returns = pd.Series(pi, index=symbols)
            optimizer = MeanVarianceOptimizer(risk_free_rate=self.risk_free_rate)
            return optimizer.optimize(bl_returns, covariance_matrix)

        # 뷰 행렬 구성
        P, q, Omega = self._build_view_matrices(views, symbols)

        # Black-Litterman 수익률 업데이트
        tau_sigma = self.tau * Sigma
        M = np.linalg.inv(
            np.linalg.inv(tau_sigma) + P.T @ np.linalg.inv(Omega) @ P
        )
        bl_mu = M @ (np.linalg.inv(tau_sigma) @ pi + P.T @ np.linalg.inv(Omega) @ q)

        bl_returns = pd.Series(bl_mu, index=symbols)

        # 최적화
        optimizer = MeanVarianceOptimizer(risk_free_rate=self.risk_free_rate)
        return optimizer.optimize(bl_returns, covariance_matrix)

    def _build_view_matrices(
        self,
        views: list[dict],
        symbols: pd.Index,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """뷰 행렬 구성"""
        n = len(symbols)
        k = len(views)
        symbol_to_idx = {s: i for i, s in enumerate(symbols)}

        P = np.zeros((k, n))
        q = np.zeros(k)
        omega_diag = np.zeros(k)

        for i, view in enumerate(views):
            if "assets" in view:
                for asset in view["assets"]:
                    if asset in symbol_to_idx:
                        P[i, symbol_to_idx[asset]] = 1.0 / len(view["assets"])

            if "outperform" in view:
                for asset in view["outperform"]:
                    if asset in symbol_to_idx:
                        P[i, symbol_to_idx[asset]] -= 1.0 / len(view["outperform"])

            q[i] = view.get("expected_return", 0.0)
            confidence = view.get("confidence", 0.5)
            omega_diag[i] = (1 - confidence) / confidence

        Omega = np.diag(omega_diag + 1e-6)  # 수치 안정성

        return P, q, Omega

    def efficient_frontier(
        self,
        expected_returns: pd.Series,
        covariance_matrix: pd.DataFrame,
        n_points: int = 100,
    ) -> pd.DataFrame:
        optimizer = MeanVarianceOptimizer()
        return optimizer.efficient_frontier(expected_returns, covariance_matrix, n_points)
