"""백테스팅 API 라우터"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query
from pydantic import BaseModel, Field

router = APIRouter()


class BacktestRequest(BaseModel):
    strategy_id: str = Field(..., description="전략 ID")
    symbol: str = Field(..., description="종목 코드 (예: AAPL)")
    start_date: str = Field(..., description="시작일 (YYYY-MM-DD)")
    end_date: str = Field(..., description="종료일 (YYYY-MM-DD)")
    timeframe: str = Field(default="1d", description="타임프레임")
    initial_capital: float = Field(default=100_000_000.0, description="초기자본 (원)")
    commission_bps: float = Field(default=5.0, description="수수료 (bps)")
    slippage_bps: float = Field(default=2.0, description="슬리피지 (bps)")
    enable_monte_carlo: bool = Field(default=False)
    enable_walk_forward: bool = Field(default=False)
    n_wf_splits: int = Field(default=5)


class OptimizeRequest(BaseModel):
    strategy_type: str
    symbol: str
    start_date: str
    end_date: str
    param_grid: dict[str, list]
    metric: str = "sharpe_ratio"


@router.post("/run", summary="백테스트 실행")
async def run_backtest(
    request: BacktestRequest,
    background_tasks: BackgroundTasks,
) -> dict[str, Any]:
    """
    전략 백테스트 실행.

    - **strategy_id**: 실행할 전략 ID
    - **symbol**: 대상 종목 코드
    - **initial_capital**: 초기 자본금

    반환: run_id (비동기 실행), 완료 후 /results/{run_id}로 결과 조회
    """
    run_id = str(uuid.uuid4())

    # TODO: 실제 백테스트 Celery 태스크 등록
    # background_tasks.add_task(execute_backtest, run_id, request)

    return {
        "run_id": run_id,
        "status": "queued",
        "message": "Backtest queued for execution",
        "estimated_time_seconds": 30,
    }


@router.get("/results/{run_id}", summary="백테스트 결과 조회")
async def get_backtest_result(run_id: str) -> dict[str, Any]:
    """백테스트 결과 조회"""
    # TODO: DB에서 결과 조회
    return {
        "run_id": run_id,
        "status": "completed",
        "metrics": {
            "total_return": 0.245,
            "cagr": 0.182,
            "sharpe_ratio": 1.82,
            "sortino_ratio": 2.14,
            "max_drawdown": -0.127,
            "win_rate": 0.548,
            "profit_factor": 1.85,
            "total_trades": 142,
            "var_95": 0.018,
        }
    }


@router.get("/history", summary="백테스트 히스토리")
async def get_backtest_history(
    strategy_id: Optional[str] = Query(None),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
) -> dict[str, Any]:
    """백테스트 실행 이력 조회"""
    return {"results": [], "total": 0, "limit": limit, "offset": offset}


@router.post("/optimize", summary="파라미터 최적화")
async def optimize_parameters(
    request: OptimizeRequest,
    background_tasks: BackgroundTasks,
) -> dict[str, Any]:
    """Optuna 기반 하이퍼파라미터 최적화"""
    run_id = str(uuid.uuid4())
    return {
        "run_id": run_id,
        "status": "queued",
        "n_combinations": sum(len(v) for v in request.param_grid.values()),
    }


@router.post("/monte-carlo", summary="Monte Carlo 시뮬레이션")
async def run_monte_carlo(
    run_id: str,
    n_simulations: int = Query(10_000, ge=100, le=100_000),
) -> dict[str, Any]:
    """백테스트 결과 기반 Monte Carlo 시뮬레이션"""
    return {
        "run_id": run_id,
        "n_simulations": n_simulations,
        "results": {
            "prob_positive": 0.73,
            "prob_2x": 0.21,
            "final_p50": 145_000_000,
            "final_p5": 82_000_000,
            "final_p95": 245_000_000,
            "mdd_p50": 0.12,
        }
    }
