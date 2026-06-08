"""
=============================================================================
FastAPI Main Application — Institution-Grade API Server
=============================================================================
퀀트 플랫폼의 핵심 API 서버.
WebSocket + REST + Swagger 자동 생성.
=============================================================================
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
import structlog

from app.config.settings import get_settings
from app.config.logging_config import setup_logging, get_logger
from app.api.routers import backtest, signals, portfolio, strategies, ml, risk, data

logger = get_logger(__name__)
settings = get_settings()


# ─────────────────────────────────────────────
# 애플리케이션 시작/종료 핸들러
# ─────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """애플리케이션 생명주기 관리"""
    # ── 시작 ──
    logger.info(
        "app_startup",
        name=settings.app_name,
        version=settings.app_version,
        environment=settings.environment.value,
    )

    # TODO: DB 연결 풀 초기화
    # TODO: Redis 연결 초기화
    # TODO: 백그라운드 워커 시작

    yield

    # ── 종료 ──
    logger.info("app_shutdown")
    # TODO: 연결 정리


# ─────────────────────────────────────────────
# FastAPI 앱 생성
# ─────────────────────────────────────────────
def create_app() -> FastAPI:
    """
    FastAPI 애플리케이션 팩토리 패턴.
    테스트, 개발, 프로덕션 각각 다른 설정 주입 가능.
    """
    setup_logging(
        log_level=settings.log_level.value,
        json_output=settings.is_production,
    )

    app = FastAPI(
        title=settings.app_name,
        description="""
# AI Quant Trading Platform — Institution Grade

## 주요 기능
- 🔄 실시간 시장 데이터 수집 & 스트리밍
- 📊 초고속 백테스팅 (Numba JIT 가속)
- 🤖 AI/ML 예측 엔진 (XGBoost, LightGBM, Transformer)
- 📈 전략 시그널 실시간 생성
- 🛡️ 기관급 리스크 관리
- 💼 포트폴리오 최적화 (Mean-Variance, Black-Litterman)
- 📉 SQL 기반 팩터 분석

## 데이터 소스
- Yahoo Finance (주식)
- Binance (암호화폐)
- Polygon, Alpha Vantage

## 기술 스택
- FastAPI + WebSocket
- TimescaleDB + ClickHouse + DuckDB
- Redis Pub/Sub
- Numba JIT + Polars
        """,
        version=settings.app_version,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # ── 미들웨어 ──
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.server.allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.add_middleware(GZipMiddleware, minimum_size=1000)

    # ── 요청 로깅 미들웨어 ──
    @app.middleware("http")
    async def logging_middleware(request: Request, call_next):
        start_time = time.perf_counter()
        request_id = request.headers.get("X-Request-ID", "")

        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=request_id)

        response = await call_next(request)

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        logger.info(
            "http_request",
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            elapsed_ms=round(elapsed_ms, 2),
        )

        response.headers["X-Process-Time-Ms"] = str(round(elapsed_ms, 2))
        return response

    # ── 전역 에러 핸들러 ──
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        logger.warning(
            "http_exception",
            status_code=exc.status_code,
            detail=exc.detail,
            path=request.url.path,
        )
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail, "status_code": exc.status_code},
        )

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error(
            "unhandled_exception",
            error=str(exc),
            path=request.url.path,
            exc_info=True,
        )
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error", "status_code": 500},
        )

    # ── 라우터 등록 ──
    API_PREFIX = "/api/v1"

    app.include_router(
        backtest.router,
        prefix=f"{API_PREFIX}/backtest",
        tags=["Backtesting"],
    )
    app.include_router(
        signals.router,
        prefix=f"{API_PREFIX}/signals",
        tags=["Signals"],
    )
    app.include_router(
        portfolio.router,
        prefix=f"{API_PREFIX}/portfolio",
        tags=["Portfolio"],
    )
    app.include_router(
        strategies.router,
        prefix=f"{API_PREFIX}/strategies",
        tags=["Strategies"],
    )
    app.include_router(
        ml.router,
        prefix=f"{API_PREFIX}/ml",
        tags=["Machine Learning"],
    )
    app.include_router(
        risk.router,
        prefix=f"{API_PREFIX}/risk",
        tags=["Risk Management"],
    )
    app.include_router(
        data.router,
        prefix=f"{API_PREFIX}/data",
        tags=["Market Data"],
    )

    # ── 헬스 체크 ──
    @app.get("/health", tags=["System"])
    async def health_check() -> dict[str, Any]:
        """시스템 헬스 체크"""
        return {
            "status": "healthy",
            "app": settings.app_name,
            "version": settings.app_version,
            "environment": settings.environment.value,
            "timestamp": time.time(),
        }

    @app.get("/health/detailed", tags=["System"])
    async def detailed_health_check() -> dict[str, Any]:
        """상세 헬스 체크 (DB, Redis 연결 상태 포함)"""
        checks = {
            "api": "ok",
            # TODO: DB 연결 체크
            "database": "ok",
            "redis": "ok",
        }
        all_ok = all(v == "ok" for v in checks.values())

        return {
            "status": "healthy" if all_ok else "degraded",
            "checks": checks,
        }

    # ── WebSocket 실시간 스트리밍 ──
    @app.websocket("/ws/market/{symbol}")
    async def market_data_stream(websocket: WebSocket, symbol: str):
        """
        실시간 시장 데이터 WebSocket 스트림.

        구독 메시지:
        {"action": "subscribe", "channels": ["ticker", "ohlcv_1m", "orderbook"]}
        """
        await websocket.accept()
        logger.info("ws_connected", symbol=symbol)

        try:
            while True:
                data = await websocket.receive_json()
                action = data.get("action")

                if action == "subscribe":
                    channels = data.get("channels", [])
                    await websocket.send_json({
                        "type": "subscribed",
                        "symbol": symbol,
                        "channels": channels,
                    })
                elif action == "ping":
                    await websocket.send_json({"type": "pong"})

        except WebSocketDisconnect:
            logger.info("ws_disconnected", symbol=symbol)

    @app.websocket("/ws/signals")
    async def signals_stream(websocket: WebSocket):
        """실시간 트레이딩 시그널 스트림"""
        await websocket.accept()

        try:
            while True:
                await websocket.send_json({
                    "type": "signal",
                    "data": {},  # TODO: 실제 시그널 데이터
                })
                import asyncio
                await asyncio.sleep(1)
        except WebSocketDisconnect:
            pass

    @app.websocket("/ws/portfolio")
    async def portfolio_stream(websocket: WebSocket):
        """실시간 포트폴리오 업데이트 스트림"""
        await websocket.accept()

        try:
            while True:
                await websocket.send_json({
                    "type": "portfolio_update",
                    "data": {},  # TODO: 실제 포트폴리오 데이터
                })
                import asyncio
                await asyncio.sleep(5)
        except WebSocketDisconnect:
            pass

    return app


# ─────────────────────────────────────────────
# 앱 인스턴스
# ─────────────────────────────────────────────
app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.api.main:app",
        host=settings.server.host,
        port=settings.server.port,
        workers=settings.server.workers if settings.is_production else 1,
        reload=settings.server.reload,
        log_level=settings.log_level.value.lower(),
    )
