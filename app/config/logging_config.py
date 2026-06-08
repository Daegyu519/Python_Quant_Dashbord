"""
=============================================================================
Production-Grade Logging Configuration
=============================================================================
구조화된 JSON 로깅 + 성능 모니터링 + 컨텍스트 추적.
Prometheus 메트릭 수집 포함.
=============================================================================
"""

from __future__ import annotations

import logging
import logging.config
import sys
import time
from contextvars import ContextVar
from typing import Any
from uuid import uuid4

import structlog
from pythonjsonlogger import jsonlogger

# ─────────────────────────────────────────────
# 요청 컨텍스트 변수
# ─────────────────────────────────────────────
request_id_var: ContextVar[str] = ContextVar("request_id", default="")
user_id_var: ContextVar[str] = ContextVar("user_id", default="")
strategy_id_var: ContextVar[str] = ContextVar("strategy_id", default="")


def setup_logging(log_level: str = "INFO", json_output: bool = True) -> None:
    """
    애플리케이션 전체 로깅 초기화.

    Args:
        log_level: 최소 로그 레벨
        json_output: JSON 포맷 여부 (프로덕션에서는 True)
    """
    # structlog 공유 프로세서
    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if json_output:
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=shared_processors + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # 표준 logging 설정
    logging.config.dictConfig({
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "json": {
                "()": structlog.stdlib.ProcessorFormatter,
                "processor": renderer,
                "foreign_pre_chain": shared_processors,
            },
            "standard": {
                "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "json" if json_output else "standard",
                "stream": "ext://sys.stdout",
            },
            "file": {
                "class": "logging.handlers.RotatingFileHandler",
                "formatter": "json",
                "filename": "logs/quant_platform.log",
                "maxBytes": 100 * 1024 * 1024,  # 100MB
                "backupCount": 10,
                "encoding": "utf-8",
            },
            "error_file": {
                "class": "logging.handlers.RotatingFileHandler",
                "formatter": "json",
                "filename": "logs/error.log",
                "maxBytes": 50 * 1024 * 1024,  # 50MB
                "backupCount": 5,
                "encoding": "utf-8",
                "level": "ERROR",
            },
        },
        "root": {
            "level": log_level,
            "handlers": ["console", "file", "error_file"],
        },
        "loggers": {
            # 노이즈 줄이기
            "uvicorn.access": {"level": "WARNING"},
            "sqlalchemy.engine": {"level": "WARNING"},
            "httpx": {"level": "WARNING"},
            "asyncio": {"level": "WARNING"},
            # 퀀트 플랫폼 모듈
            "app": {"level": log_level, "propagate": True},
        },
    })


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """
    모듈별 logger 인스턴스 반환.

    Usage:
        logger = get_logger(__name__)
        logger.info("backtest_started", strategy="MA_Cross", symbol="AAPL")
    """
    return structlog.get_logger(name)


class PerformanceLogger:
    """
    성능 측정용 컨텍스트 매니저.
    함수 실행 시간을 자동 측정하고 로깅.
    """

    def __init__(self, operation: str, **kwargs: Any) -> None:
        self.operation = operation
        self.kwargs = kwargs
        self.logger = get_logger("performance")
        self._start: float = 0.0

    def __enter__(self) -> "PerformanceLogger":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args: Any) -> None:
        elapsed_ms = (time.perf_counter() - self._start) * 1000
        self.logger.info(
            "performance_metric",
            operation=self.operation,
            elapsed_ms=round(elapsed_ms, 3),
            **self.kwargs,
        )


def log_execution_time(logger: structlog.stdlib.BoundLogger):
    """
    함수 실행 시간 자동 로깅 데코레이터.

    Usage:
        @log_execution_time(logger)
        def run_backtest(self, ...) -> ...:
    """
    import functools

    def decorator(func):
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                elapsed = (time.perf_counter() - start) * 1000
                logger.info(
                    "function_executed",
                    function=func.__qualname__,
                    elapsed_ms=round(elapsed, 3),
                )
                return result
            except Exception as e:
                elapsed = (time.perf_counter() - start) * 1000
                logger.error(
                    "function_failed",
                    function=func.__qualname__,
                    elapsed_ms=round(elapsed, 3),
                    error=str(e),
                )
                raise

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                result = await func(*args, **kwargs)
                elapsed = (time.perf_counter() - start) * 1000
                logger.info(
                    "async_function_executed",
                    function=func.__qualname__,
                    elapsed_ms=round(elapsed, 3),
                )
                return result
            except Exception as e:
                elapsed = (time.perf_counter() - start) * 1000
                logger.error(
                    "async_function_failed",
                    function=func.__qualname__,
                    elapsed_ms=round(elapsed, 3),
                    error=str(e),
                )
                raise

        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator
