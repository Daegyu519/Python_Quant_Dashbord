"""
=============================================================================
Yahoo Finance Data Collector
=============================================================================
주식 데이터 수집기. yfinance 라이브러리 기반.
Rate Limiting, 재시도 로직, 데이터 검증 포함.
=============================================================================
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from typing import Any, AsyncGenerator, Optional

import numpy as np
import pandas as pd
import yfinance as yf
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.core.base_classes import BaseDataCollector
from app.core.types import (
    OHLCV, OrderBook, OrderBookLevel, OHLCVFrame,
    Symbol, Tick, Timeframe, OrderSide
)
from app.config.logging_config import get_logger, PerformanceLogger

logger = get_logger(__name__)

# yfinance 타임프레임 매핑
YFINANCE_TIMEFRAME_MAP: dict[Timeframe, str] = {
    Timeframe.ONE_MIN: "1m",
    Timeframe.FIVE_MIN: "5m",
    Timeframe.FIFTEEN_MIN: "15m",
    Timeframe.THIRTY_MIN: "30m",
    Timeframe.ONE_HOUR: "1h",
    Timeframe.ONE_DAY: "1d",
    Timeframe.ONE_WEEK: "1wk",
    Timeframe.ONE_MONTH: "1mo",
}

# yfinance 최대 조회 기간 제한
YFINANCE_MAX_PERIOD: dict[str, int] = {
    "1m": 7,      # 7일
    "5m": 60,     # 60일
    "15m": 60,
    "30m": 60,
    "1h": 730,    # 2년
    "1d": 36500,  # 100년
    "1wk": 36500,
    "1mo": 36500,
}


class YahooFinanceCollector(BaseDataCollector):
    """
    Yahoo Finance 데이터 수집기.

    특징:
    - 비동기 배치 수집 (ThreadPoolExecutor 활용)
    - 자동 재시도 (지수 백오프)
    - 데이터 품질 검증
    - 멀티 심볼 병렬 수집
    """

    def __init__(self, max_workers: int = 10) -> None:
        super().__init__(source_name="yahoo_finance")
        self._max_workers = max_workers
        self._semaphore = asyncio.Semaphore(max_workers)
        self._executor = None

    @retry(
        retry=retry_if_exception_type((ConnectionError, TimeoutError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
    )
    async def fetch_ohlcv(
        self,
        symbol: Symbol,
        timeframe: Timeframe,
        start: datetime,
        end: Optional[datetime] = None,
        limit: int = 1000,
    ) -> OHLCVFrame:
        """
        OHLCV 데이터 수집.

        Args:
            symbol: 종목 코드 (예: "AAPL", "005930.KS")
            timeframe: 타임프레임
            start: 시작 날짜
            end: 종료 날짜 (None이면 현재)
            limit: 최대 레코드 수

        Returns:
            OHLCVFrame 객체
        """
        async with self._semaphore:
            with PerformanceLogger("yahoo_fetch_ohlcv", symbol=symbol, timeframe=timeframe.value):
                return await asyncio.get_event_loop().run_in_executor(
                    None,
                    self._sync_fetch_ohlcv,
                    symbol, timeframe, start, end, limit
                )

    def _sync_fetch_ohlcv(
        self,
        symbol: Symbol,
        timeframe: Timeframe,
        start: datetime,
        end: Optional[datetime],
        limit: int,
    ) -> OHLCVFrame:
        """동기 OHLCV 수집 (ThreadPool에서 실행)"""
        yf_interval = YFINANCE_TIMEFRAME_MAP.get(timeframe, "1d")
        end = end or datetime.utcnow()

        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(
                start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
                interval=yf_interval,
                auto_adjust=True,
                prepost=False,
            )

            if df.empty:
                logger.warning("yahoo_empty_data", symbol=symbol, timeframe=timeframe.value)
                return self._empty_frame(symbol, timeframe)

            # 컬럼명 정규화
            df.columns = [c.lower() for c in df.columns]
            df = df.rename(columns={
                "stock splits": "stock_splits",
                "capital gains": "capital_gains",
            })

            # 필수 컬럼 확인
            required = ["open", "high", "low", "close", "volume"]
            df = df[required].copy()

            # 데이터 검증
            df = self._validate_and_clean(df, symbol)

            # 최신 데이터 limit 제한
            if len(df) > limit:
                df = df.iloc[-limit:]

            logger.info(
                "yahoo_data_collected",
                symbol=symbol,
                timeframe=timeframe.value,
                bars=len(df),
                start=df.index[0].isoformat() if len(df) > 0 else None,
                end=df.index[-1].isoformat() if len(df) > 0 else None,
            )

            return OHLCVFrame(symbol=symbol, timeframe=timeframe, data=df)

        except Exception as e:
            logger.error(
                "yahoo_fetch_failed",
                symbol=symbol,
                error=str(e),
                exc_info=True,
            )
            raise

    def _validate_and_clean(self, df: pd.DataFrame, symbol: Symbol) -> pd.DataFrame:
        """
        데이터 품질 검증 및 정제.

        처리:
        1. NaN 제거
        2. 음수 가격/거래량 제거
        3. OHLC 논리 검증 (H >= max(O,C), L <= min(O,C))
        4. 이상치 감지 (5-sigma)
        """
        original_len = len(df)

        # NaN 제거
        df = df.dropna()

        # 음수값 제거
        df = df[(df["open"] > 0) & (df["high"] > 0) &
                (df["low"] > 0) & (df["close"] > 0) & (df["volume"] >= 0)]

        # OHLC 논리 검증
        valid_ohlc = (
            (df["high"] >= df["open"]) &
            (df["high"] >= df["close"]) &
            (df["low"] <= df["open"]) &
            (df["low"] <= df["close"])
        )
        df = df[valid_ohlc]

        # 이상치 감지 (수익률 기준 5-sigma)
        returns = df["close"].pct_change().abs()
        threshold = returns.mean() + 5 * returns.std()
        df = df[returns <= threshold]

        cleaned_len = len(df)
        if cleaned_len < original_len:
            logger.warning(
                "yahoo_data_cleaned",
                symbol=symbol,
                removed=original_len - cleaned_len,
                pct_removed=f"{(original_len - cleaned_len) / original_len * 100:.1f}%",
            )

        return df

    def _empty_frame(self, symbol: Symbol, timeframe: Timeframe) -> OHLCVFrame:
        """빈 OHLCVFrame 반환"""
        empty_df = pd.DataFrame(
            columns=["open", "high", "low", "close", "volume"],
            index=pd.DatetimeIndex([], tz="UTC"),
        )
        return OHLCVFrame(symbol=symbol, timeframe=timeframe, data=empty_df)

    async def fetch_multiple_symbols(
        self,
        symbols: list[Symbol],
        timeframe: Timeframe,
        start: datetime,
        end: Optional[datetime] = None,
    ) -> dict[Symbol, OHLCVFrame]:
        """
        여러 종목 병렬 수집.

        asyncio.gather를 사용하여 모든 종목을 동시에 수집.
        Semaphore로 동시 요청 수 제한.
        """
        tasks = [
            self.fetch_ohlcv(symbol, timeframe, start, end)
            for symbol in symbols
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        data: dict[Symbol, OHLCVFrame] = {}
        for symbol, result in zip(symbols, results):
            if isinstance(result, Exception):
                logger.error(
                    "symbol_fetch_failed",
                    symbol=symbol,
                    error=str(result),
                )
            else:
                data[symbol] = result

        logger.info(
            "batch_fetch_completed",
            requested=len(symbols),
            succeeded=len(data),
            failed=len(symbols) - len(data),
        )

        return data

    async def fetch_ticker(self, symbol: Symbol) -> dict[str, Any]:
        """현재 가격 정보 조회"""
        return await asyncio.get_event_loop().run_in_executor(
            None, self._sync_fetch_ticker, symbol
        )

    def _sync_fetch_ticker(self, symbol: Symbol) -> dict[str, Any]:
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.fast_info
            return {
                "symbol": symbol,
                "price": getattr(info, "last_price", 0.0),
                "volume": getattr(info, "three_month_average_volume", 0.0),
                "market_cap": getattr(info, "market_cap", 0.0),
            }
        except Exception as e:
            logger.error("yahoo_ticker_failed", symbol=symbol, error=str(e))
            return {"symbol": symbol, "price": 0.0, "volume": 0.0}

    async def fetch_orderbook(self, symbol: Symbol, depth: int = 20) -> OrderBook:
        """Yahoo Finance는 호가창 미지원 → 더미 반환"""
        logger.warning("yahoo_orderbook_not_supported", symbol=symbol)
        return OrderBook(
            symbol=symbol,
            timestamp=datetime.utcnow(),
            bids=[],
            asks=[],
        )

    async def stream_ticks(
        self, symbols: list[Symbol]
    ) -> AsyncGenerator[Tick, None]:
        """Yahoo Finance는 실시간 틱 미지원"""
        raise NotImplementedError(
            "Yahoo Finance does not support real-time tick streaming. "
            "Use BinanceCollector or PolygonCollector for real-time data."
        )

    async def stream_ohlcv(
        self, symbol: Symbol, timeframe: Timeframe
    ) -> AsyncGenerator[OHLCV, None]:
        """Yahoo Finance는 실시간 스트리밍 미지원"""
        raise NotImplementedError(
            "Yahoo Finance does not support real-time streaming."
        )

    def get_supported_symbols(self) -> list[Symbol]:
        """Yahoo Finance 지원 심볼 (제한 없음 - 모든 야후 심볼 지원)"""
        return []  # 동적 - 미리 알 수 없음

    # ─────────────────────────────────────────────
    # 유틸리티 메서드
    # ─────────────────────────────────────────────

    async def get_stock_info(self, symbol: Symbol) -> dict[str, Any]:
        """종목 기본 정보 조회"""
        return await asyncio.get_event_loop().run_in_executor(
            None, lambda: yf.Ticker(symbol).info
        )

    async def get_financial_statements(
        self, symbol: Symbol
    ) -> dict[str, pd.DataFrame]:
        """재무제표 조회"""
        def _sync_get():
            ticker = yf.Ticker(symbol)
            return {
                "income_statement": ticker.financials,
                "balance_sheet": ticker.balance_sheet,
                "cashflow": ticker.cashflow,
            }

        return await asyncio.get_event_loop().run_in_executor(None, _sync_get)

    async def download_bulk(
        self,
        symbols: list[Symbol],
        timeframe: Timeframe,
        start: datetime,
        end: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        yfinance.download() 활용 대량 다운로드.
        단일 API 호출로 여러 종목 수집 (더 효율적).
        """
        yf_interval = YFINANCE_TIMEFRAME_MAP.get(timeframe, "1d")
        end = end or datetime.utcnow()

        def _sync_download():
            return yf.download(
                tickers=symbols,
                start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
                interval=yf_interval,
                auto_adjust=True,
                threads=True,
                group_by="ticker",
            )

        with PerformanceLogger("yahoo_bulk_download", n_symbols=len(symbols)):
            df = await asyncio.get_event_loop().run_in_executor(None, _sync_download)

        logger.info(
            "yahoo_bulk_download_complete",
            symbols=len(symbols),
            rows=len(df),
        )
        return df
