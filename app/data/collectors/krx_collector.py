"""
=============================================================================
KRX / 한국 주식 데이터 수집기 — 완전 무료 (가입/키 불필요)
=============================================================================
FinanceDataReader + pykrx 기반.
두 라이브러리 모두 KRX 공식 데이터를 크롤링하며 API 키 불필요.

데이터 소스:
  - FinanceDataReader: 빠른 OHLCV, 종목 리스트
  - pykrx: PER/PBR/DIV 등 펀더멘털 포함, KRX 공식 자료
=============================================================================
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from typing import Any, AsyncGenerator, Optional

import numpy as np
import pandas as pd

from app.core.base_classes import BaseDataCollector
from app.core.types import (
    OHLCV, OrderBook, OHLCVFrame, Symbol, Tick, Timeframe,
)
from app.config.logging_config import get_logger, PerformanceLogger

logger = get_logger(__name__)

# KRX는 일봉만 지원
KRX_SUPPORTED_TIMEFRAMES = {Timeframe.ONE_DAY, Timeframe.ONE_WEEK, Timeframe.ONE_MONTH}


class KRXCollector(BaseDataCollector):
    """
    한국거래소(KRX) 데이터 수집기.

    ✅ 완전 무료 — 가입/API 키 불필요
    ✅ 코스피/코스닥 전 종목 지원
    ✅ PER, PBR, DIV 등 펀더멘털 데이터 포함

    지원 데이터:
      - OHLCV (일/주/월)
      - 시가총액
      - PER, PBR, DIV (pykrx)
      - 전체 종목 리스트 (FinanceDataReader)

    주의:
      - 실시간 WebSocket 미지원 (T+0 장중 데이터 없음)
      - 전일 종가까지만 제공 (약 15분~1시간 지연)
    """

    def __init__(self) -> None:
        super().__init__(source_name="krx")
        self._fdr_available = False
        self._pykrx_available = False
        self._check_dependencies()

    def _check_dependencies(self) -> None:
        """라이브러리 설치 여부 확인"""
        try:
            import FinanceDataReader  # noqa
            self._fdr_available = True
        except ImportError:
            logger.warning(
                "fdr_not_installed",
                hint="pip install finance-datareader",
            )

        try:
            from pykrx import stock  # noqa
            self._pykrx_available = True
        except ImportError:
            logger.warning(
                "pykrx_not_installed",
                hint="pip install pykrx",
            )

    async def fetch_ohlcv(
        self,
        symbol: Symbol,
        timeframe: Timeframe,
        start: datetime,
        end: Optional[datetime] = None,
        limit: int = 2000,
    ) -> OHLCVFrame:
        """
        KRX OHLCV 수집.

        Args:
            symbol: 종목 코드 (예: "005930" = 삼성전자, "000660" = SK하이닉스)
                    또는 지수 코드 ("KS11" = 코스피, "KQ11" = 코스닥)
            timeframe: 일봉만 지원 (ONE_DAY)
        """
        if timeframe not in KRX_SUPPORTED_TIMEFRAMES:
            logger.warning(
                "krx_timeframe_not_supported",
                timeframe=timeframe.value,
                supported=[t.value for t in KRX_SUPPORTED_TIMEFRAMES],
            )
            # 일봉으로 수집 후 리샘플링
            timeframe = Timeframe.ONE_DAY

        end = end or datetime.now()

        return await asyncio.get_event_loop().run_in_executor(
            None,
            self._sync_fetch_ohlcv,
            symbol, timeframe, start, end, limit,
        )

    def _sync_fetch_ohlcv(
        self,
        symbol: Symbol,
        timeframe: Timeframe,
        start: datetime,
        end: datetime,
        limit: int,
    ) -> OHLCVFrame:
        """동기 OHLCV 수집"""
        with PerformanceLogger("krx_fetch_ohlcv", symbol=symbol):
            df = self._fetch_with_fdr(symbol, start, end)

            if df is None or df.empty:
                logger.warning("krx_empty_data", symbol=symbol)
                return self._empty_frame(symbol, timeframe)

            # 컬럼 정규화
            df.columns = [c.lower() for c in df.columns]
            col_map = {
                "시가": "open", "고가": "high", "저가": "low",
                "종가": "close", "거래량": "volume",
                "open": "open", "high": "high", "low": "low",
                "close": "close", "volume": "volume",
            }
            df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

            # 필수 컬럼 확인
            required = ["open", "high", "low", "close", "volume"]
            for col in required:
                if col not in df.columns:
                    logger.error("krx_missing_column", symbol=symbol, column=col)
                    return self._empty_frame(symbol, timeframe)

            df = df[required].copy()
            df = df.astype(float)
            df = df.dropna()

            # 인덱스 timezone 설정
            if df.index.tz is None:
                df.index = df.index.tz_localize("Asia/Seoul")
            df.index = df.index.tz_convert("UTC")

            # 주봉/월봉 리샘플링
            if timeframe == Timeframe.ONE_WEEK:
                df = self._resample(df, "W-FRI")
            elif timeframe == Timeframe.ONE_MONTH:
                df = self._resample(df, "ME")

            if len(df) > limit:
                df = df.iloc[-limit:]

            logger.info(
                "krx_data_collected",
                symbol=symbol,
                timeframe=timeframe.value,
                bars=len(df),
                start=df.index[0].date().isoformat(),
                end=df.index[-1].date().isoformat(),
            )

            return OHLCVFrame(symbol=symbol, timeframe=timeframe, data=df)

    def _fetch_with_fdr(
        self, symbol: str, start: datetime, end: datetime
    ) -> Optional[pd.DataFrame]:
        """FinanceDataReader로 데이터 수집"""
        if not self._fdr_available:
            return self._fetch_with_pykrx(symbol, start, end)

        try:
            import FinanceDataReader as fdr
            df = fdr.DataReader(
                symbol,
                start.strftime("%Y-%m-%d"),
                end.strftime("%Y-%m-%d"),
            )
            return df
        except Exception as e:
            logger.error("fdr_fetch_failed", symbol=symbol, error=str(e))
            # pykrx로 폴백
            return self._fetch_with_pykrx(symbol, start, end)

    def _fetch_with_pykrx(
        self, symbol: str, start: datetime, end: datetime
    ) -> Optional[pd.DataFrame]:
        """pykrx로 데이터 수집 (폴백)"""
        if not self._pykrx_available:
            return None

        try:
            from pykrx import stock
            df = stock.get_market_ohlcv(
                start.strftime("%Y%m%d"),
                end.strftime("%Y%m%d"),
                symbol,
            )
            return df
        except Exception as e:
            logger.error("pykrx_fetch_failed", symbol=symbol, error=str(e))
            return None

    def _resample(self, df: pd.DataFrame, rule: str) -> pd.DataFrame:
        """OHLCV 리샘플링"""
        return df.resample(rule).agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }).dropna()

    def _empty_frame(self, symbol: Symbol, timeframe: Timeframe) -> OHLCVFrame:
        empty = pd.DataFrame(
            columns=["open", "high", "low", "close", "volume"],
            index=pd.DatetimeIndex([], tz="UTC"),
        )
        return OHLCVFrame(symbol=symbol, timeframe=timeframe, data=empty)

    async def fetch_fundamentals(
        self, symbol: Symbol, start: datetime, end: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        PER, PBR, DIV 등 펀더멘털 데이터 (pykrx).
        팩터 모델 구축에 활용.
        """
        end = end or datetime.now()

        def _sync():
            if not self._pykrx_available:
                raise RuntimeError("pykrx required: pip install pykrx")

            from pykrx import stock
            df = stock.get_market_fundamental(
                start.strftime("%Y%m%d"),
                end.strftime("%Y%m%d"),
                symbol,
            )
            # 컬럼: BPS, PER, PBR, EPS, DIV, DPS
            return df

        return await asyncio.get_event_loop().run_in_executor(None, _sync)

    async def get_stock_listing(self, market: str = "KRX") -> pd.DataFrame:
        """
        전체 종목 리스트 조회.

        Args:
            market: "KRX", "KOSPI", "KOSDAQ", "KONEX"

        Returns:
            DataFrame (Code, Name, Sector, Industry, ...)
        """
        def _sync():
            if self._fdr_available:
                import FinanceDataReader as fdr
                return fdr.StockListing(market)
            elif self._pykrx_available:
                from pykrx import stock
                tickers = stock.get_market_ticker_list(market=market)
                names = [stock.get_market_ticker_name(t) for t in tickers]
                return pd.DataFrame({"Code": tickers, "Name": names})
            else:
                raise RuntimeError("Either FinanceDataReader or pykrx required")

        return await asyncio.get_event_loop().run_in_executor(None, _sync)

    async def fetch_multiple_symbols(
        self,
        symbols: list[Symbol],
        timeframe: Timeframe,
        start: datetime,
        end: Optional[datetime] = None,
    ) -> dict[Symbol, OHLCVFrame]:
        """여러 종목 병렬 수집"""
        tasks = [
            self.fetch_ohlcv(symbol, timeframe, start, end)
            for symbol in symbols
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        data: dict[Symbol, OHLCVFrame] = {}
        for symbol, result in zip(symbols, results):
            if isinstance(result, Exception):
                logger.error("krx_symbol_failed", symbol=symbol, error=str(result))
            else:
                data[symbol] = result

        return data

    # ── BaseDataCollector 필수 구현 ──

    async def fetch_ticker(self, symbol: Symbol) -> dict[str, Any]:
        """최신 가격 정보"""
        data = await self.fetch_ohlcv(
            symbol, Timeframe.ONE_DAY,
            start=datetime.now() - timedelta(days=5),
        )
        if data.n_bars > 0:
            latest = data.data.iloc[-1]
            return {
                "symbol": symbol,
                "price": float(latest["close"]),
                "volume": float(latest["volume"]),
                "open": float(latest["open"]),
                "high": float(latest["high"]),
                "low": float(latest["low"]),
            }
        return {"symbol": symbol, "price": 0.0, "volume": 0.0}

    async def fetch_orderbook(self, symbol: Symbol, depth: int = 20) -> OrderBook:
        """KRX는 실시간 호가창 미지원"""
        from app.core.types import OrderBook
        logger.warning("krx_orderbook_not_supported", symbol=symbol)
        return OrderBook(
            symbol=symbol,
            timestamp=datetime.utcnow(),
            bids=[],
            asks=[],
        )

    async def stream_ticks(self, symbols: list[Symbol]) -> AsyncGenerator[Tick, None]:
        raise NotImplementedError("KRX does not support real-time tick streaming")

    async def stream_ohlcv(self, symbol: Symbol, timeframe: Timeframe) -> AsyncGenerator[OHLCV, None]:
        raise NotImplementedError("KRX does not support real-time streaming")

    def get_supported_symbols(self) -> list[Symbol]:
        """주요 대형주 코드 반환 (전체는 get_stock_listing() 사용)"""
        return [
            "005930",  # 삼성전자
            "000660",  # SK하이닉스
            "035420",  # NAVER
            "005380",  # 현대차
            "051910",  # LG화학
            "006400",  # 삼성SDI
            "035720",  # 카카오
            "207940",  # 삼성바이오로직스
            "068270",  # 셀트리온
            "105560",  # KB금융
            # 지수
            "KS11",    # KOSPI
            "KQ11",    # KOSDAQ
        ]
