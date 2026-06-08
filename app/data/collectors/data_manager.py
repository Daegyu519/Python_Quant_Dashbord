"""
=============================================================================
DataManager — 통합 데이터 수집 관리자
=============================================================================
모든 수집기를 하나의 인터페이스로 통합.
심볼 형식을 자동 감지하여 적절한 수집기로 라우팅.

지원 소스 (전부 무료):
  - Yahoo Finance   → 미국 주식, ETF, 지수, 환율 (키 불필요)
  - KRX             → 한국 주식 (키 불필요)
  - Binance Public  → 암호화폐 실시간 (키 불필요)
  - CoinGecko       → 암호화폐 OHLCV (키 불필요)
  - Alpha Vantage   → 미국 주식 보완 (무료 키)
  - FRED            → 거시경제 지표 (무료 키)
=============================================================================
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Optional

import pandas as pd

from app.core.types import OHLCVFrame, Symbol, Timeframe
from app.config.logging_config import get_logger, PerformanceLogger

logger = get_logger(__name__)


class DataSource(str, Enum):
    YAHOO = "yahoo"
    KRX = "krx"
    BINANCE = "binance"
    COINGECKO = "coingecko"
    ALPHA_VANTAGE = "alpha_vantage"
    FRED = "fred"


def detect_source(symbol: Symbol) -> DataSource:
    """
    심볼 형식으로 데이터 소스 자동 감지.

    규칙:
      6자리 숫자          → KRX  (예: "005930")
      USDT/USD 포함       → Binance (예: "BTCUSDT", "BTC/USDT")
      BTC, ETH, SOL 등   → CoinGecko
      ^로 시작 (지수)     → Yahoo Finance
      그 외               → Yahoo Finance
    """
    s = symbol.upper().strip()

    # 한국 주식: 6자리 숫자
    if re.match(r"^\d{6}$", s):
        return DataSource.KRX

    # 코스피/코스닥 지수
    if s in ("KS11", "KQ11", "KS200", "KS50"):
        return DataSource.KRX

    # 암호화폐 (USDT 페어)
    if "USDT" in s or "/USDT" in s or "/USD" in s:
        return DataSource.BINANCE

    # 주요 암호화폐 심볼
    crypto_symbols = {
        "BTC", "ETH", "BNB", "SOL", "ADA", "XRP", "DOT",
        "AVAX", "MATIC", "LINK", "DOGE", "SHIB", "LTC",
    }
    if s in crypto_symbols:
        return DataSource.COINGECKO

    # FRED 경제지표 (알파벳+숫자 조합, 짧음)
    fred_patterns = ["DFF", "CPIAUCSL", "UNRATE", "GDP", "T10Y2Y", "VIXCLS"]
    if s in fred_patterns:
        return DataSource.FRED

    # 그 외 → Yahoo Finance (미국 주식, ETF, 지수 등)
    return DataSource.YAHOO


class DataManager:
    """
    통합 데이터 수집 관리자.

    사용법:
        dm = DataManager()

        # 자동 소스 감지
        aapl = await dm.fetch("AAPL", Timeframe.ONE_DAY, start=datetime(2020,1,1))
        samsung = await dm.fetch("005930", Timeframe.ONE_DAY, start=datetime(2020,1,1))
        btc = await dm.fetch("BTCUSDT", Timeframe.ONE_DAY, start=datetime(2020,1,1))

        # 여러 종목 동시 수집
        portfolio = await dm.fetch_multiple(
            ["AAPL", "MSFT", "005930", "000660", "BTCUSDT"],
            Timeframe.ONE_DAY,
            start=datetime(2020,1,1),
        )
    """

    def __init__(
        self,
        alpha_vantage_key: Optional[str] = None,
        fred_key: Optional[str] = None,
    ) -> None:
        self._collectors: dict[DataSource, Any] = {}
        self._alpha_vantage_key = alpha_vantage_key
        self._fred_key = fred_key

    def _get_collector(self, source: DataSource) -> Any:
        """수집기 지연 초기화 (필요할 때만 생성)"""
        if source not in self._collectors:
            if source == DataSource.YAHOO:
                from app.data.collectors.yahoo_collector import YahooFinanceCollector
                self._collectors[source] = YahooFinanceCollector()

            elif source == DataSource.KRX:
                from app.data.collectors.krx_collector import KRXCollector
                self._collectors[source] = KRXCollector()

            elif source == DataSource.BINANCE:
                from app.data.collectors.binance_collector import BinanceCollector
                # 키 없이 공개 API 사용
                self._collectors[source] = BinanceCollector(api_key=None, secret_key=None)

            elif source == DataSource.COINGECKO:
                from app.data.collectors.coingecko_collector import CoinGeckoCollector
                self._collectors[source] = CoinGeckoCollector()

            elif source == DataSource.ALPHA_VANTAGE:
                if not self._alpha_vantage_key:
                    logger.warning(
                        "alpha_vantage_key_missing",
                        hint="발급: https://www.alphavantage.co/support/#api-key",
                    )
                    # 키 없으면 Yahoo로 폴백
                    from app.data.collectors.yahoo_collector import YahooFinanceCollector
                    self._collectors[source] = YahooFinanceCollector()
                else:
                    from app.data.collectors.alpha_vantage_collector import AlphaVantageCollector
                    self._collectors[source] = AlphaVantageCollector(self._alpha_vantage_key)

            elif source == DataSource.FRED:
                if not self._fred_key:
                    logger.warning(
                        "fred_key_missing",
                        hint="발급: https://fred.stlouisfed.org/docs/api/api_key.html",
                    )
                    return None
                from app.data.collectors.fred_collector import FREDCollector
                self._collectors[source] = FREDCollector(self._fred_key)

        return self._collectors.get(source)

    async def fetch(
        self,
        symbol: Symbol,
        timeframe: Timeframe = Timeframe.ONE_DAY,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: int = 2000,
        source: Optional[DataSource] = None,
    ) -> OHLCVFrame:
        """
        단일 종목 데이터 수집.

        소스를 지정하지 않으면 심볼 형식에서 자동 감지.

        Args:
            symbol: 종목 코드
            timeframe: 타임프레임
            start: 시작일 (None이면 2년 전)
            end: 종료일 (None이면 오늘)
            limit: 최대 레코드 수
            source: 강제 소스 지정 (None이면 자동)

        Examples:
            # 미국 주식 (Yahoo)
            await dm.fetch("AAPL", Timeframe.ONE_DAY, datetime(2020,1,1))

            # 한국 주식 (KRX)
            await dm.fetch("005930", Timeframe.ONE_DAY, datetime(2020,1,1))

            # 암호화폐 실시간 (Binance)
            await dm.fetch("BTCUSDT", Timeframe.ONE_HOUR, datetime(2023,1,1))

            # 코스피 지수
            await dm.fetch("KS11", Timeframe.ONE_DAY, datetime(2020,1,1))
        """
        start = start or (datetime.now() - timedelta(days=730))
        end = end or datetime.now()
        source = source or detect_source(symbol)

        collector = self._get_collector(source)
        if collector is None:
            logger.error("no_collector_available", symbol=symbol, source=source)
            from app.data.collectors.yahoo_collector import YahooFinanceCollector
            collector = YahooFinanceCollector()

        with PerformanceLogger("data_manager_fetch", symbol=symbol, source=source.value):
            data = await collector.fetch_ohlcv(symbol, timeframe, start, end, limit)

        logger.info(
            "data_fetched",
            symbol=symbol,
            source=source.value,
            timeframe=timeframe.value,
            bars=data.n_bars,
        )
        return data

    async def fetch_multiple(
        self,
        symbols: list[Symbol],
        timeframe: Timeframe = Timeframe.ONE_DAY,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> dict[Symbol, OHLCVFrame]:
        """
        여러 종목 동시 수집.
        소스별로 그룹핑하여 배치 처리.

        Examples:
            portfolio = await dm.fetch_multiple(
                ["AAPL", "MSFT", "GOOGL",        # → Yahoo
                 "005930", "000660",               # → KRX
                 "BTCUSDT", "ETHUSDT"],            # → Binance
                Timeframe.ONE_DAY,
                start=datetime(2022, 1, 1),
            )
        """
        import asyncio

        start = start or (datetime.now() - timedelta(days=730))
        end = end or datetime.now()

        # 소스별 그룹핑
        source_groups: dict[DataSource, list[Symbol]] = {}
        for symbol in symbols:
            source = detect_source(symbol)
            source_groups.setdefault(source, []).append(symbol)

        logger.info(
            "batch_fetch_started",
            total_symbols=len(symbols),
            sources={k.value: len(v) for k, v in source_groups.items()},
        )

        # 소스별 배치 수집 (병렬)
        all_tasks = []
        for source, source_symbols in source_groups.items():
            for symbol in source_symbols:
                all_tasks.append(self.fetch(symbol, timeframe, start, end, source=source))

        results_list = await asyncio.gather(*all_tasks, return_exceptions=True)

        results: dict[Symbol, OHLCVFrame] = {}
        for symbol, result in zip(symbols, results_list):
            if isinstance(result, Exception):
                logger.error("fetch_failed", symbol=symbol, error=str(result))
            else:
                results[symbol] = result

        logger.info(
            "batch_fetch_completed",
            requested=len(symbols),
            succeeded=len(results),
            failed=len(symbols) - len(results),
        )

        return results

    def get_source_for_symbol(self, symbol: Symbol) -> DataSource:
        """심볼의 데이터 소스 반환"""
        return detect_source(symbol)

    def get_status(self) -> dict[str, Any]:
        """수집기 상태 조회"""
        return {
            "active_collectors": [s.value for s in self._collectors],
            "has_alpha_vantage": bool(self._alpha_vantage_key),
            "has_fred": bool(self._fred_key),
        }


# ─────────────────────────────────────────────
# 글로벌 싱글턴
# ─────────────────────────────────────────────
_data_manager: Optional[DataManager] = None


def get_data_manager(
    alpha_vantage_key: Optional[str] = None,
    fred_key: Optional[str] = None,
) -> DataManager:
    """DataManager 싱글턴 반환"""
    global _data_manager
    if _data_manager is None:
        from app.config.settings import get_settings
        cfg = get_settings()
        _data_manager = DataManager(
            alpha_vantage_key=getattr(cfg.api_keys, "alpha_vantage_key", None) or alpha_vantage_key,
            fred_key=getattr(cfg.api_keys, "fred_api_key", None) or fred_key,
        )
    return _data_manager
