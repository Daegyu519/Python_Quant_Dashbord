"""
=============================================================================
CoinGecko 암호화폐 데이터 수집기 — 완전 무료 (가입/키 불필요)
=============================================================================
CoinGecko 공개 API 기반.
시가총액, 거래량, 도미넌스 등 온체인 지표 포함.

무료 한도: ~10-30 req/min (키 없이)
=============================================================================
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, AsyncGenerator, Optional

import ssl
import certifi
import aiohttp
import pandas as pd
import numpy as np
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.core.base_classes import BaseDataCollector
from app.core.types import (
    OHLCV, OrderBook, OHLCVFrame, Symbol, Tick, Timeframe,
)
from app.config.logging_config import get_logger, PerformanceLogger

logger = get_logger(__name__)

COINGECKO_BASE_URL = "https://api.coingecko.com/api/v3/"

# CoinGecko ID ↔ 심볼 매핑
COINGECKO_IDS: dict[str, str] = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "BNB": "binancecoin",
    "SOL": "solana",
    "ADA": "cardano",
    "XRP": "ripple",
    "DOT": "polkadot",
    "AVAX": "avalanche-2",
    "MATIC": "matic-network",
    "LINK": "chainlink",
    "DOGE": "dogecoin",
    "SHIB": "shiba-inu",
    "LTC": "litecoin",
    "ATOM": "cosmos",
    "UNI": "uniswap",
}

# 일수 → CoinGecko days 파라미터 매핑
TIMEFRAME_TO_DAYS: dict[Timeframe, int] = {
    Timeframe.ONE_DAY: 1,
    Timeframe.ONE_HOUR: 1,
    Timeframe.FOUR_HOUR: 7,
    Timeframe.ONE_DAY: 365,
}


class CoinGeckoCollector(BaseDataCollector):
    """
    CoinGecko 암호화폐 데이터 수집기.

    ✅ 완전 무료 — 가입/키 불필요
    ✅ 시가총액, 거래량 포함
    ✅ 1일~최대 365일 OHLCV 지원

    한계:
      - 무료 rate limit: ~10-30 req/min
      - 실시간 스트리밍 없음 (Binance 사용 권장)
      - 1분봉 없음 (일봉/시간봉만)
    """

    def __init__(self) -> None:
        super().__init__(source_name="coingecko")
        self._session: Optional[aiohttp.ClientSession] = None
        self._semaphore = asyncio.Semaphore(3)  # 동시 요청 3개 제한

    async def connect(self) -> None:
        ssl_ctx = ssl.create_default_context(cafile=certifi.where())
        self._session = aiohttp.ClientSession(
            base_url=COINGECKO_BASE_URL,
            headers={"Accept": "application/json"},
            connector=aiohttp.TCPConnector(ssl=ssl_ctx, limit=10),
            timeout=aiohttp.ClientTimeout(total=30),
        )
        self._is_connected = True
        logger.info("coingecko_connected")

    async def disconnect(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
        self._is_connected = False

    @retry(
        retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(min=2, max=30),
    )
    async def _get(self, endpoint: str, params: dict = None) -> Any:
        """Rate-limited GET 요청"""
        async with self._semaphore:
            async with self._session.get(endpoint, params=params) as resp:
                if resp.status == 429:  # Too Many Requests
                    retry_after = int(resp.headers.get("Retry-After", 60))
                    logger.warning("coingecko_rate_limited", retry_after=retry_after)
                    await asyncio.sleep(retry_after)
                    raise aiohttp.ClientError("Rate limited")
                resp.raise_for_status()
                return await resp.json()

    def _symbol_to_id(self, symbol: Symbol) -> str:
        """심볼을 CoinGecko ID로 변환"""
        # "BTC/USDT" → "BTC" → "bitcoin"
        clean = symbol.replace("/USDT", "").replace("/USD", "").replace("USDT", "").upper()
        return COINGECKO_IDS.get(clean, clean.lower())

    async def fetch_ohlcv(
        self,
        symbol: Symbol,
        timeframe: Timeframe,
        start: datetime,
        end: Optional[datetime] = None,
        limit: int = 365,
    ) -> OHLCVFrame:
        """
        CoinGecko OHLCV 수집.

        CoinGecko는 days 파라미터로 조회:
          - 1일: 5분봉
          - 2~90일: 1시간봉
          - 91일+: 일봉
        """
        if not self._session:
            await self.connect()

        coin_id = self._symbol_to_id(symbol)
        end = end or datetime.now(tz=timezone.utc)

        # 기간 계산
        days = (end - start.replace(tzinfo=timezone.utc) if start.tzinfo else end - start.replace(tzinfo=timezone.utc)).days
        days = max(1, min(days, 365))

        with PerformanceLogger("coingecko_fetch_ohlcv", symbol=symbol, days=days):
            data = await self._get(
                f"coins/{coin_id}/ohlc",
                params={"vs_currency": "usd", "days": str(days)},
            )

        if not data:
            logger.warning("coingecko_empty_data", symbol=symbol)
            return self._empty_frame(symbol, timeframe)

        df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.set_index("timestamp")
        df["volume"] = 0.0  # OHLC 엔드포인트는 거래량 없음

        # 거래량은 market_chart로 별도 조회
        try:
            volume_data = await self._get(
                f"coins/{coin_id}/market_chart",
                params={"vs_currency": "usd", "days": str(days)},
            )
            if volume_data and "total_volumes" in volume_data:
                vol_df = pd.DataFrame(
                    volume_data["total_volumes"],
                    columns=["timestamp", "volume"]
                )
                vol_df["timestamp"] = pd.to_datetime(vol_df["timestamp"], unit="ms", utc=True)
                vol_df = vol_df.set_index("timestamp").resample("4h").last()
                df["volume"] = vol_df["volume"].reindex(df.index, method="nearest")
        except Exception:
            pass  # 거래량 없어도 동작

        df = df.dropna(subset=["open", "high", "low", "close"])

        logger.info(
            "coingecko_data_collected",
            symbol=symbol,
            coin_id=coin_id,
            bars=len(df),
        )

        return OHLCVFrame(symbol=symbol, timeframe=timeframe, data=df)

    async def get_market_overview(self) -> dict[str, Any]:
        """
        전체 암호화폐 시장 개요.
        BTC 도미넌스, 총 시가총액 등.
        """
        if not self._session:
            await self.connect()

        data = await self._get("global")
        return data.get("data", {})

    async def get_top_coins(self, n: int = 100) -> pd.DataFrame:
        """
        시가총액 상위 N개 코인 정보.
        팩터 분석에 활용.
        """
        if not self._session:
            await self.connect()

        data = await self._get(
            "coins/markets",
            params={
                "vs_currency": "usd",
                "order": "market_cap_desc",
                "per_page": min(n, 250),
                "page": 1,
                "sparkline": False,
                "price_change_percentage": "1h,24h,7d,30d",
            },
        )

        df = pd.DataFrame(data)
        return df

    async def fetch_ticker(self, symbol: Symbol) -> dict[str, Any]:
        """현재 가격"""
        if not self._session:
            await self.connect()

        coin_id = self._symbol_to_id(symbol)
        data = await self._get(
            "simple/price",
            params={
                "ids": coin_id,
                "vs_currencies": "usd",
                "include_market_cap": "true",
                "include_24hr_vol": "true",
                "include_24hr_change": "true",
            },
        )
        info = data.get(coin_id, {})
        return {
            "symbol": symbol,
            "price": info.get("usd", 0.0),
            "market_cap": info.get("usd_market_cap", 0.0),
            "volume_24h": info.get("usd_24h_vol", 0.0),
            "change_24h": info.get("usd_24h_change", 0.0),
        }

    def _empty_frame(self, symbol: Symbol, timeframe: Timeframe) -> OHLCVFrame:
        empty = pd.DataFrame(
            columns=["open", "high", "low", "close", "volume"],
            index=pd.DatetimeIndex([], tz="UTC"),
        )
        return OHLCVFrame(symbol=symbol, timeframe=timeframe, data=empty)

    async def fetch_orderbook(self, symbol: Symbol, depth: int = 20) -> OrderBook:
        from app.core.types import OrderBook
        logger.warning("coingecko_orderbook_not_supported")
        return OrderBook(symbol=symbol, timestamp=datetime.utcnow(), bids=[], asks=[])

    async def stream_ticks(self, symbols: list[Symbol]) -> AsyncGenerator[Tick, None]:
        raise NotImplementedError("Use BinanceCollector for real-time streaming")

    async def stream_ohlcv(self, symbol: Symbol, timeframe: Timeframe) -> AsyncGenerator[OHLCV, None]:
        raise NotImplementedError("Use BinanceCollector for real-time streaming")

    def get_supported_symbols(self) -> list[Symbol]:
        return list(COINGECKO_IDS.keys())
