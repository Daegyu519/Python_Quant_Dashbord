"""
=============================================================================
Binance Data Collector — WebSocket Real-Time + REST Historical
=============================================================================
바이낸스 실시간 WebSocket + REST API 기반 데이터 수집기.
재연결 로직, 핑/퐁 유지, 메시지 큐 처리 포함.
=============================================================================
"""

from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timezone
from typing import Any, AsyncGenerator, Optional

import ssl
import certifi
import aiohttp
import websockets
from tenacity import (
    retry, retry_if_exception_type, stop_after_attempt, wait_exponential
)

import numpy as np
import pandas as pd

from app.core.base_classes import BaseDataCollector
from app.core.types import (
    OHLCV, OrderBook, OrderBookLevel, OHLCVFrame,
    Symbol, Tick, Timeframe, OrderSide,
)
from app.config.logging_config import get_logger, PerformanceLogger
from app.config.settings import get_settings

logger = get_logger(__name__)
settings = get_settings()

# 바이낸스 타임프레임 매핑
BINANCE_TIMEFRAME_MAP: dict[Timeframe, str] = {
    Timeframe.ONE_MIN: "1m",
    Timeframe.THREE_MIN: "3m",
    Timeframe.FIVE_MIN: "5m",
    Timeframe.FIFTEEN_MIN: "15m",
    Timeframe.THIRTY_MIN: "30m",
    Timeframe.ONE_HOUR: "1h",
    Timeframe.TWO_HOUR: "2h",
    Timeframe.FOUR_HOUR: "4h",
    Timeframe.SIX_HOUR: "6h",
    Timeframe.TWELVE_HOUR: "12h",
    Timeframe.ONE_DAY: "1d",
    Timeframe.THREE_DAY: "3d",
    Timeframe.ONE_WEEK: "1w",
    Timeframe.ONE_MONTH: "1M",
}

# API 엔드포인트
BINANCE_REST_URL = "https://api.binance.com"
BINANCE_TESTNET_REST_URL = "https://testnet.binance.vision"
BINANCE_WS_URL = "wss://stream.binance.com:9443/ws"
BINANCE_WS_COMBINED = "wss://stream.binance.com:9443/stream"


class BinanceCollector(BaseDataCollector):
    """
    바이낸스 데이터 수집기.

    특징:
    - REST API: 과거 OHLCV, 현재 가격, 호가창
    - WebSocket: 실시간 틱, 실시간 OHLCV, 실시간 호가창
    - 자동 재연결 (최대 5회)
    - Rate Limit 자동 관리 (1200 req/min)
    - 비동기 메시지 큐
    """

    # 바이낸스 Rate Limit
    REST_WEIGHT_LIMIT = 1200  # per minute
    REST_ORDER_LIMIT = 100    # per 10 seconds

    def __init__(
        self,
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        testnet: bool = False,
    ) -> None:
        super().__init__(source_name="binance")
        self._api_key = api_key or settings.api_keys.binance_api_key
        self._secret_key = secret_key or settings.api_keys.binance_secret_key
        self._testnet = testnet

        self._base_url = BINANCE_TESTNET_REST_URL if testnet else BINANCE_REST_URL
        self._session: Optional[aiohttp.ClientSession] = None
        self._ws_connections: dict[str, Any] = {}

        # Rate Limit 관리
        self._request_timestamps: list[float] = []
        self._weight_used: int = 0

    async def connect(self) -> None:
        """HTTP 세션 초기화"""
        headers = {}
        if self._api_key:
            headers["X-MBX-APIKEY"] = self._api_key

        ssl_ctx = ssl.create_default_context(cafile=certifi.where())
        self._session = aiohttp.ClientSession(
            base_url=self._base_url,
            headers=headers,
            connector=aiohttp.TCPConnector(
                ssl=ssl_ctx,
                limit=100,
                ttl_dns_cache=300,
                enable_cleanup_closed=True,
            ),
            timeout=aiohttp.ClientTimeout(total=30),
        )
        self._is_connected = True
        logger.info("binance_connected", testnet=self._testnet)

    async def disconnect(self) -> None:
        """세션 & WebSocket 종료"""
        if self._session and not self._session.closed:
            await self._session.close()

        for ws in self._ws_connections.values():
            await ws.close()

        self._is_connected = False
        logger.info("binance_disconnected")

    async def _rate_limit_check(self, weight: int = 1) -> None:
        """Rate Limit 준수"""
        now = time.monotonic()
        # 1분 윈도우 유지
        self._request_timestamps = [
            ts for ts in self._request_timestamps if now - ts < 60
        ]

        self._weight_used += weight
        if self._weight_used >= self.REST_WEIGHT_LIMIT * 0.9:  # 90% 도달시 대기
            wait_time = 60 - (now - self._request_timestamps[0]) if self._request_timestamps else 1
            logger.warning("rate_limit_approaching", wait_seconds=wait_time)
            await asyncio.sleep(wait_time)
            self._weight_used = 0

        self._request_timestamps.append(now)

    @retry(
        retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError)),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=30),
    )
    async def _get(self, endpoint: str, params: dict = None, weight: int = 1) -> Any:
        """REST GET 요청"""
        await self._rate_limit_check(weight)

        async with self._session.get(endpoint, params=params) as resp:
            if resp.status == 429:
                retry_after = int(resp.headers.get("Retry-After", 60))
                logger.warning("rate_limited", retry_after=retry_after)
                await asyncio.sleep(retry_after)
                raise aiohttp.ClientError("Rate limited")

            resp.raise_for_status()
            return await resp.json()

    async def fetch_ohlcv(
        self,
        symbol: Symbol,
        timeframe: Timeframe,
        start: datetime,
        end: Optional[datetime] = None,
        limit: int = 1000,
    ) -> OHLCVFrame:
        """
        바이낸스 OHLCV 수집.

        바이낸스는 한 번에 최대 1000개 반환.
        1000개 초과 시 자동 페이지네이션.
        """
        if not self._session:
            await self.connect()

        binance_interval = BINANCE_TIMEFRAME_MAP.get(timeframe, "1d")
        start_ms = int(start.timestamp() * 1000)
        end_ms = int((end or datetime.utcnow()).timestamp() * 1000)

        all_klines = []
        current_start = start_ms

        with PerformanceLogger("binance_fetch_ohlcv", symbol=symbol):
            while current_start < end_ms:
                params = {
                    "symbol": symbol,
                    "interval": binance_interval,
                    "startTime": current_start,
                    "endTime": end_ms,
                    "limit": min(limit - len(all_klines), 1000),
                }

                klines = await self._get("/api/v3/klines", params, weight=1)

                if not klines:
                    break

                all_klines.extend(klines)

                if len(klines) < 1000:
                    break

                # 다음 페이지 시작점
                current_start = klines[-1][0] + 1

                if len(all_klines) >= limit:
                    break

        if not all_klines:
            logger.warning("binance_empty_data", symbol=symbol)
            return self._empty_frame(symbol, timeframe)

        df = self._klines_to_dataframe(all_klines)

        logger.info(
            "binance_data_collected",
            symbol=symbol,
            timeframe=timeframe.value,
            bars=len(df),
        )

        return OHLCVFrame(symbol=symbol, timeframe=timeframe, data=df)

    def _klines_to_dataframe(self, klines: list) -> pd.DataFrame:
        """바이낸스 klines 응답을 DataFrame으로 변환"""
        df = pd.DataFrame(klines, columns=[
            "timestamp", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "num_trades",
            "taker_buy_base", "taker_buy_quote", "ignore"
        ])

        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.set_index("timestamp")

        # 숫자 변환
        numeric_cols = ["open", "high", "low", "close", "volume",
                       "quote_volume", "taker_buy_base"]
        df[numeric_cols] = df[numeric_cols].astype(np.float64)
        df["num_trades"] = df["num_trades"].astype(int)

        return df[["open", "high", "low", "close", "volume"]].copy()

    def _empty_frame(self, symbol: Symbol, timeframe: Timeframe) -> OHLCVFrame:
        empty_df = pd.DataFrame(
            columns=["open", "high", "low", "close", "volume"],
            index=pd.DatetimeIndex([], tz="UTC"),
        )
        return OHLCVFrame(symbol=symbol, timeframe=timeframe, data=empty_df)

    async def fetch_ticker(self, symbol: Symbol) -> dict[str, Any]:
        """현재 가격 (24시간 통계 포함)"""
        data = await self._get("/api/v3/ticker/24hr", {"symbol": symbol}, weight=1)
        return {
            "symbol": symbol,
            "price": float(data["lastPrice"]),
            "volume": float(data["volume"]),
            "price_change_pct": float(data["priceChangePercent"]),
            "high_24h": float(data["highPrice"]),
            "low_24h": float(data["lowPrice"]),
        }

    async def fetch_orderbook(self, symbol: Symbol, depth: int = 20) -> OrderBook:
        """실시간 호가창"""
        data = await self._get("/api/v3/depth", {"symbol": symbol, "limit": depth}, weight=1)

        bids = [
            OrderBookLevel(price=float(p), quantity=float(q))
            for p, q in data["bids"]
        ]
        asks = [
            OrderBookLevel(price=float(p), quantity=float(q))
            for p, q in data["asks"]
        ]

        return OrderBook(
            symbol=symbol,
            timestamp=datetime.utcnow(),
            bids=bids,
            asks=asks,
        )

    async def stream_ticks(
        self, symbols: list[Symbol]
    ) -> AsyncGenerator[Tick, None]:
        """
        실시간 틱 스트리밍 (WebSocket).

        Combined Stream 사용으로 단일 연결에서 다수 심볼 처리.
        자동 재연결 포함.
        """
        streams = [f"{s.lower()}@aggTrade" for s in symbols]
        url = f"{BINANCE_WS_COMBINED}?streams={'/'.join(streams)}"

        reconnect_delay = 1.0

        while True:
            try:
                async with websockets.connect(
                    url,
                    ping_interval=20,
                    ping_timeout=10,
                    close_timeout=10,
                ) as ws:
                    reconnect_delay = 1.0
                    logger.info("binance_ws_connected", symbols=symbols)

                    async for message in ws:
                        try:
                            data = json.loads(message)
                            stream_data = data.get("data", data)

                            yield Tick(
                                symbol=stream_data["s"],
                                timestamp=datetime.fromtimestamp(
                                    stream_data["T"] / 1000, tz=timezone.utc
                                ),
                                price=float(stream_data["p"]),
                                volume=float(stream_data["q"]),
                                side=OrderSide.BUY if stream_data["m"] else OrderSide.SELL,
                                trade_id=str(stream_data.get("a", "")),
                            )

                        except (KeyError, ValueError) as e:
                            logger.error("tick_parse_error", error=str(e))
                            continue

            except (websockets.exceptions.ConnectionClosed,
                    websockets.exceptions.WebSocketException) as e:
                logger.warning(
                    "ws_disconnected",
                    error=str(e),
                    reconnect_in=reconnect_delay,
                )
                await asyncio.sleep(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 2, 60)  # 최대 60초

    async def stream_ohlcv(
        self, symbol: Symbol, timeframe: Timeframe
    ) -> AsyncGenerator[OHLCV, None]:
        """실시간 OHLCV 캔들 스트리밍"""
        binance_interval = BINANCE_TIMEFRAME_MAP.get(timeframe, "1m")
        url = f"{BINANCE_WS_URL}/{symbol.lower()}@kline_{binance_interval}"

        async with websockets.connect(url, ping_interval=20) as ws:
            async for message in ws:
                data = json.loads(message)
                k = data.get("k", {})

                if k.get("x", False):  # 완성된 캔들만
                    yield OHLCV(
                        symbol=symbol,
                        timestamp=datetime.fromtimestamp(k["t"] / 1000, tz=timezone.utc),
                        open=float(k["o"]),
                        high=float(k["h"]),
                        low=float(k["l"]),
                        close=float(k["c"]),
                        volume=float(k["v"]),
                        timeframe=timeframe,
                    )

    def get_supported_symbols(self) -> list[Symbol]:
        """지원 심볼 목록 (동기 버전)"""
        # 주요 USDT 페어 목록 반환
        return [
            "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "ADAUSDT",
            "XRPUSDT", "DOTUSDT", "AVAXUSDT", "MATICUSDT", "LINKUSDT",
        ]

    async def get_exchange_info(self) -> dict[str, Any]:
        """거래소 정보 및 지원 심볼 전체 목록"""
        return await self._get("/api/v3/exchangeInfo", weight=10)
