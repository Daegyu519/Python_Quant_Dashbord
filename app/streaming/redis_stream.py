"""
=============================================================================
Redis Streaming — Real-Time Market Data Pipeline
=============================================================================
Redis Pub/Sub + Streams 기반 실시간 데이터 파이프라인.
초저지연 틱 데이터 분배 + 시그널 전파.
=============================================================================
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from typing import Any, AsyncGenerator, Callable, Optional

import redis.asyncio as aioredis

from app.config.settings import get_settings
from app.config.logging_config import get_logger
from app.core.types import OHLCV, Signal, Tick

logger = get_logger(__name__)
settings = get_settings()

# ─────────────────────────────────────────────
# 채널 이름 상수
# ─────────────────────────────────────────────
class RedisChannels:
    """Redis 채널 이름 관리"""
    TICKS = "market:ticks:{symbol}"
    OHLCV = "market:ohlcv:{symbol}:{timeframe}"
    SIGNALS = "signals:{strategy_id}"
    ALL_SIGNALS = "signals:all"
    PORTFOLIO_UPDATE = "portfolio:{portfolio_id}"
    RISK_ALERT = "risk:alerts"
    SYSTEM_EVENTS = "system:events"

    @staticmethod
    def tick(symbol: str) -> str:
        return f"market:ticks:{symbol}"

    @staticmethod
    def ohlcv(symbol: str, timeframe: str) -> str:
        return f"market:ohlcv:{symbol}:{timeframe}"

    @staticmethod
    def signal(strategy_id: str) -> str:
        return f"signals:{strategy_id}"


class RedisStreamManager:
    """
    Redis 스트리밍 매니저.

    기능:
    - 틱 데이터 Pub/Sub
    - OHLCV 실시간 분배
    - 시그널 전파
    - 포트폴리오 업데이트
    - 캐싱 (최신 가격, 최신 시그널)
    """

    def __init__(self) -> None:
        self._pubsub_client: Optional[aioredis.Redis] = None
        self._cache_client: Optional[aioredis.Redis] = None
        self._stream_client: Optional[aioredis.Redis] = None

    async def connect(self) -> None:
        """Redis 연결 초기화"""
        cfg = settings.redis

        # 목적별 연결 풀 분리
        self._pubsub_client = await aioredis.from_url(
            cfg.url,
            encoding="utf-8",
            decode_responses=True,
            max_connections=20,
        )
        self._cache_client = await aioredis.from_url(
            f"redis://{cfg.host}:{cfg.port}/{cfg.cache_db}",
            encoding="utf-8",
            decode_responses=True,
            max_connections=50,
        )
        self._stream_client = await aioredis.from_url(
            f"redis://{cfg.host}:{cfg.port}/{cfg.stream_db}",
            encoding="utf-8",
            decode_responses=True,
            max_connections=30,
        )

        logger.info("redis_connected")

    async def disconnect(self) -> None:
        """Redis 연결 종료"""
        for client in [self._pubsub_client, self._cache_client, self._stream_client]:
            if client:
                await client.aclose()
        logger.info("redis_disconnected")

    # ─────────────────────────────────────────────
    # 틱 데이터 발행/구독
    # ─────────────────────────────────────────────

    async def publish_tick(self, tick: Tick) -> None:
        """틱 데이터 발행"""
        channel = RedisChannels.tick(tick.symbol)
        payload = json.dumps({
            "symbol": tick.symbol,
            "price": tick.price,
            "volume": tick.volume,
            "side": tick.side.value,
            "timestamp": tick.timestamp.isoformat(),
        })

        await self._pubsub_client.publish(channel, payload)

        # 최신 가격 캐시 (5초 TTL)
        await self._cache_client.setex(
            f"price:{tick.symbol}",
            settings.redis.ticker_cache_ttl,
            str(tick.price),
        )

    async def subscribe_ticks(
        self,
        symbols: list[str],
        callback: Callable[[Tick], None],
    ) -> None:
        """틱 데이터 구독"""
        channels = [RedisChannels.tick(s) for s in symbols]
        pubsub = self._pubsub_client.pubsub()

        await pubsub.subscribe(*channels)

        logger.info("tick_subscribed", symbols=symbols)

        async for message in pubsub.listen():
            if message["type"] == "message":
                try:
                    data = json.loads(message["data"])
                    tick = Tick(
                        symbol=data["symbol"],
                        timestamp=datetime.fromisoformat(data["timestamp"]),
                        price=float(data["price"]),
                        volume=float(data["volume"]),
                        side=data["side"],
                    )
                    callback(tick)
                except Exception as e:
                    logger.error("tick_parse_error", error=str(e))

    # ─────────────────────────────────────────────
    # 시그널 발행/구독
    # ─────────────────────────────────────────────

    async def publish_signal(self, signal: Signal) -> None:
        """시그널 발행"""
        payload = json.dumps({
            "signal_id": signal.signal_id,
            "strategy_id": signal.strategy_id,
            "symbol": signal.symbol,
            "direction": signal.direction.value,
            "strength": signal.strength,
            "confidence": signal.confidence,
            "timestamp": signal.timestamp.isoformat(),
            "source": signal.source.value,
            "metadata": signal.metadata,
        })

        # 전략별 채널 + 전체 채널에 동시 발행
        pipeline = self._pubsub_client.pipeline()
        pipeline.publish(RedisChannels.signal(signal.strategy_id), payload)
        pipeline.publish(RedisChannels.ALL_SIGNALS, payload)

        # 최신 시그널 캐시 (30초 TTL)
        pipeline.setex(
            f"signal:latest:{signal.symbol}:{signal.strategy_id}",
            settings.redis.signal_cache_ttl,
            payload,
        )

        await pipeline.execute()

    async def get_latest_signal(
        self, symbol: str, strategy_id: str
    ) -> Optional[dict]:
        """최신 시그널 캐시 조회"""
        data = await self._cache_client.get(
            f"signal:latest:{symbol}:{strategy_id}"
        )
        return json.loads(data) if data else None

    # ─────────────────────────────────────────────
    # 가격 캐시
    # ─────────────────────────────────────────────

    async def get_latest_price(self, symbol: str) -> Optional[float]:
        """최신 가격 캐시 조회"""
        price = await self._cache_client.get(f"price:{symbol}")
        return float(price) if price else None

    async def get_multiple_prices(
        self, symbols: list[str]
    ) -> dict[str, Optional[float]]:
        """여러 종목 최신 가격 일괄 조회"""
        keys = [f"price:{s}" for s in symbols]
        values = await self._cache_client.mget(keys)

        return {
            symbol: float(v) if v else None
            for symbol, v in zip(symbols, values)
        }

    # ─────────────────────────────────────────────
    # Redis Streams (지속성 있는 스트림)
    # ─────────────────────────────────────────────

    async def add_to_stream(
        self,
        stream_name: str,
        data: dict[str, Any],
        maxlen: int = 10_000,
    ) -> str:
        """Redis Stream에 데이터 추가"""
        msg_id = await self._stream_client.xadd(
            stream_name,
            {k: json.dumps(v) if not isinstance(v, str) else v for k, v in data.items()},
            maxlen=maxlen,
            approximate=True,
        )
        return msg_id

    async def read_stream(
        self,
        stream_name: str,
        last_id: str = "0",
        count: int = 100,
        block_ms: int = 1000,
    ) -> list[tuple[str, dict]]:
        """Redis Stream 읽기"""
        messages = await self._stream_client.xread(
            {stream_name: last_id},
            count=count,
            block=block_ms,
        )

        if not messages:
            return []

        results = []
        for _, msgs in messages:
            for msg_id, data in msgs:
                results.append((
                    msg_id,
                    {k: json.loads(v) if v.startswith("{") or v.startswith("[") else v
                     for k, v in data.items()}
                ))

        return results

    # ─────────────────────────────────────────────
    # Rate Limit 지원
    # ─────────────────────────────────────────────

    async def check_rate_limit(
        self,
        key: str,
        limit: int,
        window_seconds: int = 60,
    ) -> tuple[bool, int]:
        """
        슬라이딩 윈도우 Rate Limit.

        Returns:
            (allowed, remaining)
        """
        pipe = self._cache_client.pipeline()
        now = asyncio.get_event_loop().time()
        window_start = now - window_seconds

        # Sorted Set으로 슬라이딩 윈도우 구현
        pipe.zremrangebyscore(key, 0, window_start)
        pipe.zadd(key, {str(now): now})
        pipe.zcard(key)
        pipe.expire(key, window_seconds)

        results = await pipe.execute()
        count = results[2]

        allowed = count <= limit
        remaining = max(0, limit - count)

        return allowed, remaining


# ─────────────────────────────────────────────
# 글로벌 싱글턴
# ─────────────────────────────────────────────
_stream_manager: Optional[RedisStreamManager] = None


async def get_stream_manager() -> RedisStreamManager:
    """Redis 스트림 매니저 싱글턴"""
    global _stream_manager
    if _stream_manager is None:
        _stream_manager = RedisStreamManager()
        await _stream_manager.connect()
    return _stream_manager
