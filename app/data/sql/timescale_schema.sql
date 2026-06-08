-- =============================================================================
-- TimescaleDB Schema — Institution-Grade Time-Series Storage
-- =============================================================================
-- OHLCV, Tick, Orderbook 등 모든 시계열 데이터 최적화.
-- Hypertable + Compression + Continuous Aggregates 활용.
-- =============================================================================

-- 확장 기능 활성화
CREATE EXTENSION IF NOT EXISTS timescaledb;
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;  -- 쿼리 성능 모니터링

-- ─────────────────────────────────────────────
-- OHLCV 하이퍼테이블 (핵심 시계열 데이터)
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS ohlcv (
    time        TIMESTAMPTZ     NOT NULL,
    symbol      VARCHAR(30)     NOT NULL,
    timeframe   VARCHAR(10)     NOT NULL,
    open        DOUBLE PRECISION NOT NULL,
    high        DOUBLE PRECISION NOT NULL,
    low         DOUBLE PRECISION NOT NULL,
    close       DOUBLE PRECISION NOT NULL,
    volume      DOUBLE PRECISION NOT NULL,
    -- 파생 컬럼 (자주 사용되는 값 사전 계산)
    vwap        DOUBLE PRECISION,           -- 거래량 가중 평균가
    num_trades  INTEGER,                    -- 거래 횟수
    taker_buy_vol DOUBLE PRECISION,         -- 타커 매수 거래량
    source      VARCHAR(30) DEFAULT 'unknown',  -- 데이터 소스
    PRIMARY KEY (time, symbol, timeframe)
);

-- TimescaleDB 하이퍼테이블 변환
-- chunk_time_interval: 7일 단위 청크 (1일봉 데이터에 최적)
SELECT create_hypertable(
    'ohlcv',
    'time',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE
);

-- 파티셔닝 (symbol 기반 공간 파티셔닝)
SELECT add_dimension('ohlcv', 'symbol', number_partitions => 4, if_not_exists => TRUE);

-- ─────────────────────────────────────────────
-- 인덱스 최적화
-- ─────────────────────────────────────────────
CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol_time
    ON ohlcv (symbol, time DESC);

CREATE INDEX IF NOT EXISTS idx_ohlcv_timeframe_symbol_time
    ON ohlcv (timeframe, symbol, time DESC);

-- BRIN 인덱스 (순차 스캔 최적화, 대용량 데이터)
CREATE INDEX IF NOT EXISTS idx_ohlcv_time_brin
    ON ohlcv USING BRIN (time) WITH (pages_per_range = 32);

-- ─────────────────────────────────────────────
-- 압축 설정 (스토리지 60-90% 절약)
-- ─────────────────────────────────────────────
ALTER TABLE ohlcv SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'symbol, timeframe',  -- 압축 세그먼트
    timescaledb.compress_orderby = 'time DESC'              -- 압축 정렬
);

-- 7일 이상 데이터 자동 압축
SELECT add_compression_policy('ohlcv', INTERVAL '7 days', if_not_exists => TRUE);

-- ─────────────────────────────────────────────
-- 틱 데이터 하이퍼테이블
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS ticks (
    time        TIMESTAMPTZ     NOT NULL,
    symbol      VARCHAR(30)     NOT NULL,
    price       DOUBLE PRECISION NOT NULL,
    quantity    DOUBLE PRECISION NOT NULL,
    side        VARCHAR(10)     NOT NULL,    -- BUY, SELL
    trade_id    VARCHAR(50),
    is_market_maker BOOLEAN DEFAULT FALSE,
    source      VARCHAR(30)     DEFAULT 'unknown',
    PRIMARY KEY (time, symbol, trade_id)
) WITH (
    autovacuum_enabled = true,
    fillfactor = 90
);

SELECT create_hypertable(
    'ticks',
    'time',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

-- 틱 데이터 압축 (1시간 이상 경과 데이터)
ALTER TABLE ticks SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'symbol',
    timescaledb.compress_orderby = 'time DESC'
);
SELECT add_compression_policy('ticks', INTERVAL '1 hour', if_not_exists => TRUE);

-- ─────────────────────────────────────────────
-- 호가창 스냅샷
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS orderbook_snapshots (
    time        TIMESTAMPTZ     NOT NULL,
    symbol      VARCHAR(30)     NOT NULL,
    bid_price_1 DOUBLE PRECISION,
    bid_qty_1   DOUBLE PRECISION,
    bid_price_2 DOUBLE PRECISION,
    bid_qty_2   DOUBLE PRECISION,
    bid_price_3 DOUBLE PRECISION,
    bid_qty_3   DOUBLE PRECISION,
    bid_price_4 DOUBLE PRECISION,
    bid_qty_4   DOUBLE PRECISION,
    bid_price_5 DOUBLE PRECISION,
    bid_qty_5   DOUBLE PRECISION,
    ask_price_1 DOUBLE PRECISION,
    ask_qty_1   DOUBLE PRECISION,
    ask_price_2 DOUBLE PRECISION,
    ask_qty_2   DOUBLE PRECISION,
    ask_price_3 DOUBLE PRECISION,
    ask_qty_3   DOUBLE PRECISION,
    ask_price_4 DOUBLE PRECISION,
    ask_qty_4   DOUBLE PRECISION,
    ask_price_5 DOUBLE PRECISION,
    ask_qty_5   DOUBLE PRECISION,
    spread      DOUBLE PRECISION GENERATED ALWAYS AS (ask_price_1 - bid_price_1) STORED,
    mid_price   DOUBLE PRECISION GENERATED ALWAYS AS ((ask_price_1 + bid_price_1) / 2.0) STORED,
    PRIMARY KEY (time, symbol)
);

SELECT create_hypertable(
    'orderbook_snapshots',
    'time',
    chunk_time_interval => INTERVAL '1 hour',
    if_not_exists => TRUE
);

-- ─────────────────────────────────────────────
-- 포트폴리오 이력 (연속 추적)
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS portfolio_history (
    time            TIMESTAMPTZ     NOT NULL,
    portfolio_id    UUID            NOT NULL,
    total_value     DOUBLE PRECISION NOT NULL,
    cash            DOUBLE PRECISION NOT NULL,
    invested_value  DOUBLE PRECISION NOT NULL,
    pnl_daily       DOUBLE PRECISION DEFAULT 0.0,
    pnl_cumulative  DOUBLE PRECISION DEFAULT 0.0,
    drawdown        DOUBLE PRECISION DEFAULT 0.0,
    num_positions   INTEGER         DEFAULT 0,
    PRIMARY KEY (time, portfolio_id)
);

SELECT create_hypertable(
    'portfolio_history',
    'time',
    chunk_time_interval => INTERVAL '30 days',
    if_not_exists => TRUE
);

-- ─────────────────────────────────────────────
-- Continuous Aggregates (자동 갱신 뷰)
-- ─────────────────────────────────────────────

-- 일봉 집계 (1분봉 → 1일봉 자동 생성)
CREATE MATERIALIZED VIEW IF NOT EXISTS ohlcv_daily
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 day', time) AS day,
    symbol,
    FIRST(open, time)          AS open,
    MAX(high)                  AS high,
    MIN(low)                   AS low,
    LAST(close, time)          AS close,
    SUM(volume)                AS volume,
    SUM(volume * vwap) / NULLIF(SUM(volume), 0) AS vwap,
    COUNT(*)                   AS num_bars
FROM ohlcv
WHERE timeframe = '1m'
GROUP BY day, symbol
WITH NO DATA;

-- 연속 집계 자동 갱신 정책 (1분마다 갱신)
SELECT add_continuous_aggregate_policy(
    'ohlcv_daily',
    start_offset => INTERVAL '3 days',
    end_offset => INTERVAL '1 minute',
    schedule_interval => INTERVAL '1 minute',
    if_not_exists => TRUE
);

-- 주봉 집계
CREATE MATERIALIZED VIEW IF NOT EXISTS ohlcv_weekly
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 week', time) AS week,
    symbol,
    FIRST(open, time)           AS open,
    MAX(high)                   AS high,
    MIN(low)                    AS low,
    LAST(close, time)           AS close,
    SUM(volume)                 AS volume,
    COUNT(*)                    AS num_days
FROM ohlcv
WHERE timeframe = '1d'
GROUP BY week, symbol
WITH NO DATA;

-- ─────────────────────────────────────────────
-- 유용한 쿼리 함수 (재사용 가능)
-- ─────────────────────────────────────────────

-- 종목별 최신 가격 조회
CREATE OR REPLACE FUNCTION get_latest_price(p_symbol VARCHAR)
RETURNS TABLE (
    symbol VARCHAR,
    price DOUBLE PRECISION,
    volume DOUBLE PRECISION,
    time TIMESTAMPTZ
) AS $$
    SELECT symbol, close, volume, time
    FROM ohlcv
    WHERE symbol = p_symbol
    ORDER BY time DESC
    LIMIT 1;
$$ LANGUAGE SQL STABLE;

-- 롤링 수익률 계산 (윈도우 함수 활용)
CREATE OR REPLACE FUNCTION calc_rolling_returns(
    p_symbol VARCHAR,
    p_timeframe VARCHAR,
    p_window INT DEFAULT 20
)
RETURNS TABLE (
    time TIMESTAMPTZ,
    symbol VARCHAR,
    close DOUBLE PRECISION,
    returns DOUBLE PRECISION,
    rolling_return DOUBLE PRECISION,
    rolling_vol DOUBLE PRECISION
) AS $$
    WITH price_data AS (
        SELECT
            time,
            symbol,
            close,
            (close - LAG(close) OVER (ORDER BY time)) / NULLIF(LAG(close) OVER (ORDER BY time), 0) AS returns
        FROM ohlcv
        WHERE symbol = p_symbol AND timeframe = p_timeframe
        ORDER BY time
    )
    SELECT
        time,
        symbol,
        close,
        returns,
        SUM(returns) OVER (ORDER BY time ROWS BETWEEN (p_window-1) PRECEDING AND CURRENT ROW) AS rolling_return,
        STDDEV(returns) OVER (ORDER BY time ROWS BETWEEN (p_window-1) PRECEDING AND CURRENT ROW) AS rolling_vol
    FROM price_data
    WHERE returns IS NOT NULL;
$$ LANGUAGE SQL STABLE;

-- ─────────────────────────────────────────────
-- 보존 정책 (데이터 자동 삭제)
-- ─────────────────────────────────────────────
-- 틱 데이터: 90일 보존
SELECT add_retention_policy('ticks', INTERVAL '90 days', if_not_exists => TRUE);

-- 호가창: 30일 보존
SELECT add_retention_policy('orderbook_snapshots', INTERVAL '30 days', if_not_exists => TRUE);

-- OHLCV: 무기한 보존 (중요 데이터)
