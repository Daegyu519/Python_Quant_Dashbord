-- =============================================================================
-- ClickHouse Schema — Ultra-High-Speed Analytics
-- =============================================================================
-- 백테스트 결과, 팩터 분석, 전략 성과 집계에 최적화.
-- MergeTree 엔진 + 컬럼 압축 + 파티셔닝 활용.
-- =============================================================================

-- 데이터베이스 생성
CREATE DATABASE IF NOT EXISTS quant;

USE quant;

-- ─────────────────────────────────────────────
-- OHLCV 데이터 (분석용 복사본)
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS ohlcv (
    date        Date,
    time        DateTime64(3, 'UTC'),   -- 밀리초 정밀도
    symbol      LowCardinality(String), -- 종목수 제한 → 최적화
    timeframe   LowCardinality(String),
    open        Float64,
    high        Float64,
    low         Float64,
    close       Float64,
    volume      Float64,
    vwap        Float64,
    num_trades  UInt32,
    source      LowCardinality(String)
)
ENGINE = MergeTree()
PARTITION BY (toYYYYMM(date), symbol)   -- 월별 + 종목별 파티셔닝
ORDER BY (symbol, timeframe, time)
TTL date + INTERVAL 10 YEAR
SETTINGS
    index_granularity = 8192,
    min_bytes_for_wide_part = 0;

-- ─────────────────────────────────────────────
-- 백테스트 결과 (대규모 저장)
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS backtest_trades (
    run_id          String,
    strategy_id     LowCardinality(String),
    symbol          LowCardinality(String),
    entry_time      DateTime64(3, 'UTC'),
    exit_time       DateTime64(3, 'UTC'),
    entry_price     Float64,
    exit_price      Float64,
    quantity        Float64,
    side            LowCardinality(String),   -- LONG, SHORT
    pnl             Float64,
    pnl_pct         Float64,
    commission      Float64,
    slippage        Float64,
    holding_days    Float32,
    -- 진입/청산 신호 정보
    entry_signal    String,
    exit_signal     String,
    -- 파라미터 스냅샷
    parameters      String  -- JSON
)
ENGINE = MergeTree()
PARTITION BY (toYYYYMM(entry_time))
ORDER BY (strategy_id, symbol, entry_time)
SETTINGS index_granularity = 4096;

-- ─────────────────────────────────────────────
-- 팩터 데이터 (Factor Store)
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS factor_data (
    date            Date,
    symbol          LowCardinality(String),
    -- 기술적 팩터
    momentum_1m     Float64,    -- 1개월 모멘텀
    momentum_3m     Float64,    -- 3개월 모멘텀
    momentum_6m     Float64,    -- 6개월 모멘텀
    momentum_12m    Float64,    -- 12개월 모멘텀
    -- 변동성 팩터
    vol_5d          Float64,    -- 5일 변동성
    vol_20d         Float64,    -- 20일 변동성
    vol_60d         Float64,    -- 60일 변동성
    -- 기술적 지표
    rsi_14          Float64,
    macd            Float64,
    macd_signal     Float64,
    bb_position     Float64,    -- 볼린저 밴드 위치 (-1 ~ 1)
    -- 거래량 팩터
    vol_ratio_20d   Float64,    -- 거래량/20일 평균 비율
    -- ML 피처
    ml_score        Float64,    -- ML 모델 점수
    regime          LowCardinality(String)  -- 시장 레짐
)
ENGINE = MergeTree()
PARTITION BY toYYYYMM(date)
ORDER BY (symbol, date)
SETTINGS index_granularity = 8192;

-- ─────────────────────────────────────────────
-- 전략 성과 집계
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS strategy_performance (
    date            Date,
    strategy_id     LowCardinality(String),
    portfolio_value Float64,
    daily_pnl       Float64,
    daily_return    Float64,
    cumulative_return Float64,
    drawdown        Float64,
    sharpe_rolling  Float64,
    num_positions   UInt16,
    num_trades      UInt16,
    turnover        Float64,
    gross_exposure  Float64,
    net_exposure    Float64
)
ENGINE = MergeTree()
PARTITION BY toYYYYMM(date)
ORDER BY (strategy_id, date)
SETTINGS index_granularity = 8192;

-- ─────────────────────────────────────────────
-- 실시간 시그널 스트림 (Kafka/Redis → ClickHouse)
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS signal_stream (
    time            DateTime64(3, 'UTC'),
    signal_id       String,
    strategy_id     LowCardinality(String),
    symbol          LowCardinality(String),
    direction       Int8,       -- -2, -1, 0, 1, 2
    strength        Float32,
    confidence      Float32,
    source          LowCardinality(String),
    metadata        String      -- JSON
)
ENGINE = MergeTree()
PARTITION BY toYYYYMMDD(time)
ORDER BY (strategy_id, symbol, time)
TTL toDateTime(time) + INTERVAL 1 YEAR
SETTINGS index_granularity = 8192;

-- ─────────────────────────────────────────────
-- 고성능 분석 뷰 (집계 쿼리)
-- ─────────────────────────────────────────────

-- 월별 전략 성과 집계
CREATE VIEW IF NOT EXISTS v_strategy_monthly AS
SELECT
    toStartOfMonth(date) AS month,
    strategy_id,
    SUM(daily_pnl)       AS monthly_pnl,
    STDDEV(daily_return) * SQRT(21) AS monthly_vol,
    MIN(drawdown)        AS max_drawdown,
    SUM(num_trades)      AS total_trades
FROM strategy_performance
GROUP BY month, strategy_id;

-- 종목별 팩터 랭킹
CREATE VIEW IF NOT EXISTS v_factor_rankings AS
SELECT
    date,
    symbol,
    momentum_1m,
    momentum_3m,
    rsi_14,
    vol_20d,
    -- 팩터 랭킹 (백분위)
    rank() OVER (PARTITION BY date ORDER BY momentum_1m DESC) AS rank_momentum_1m,
    rank() OVER (PARTITION BY date ORDER BY momentum_3m DESC) AS rank_momentum_3m,
    rank() OVER (PARTITION BY date ORDER BY rsi_14 DESC) AS rank_rsi
FROM factor_data;

-- ─────────────────────────────────────────────
-- 분석용 저장 프로시저 (집계 쿼리 최적화)
-- ─────────────────────────────────────────────

-- 롤링 샤프 비율 계산
CREATE VIEW IF NOT EXISTS v_rolling_sharpe AS
SELECT
    date,
    strategy_id,
    avg(daily_return) OVER w / nullIf(stddevPop(daily_return) OVER w, 0) * sqrt(252) AS sharpe_20d
FROM strategy_performance
WINDOW w AS (PARTITION BY strategy_id ORDER BY date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW);
