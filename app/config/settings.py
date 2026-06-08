"""
=============================================================================
Production-Grade Configuration Management
=============================================================================
Pydantic BaseSettings 기반 환경 설정 관리.
모든 설정은 환경변수 또는 .env 파일로 주입 가능.
12-factor app 원칙을 따른다.
=============================================================================
"""

from __future__ import annotations

import os
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

from pydantic import Field, PostgresDsn, RedisDsn, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# ─────────────────────────────────────────────
# 환경 타입 열거형
# ─────────────────────────────────────────────
class Environment(str, Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


# ─────────────────────────────────────────────
# 데이터베이스 설정 (MySQL)
# ─────────────────────────────────────────────
class DatabaseSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="allow",
    )

    # MySQL (로컬 설치)
    mysql_host: str = Field(default="127.0.0.1", alias="MYSQL_HOST")
    mysql_port: int = Field(default=3306, alias="MYSQL_PORT")
    mysql_user: str = Field(default="root", alias="MYSQL_USER")
    mysql_password: str = Field(default="", alias="MYSQL_PASSWORD")
    mysql_database: str = Field(default="quant_db", alias="MYSQL_DATABASE")
    mysql_pool_size: int = Field(default=10, alias="MYSQL_POOL_SIZE")
    mysql_max_overflow: int = Field(default=20, alias="MYSQL_MAX_OVERFLOW")
    # Mac 로컬 MySQL 소켓 경로 (caching_sha2_password 우회용)
    mysql_socket: Optional[str] = Field(default="/tmp/mysql.sock", alias="MYSQL_SOCKET")

    # DuckDB (로컬 분석용 — 설치 불필요)
    duckdb_path: str = Field(default="data/quant.duckdb", alias="DUCKDB_PATH")

    @property
    def async_dsn(self) -> str:
        """
        비동기 SQLAlchemy 연결 문자열 (aiomysql).
        Mac 로컬: 유닉스 소켓 사용 (MySQL 9.x caching_sha2 호환).
        Linux/서버: TCP 연결.
        """
        from sqlalchemy.engine import URL
        if self.mysql_socket:
            # 유닉스 소켓 방식 (Mac 로컬 MySQL 9.x 권장)
            url = URL.create(
                drivername="mysql+aiomysql",
                username=self.mysql_user,
                password=self.mysql_password,
                database=self.mysql_database,
                query={
                    "charset": "utf8mb4",
                    "unix_socket": self.mysql_socket,
                },
            )
        else:
            # TCP 방식 (Linux 서버, Docker 등)
            url = URL.create(
                drivername="mysql+aiomysql",
                username=self.mysql_user,
                password=self.mysql_password,
                host=self.mysql_host,
                port=self.mysql_port,
                database=self.mysql_database,
                query={"charset": "utf8mb4"},
            )
        return str(url)

    @property
    def sync_dsn(self) -> str:
        """
        동기 SQLAlchemy 연결 문자열 (pymysql).
        URL.create() 사용 → 비밀번호의 @, #, % 등 특수문자 자동 인코딩.
        """
        from sqlalchemy.engine import URL
        url = URL.create(
            drivername="mysql+pymysql",
            username=self.mysql_user,
            password=self.mysql_password,
            host=self.mysql_host,
            port=self.mysql_port,
            database=self.mysql_database,
            query={"charset": "utf8mb4"},
        )
        return str(url)


# ─────────────────────────────────────────────
# Redis 설정
# ─────────────────────────────────────────────
class RedisSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="REDIS_")

    host: str = Field(default="localhost", alias="REDIS_HOST")
    port: int = Field(default=6379, alias="REDIS_PORT")
    password: Optional[str] = Field(default=None, alias="REDIS_PASSWORD")
    db: int = Field(default=0, alias="REDIS_DB")

    # 채널별 DB 분리
    cache_db: int = Field(default=1, alias="REDIS_CACHE_DB")
    stream_db: int = Field(default=2, alias="REDIS_STREAM_DB")
    celery_db: int = Field(default=3, alias="REDIS_CELERY_DB")

    # TTL 설정 (초)
    ticker_cache_ttl: int = Field(default=5, alias="REDIS_TICKER_TTL")
    ohlcv_cache_ttl: int = Field(default=60, alias="REDIS_OHLCV_TTL")
    signal_cache_ttl: int = Field(default=30, alias="REDIS_SIGNAL_TTL")

    @property
    def url(self) -> str:
        if self.password:
            return f"redis://:{self.password}@{self.host}:{self.port}/{self.db}"
        return f"redis://{self.host}:{self.port}/{self.db}"

    @property
    def celery_url(self) -> str:
        if self.password:
            return f"redis://:{self.password}@{self.host}:{self.port}/{self.celery_db}"
        return f"redis://{self.host}:{self.port}/{self.celery_db}"


# ─────────────────────────────────────────────
# API 키 설정
# ─────────────────────────────────────────────
class APIKeySettings(BaseSettings):
    """
    외부 API 키 설정.

    ✅ 키 없이 동작하는 소스 (설정 불필요):
       - Yahoo Finance  (yfinance)
       - KRX            (FinanceDataReader / pykrx)
       - Binance 시장데이터 (공개 API)
       - CoinGecko      (공개 API)

    📝 선택사항 (무료 가입 후 발급):
       - Alpha Vantage: https://www.alphavantage.co/support/#api-key
       - FRED:          https://fred.stlouisfed.org/docs/api/api_key.html

    ❌ 현재 미사용 (실거래 시 필요):
       - Binance 트레이딩 키 (주문 실행용)
       - KIS API (한국투자증권 실거래)
    """
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="allow",
    )

    # ── 📝 선택사항: 무료 가입 후 발급 ──
    alpha_vantage_key: Optional[str] = Field(
        default=None,
        alias="ALPHA_VANTAGE_KEY",
        description="25 req/day 무료. https://www.alphavantage.co/support/#api-key",
    )
    fred_api_key: Optional[str] = Field(
        default=None,
        alias="FRED_API_KEY",
        description="무제한 무료. https://fred.stlouisfed.org/docs/api/api_key.html",
    )

    # ── ❌ 현재 미사용 (실거래 연동 시 필요) ──
    binance_api_key: Optional[str] = Field(
        default=None,
        alias="BINANCE_API_KEY",
        description="트레이딩용. 시장데이터는 키 없이 동작.",
    )
    binance_secret_key: Optional[str] = Field(default=None, alias="BINANCE_SECRET_KEY")
    binance_testnet: bool = Field(default=True, alias="BINANCE_TESTNET")

    kis_app_key: Optional[str] = Field(
        default=None,
        alias="KIS_APP_KEY",
        description="한국투자증권 실거래 연동용.",
    )
    kis_app_secret: Optional[str] = Field(default=None, alias="KIS_APP_SECRET")

    @property
    def has_alpha_vantage(self) -> bool:
        return bool(self.alpha_vantage_key)

    @property
    def has_fred(self) -> bool:
        return bool(self.fred_api_key)

    @property
    def has_binance_trading(self) -> bool:
        return bool(self.binance_api_key and self.binance_secret_key)

    def summary(self) -> dict[str, str]:
        """API 키 설정 현황 요약"""
        return {
            "Yahoo Finance":     "✅ 키 불필요 (항상 동작)",
            "KRX (한국주식)":   "✅ 키 불필요 (항상 동작)",
            "Binance 시장데이터": "✅ 키 불필요 (항상 동작)",
            "CoinGecko":         "✅ 키 불필요 (항상 동작)",
            "Alpha Vantage":     "✅ 설정됨" if self.has_alpha_vantage else "📝 미설정 (선택사항)",
            "FRED 거시경제":     "✅ 설정됨" if self.has_fred else "📝 미설정 (선택사항)",
            "Binance 트레이딩":  "✅ 설정됨" if self.has_binance_trading else "❌ 미설정 (실거래 시 필요)",
            "KIS (실거래)":      "✅ 설정됨" if self.kis_app_key else "❌ 미설정 (실거래 시 필요)",
        }


# ─────────────────────────────────────────────
# 머신러닝 설정
# ─────────────────────────────────────────────
class MLSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ML_")

    # GPU 사용 여부
    use_gpu: bool = Field(default=False, alias="ML_USE_GPU")
    gpu_device: int = Field(default=0, alias="ML_GPU_DEVICE")

    # 모델 저장 경로
    model_save_path: str = Field(default="data/models", alias="ML_MODEL_PATH")

    # 학습 설정
    default_lookback: int = Field(default=252, alias="ML_LOOKBACK")     # 거래일 1년
    default_forecast: int = Field(default=5, alias="ML_FORECAST")       # 5일 예측
    validation_size: float = Field(default=0.2, alias="ML_VAL_SIZE")
    n_cv_folds: int = Field(default=5, alias="ML_CV_FOLDS")

    # Optuna
    optuna_n_trials: int = Field(default=100, alias="ML_OPTUNA_TRIALS")
    optuna_timeout: int = Field(default=3600, alias="ML_OPTUNA_TIMEOUT")  # 1시간

    # Feature Store
    feature_store_path: str = Field(default="data/features", alias="ML_FEATURE_PATH")


# ─────────────────────────────────────────────
# 백테스팅 설정
# ─────────────────────────────────────────────
class BacktestSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="BT_")

    # 기본 수수료 (bps)
    commission_bps: float = Field(default=5.0, alias="BT_COMMISSION_BPS")    # 0.05%
    slippage_bps: float = Field(default=2.0, alias="BT_SLIPPAGE_BPS")        # 0.02%

    # 초기 자본
    initial_capital: float = Field(default=100_000_000.0, alias="BT_INITIAL_CAPITAL")  # 1억원

    # 최대 레버리지
    max_leverage: float = Field(default=1.0, alias="BT_MAX_LEVERAGE")

    # 병렬 처리
    n_jobs: int = Field(default=-1, alias="BT_N_JOBS")   # -1 = 모든 CPU 코어
    chunk_size: int = Field(default=1000, alias="BT_CHUNK_SIZE")

    # Monte Carlo
    mc_simulations: int = Field(default=10_000, alias="BT_MC_SIMULATIONS")


# ─────────────────────────────────────────────
# 리스크 설정
# ─────────────────────────────────────────────
class RiskSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RISK_")

    # VaR 설정
    var_confidence: float = Field(default=0.95, alias="RISK_VAR_CONFIDENCE")
    cvar_confidence: float = Field(default=0.99, alias="RISK_CVAR_CONFIDENCE")
    var_lookback_days: int = Field(default=252, alias="RISK_VAR_LOOKBACK")

    # 포지션 한도
    max_position_size: float = Field(default=0.1, alias="RISK_MAX_POSITION")  # 10%
    max_sector_exposure: float = Field(default=0.3, alias="RISK_MAX_SECTOR")  # 30%
    max_drawdown_limit: float = Field(default=0.15, alias="RISK_MAX_DRAWDOWN") # 15%

    # 손실 한도
    daily_loss_limit: float = Field(default=0.02, alias="RISK_DAILY_LOSS")    # 2%
    weekly_loss_limit: float = Field(default=0.05, alias="RISK_WEEKLY_LOSS")  # 5%


# ─────────────────────────────────────────────
# API 서버 설정
# ─────────────────────────────────────────────
class ServerSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SERVER_")

    host: str = Field(default="0.0.0.0", alias="SERVER_HOST")
    port: int = Field(default=8000, alias="SERVER_PORT")
    workers: int = Field(default=4, alias="SERVER_WORKERS")
    reload: bool = Field(default=False, alias="SERVER_RELOAD")

    # CORS
    allowed_origins: list[str] = Field(
        default=["http://localhost:3000", "http://localhost:3001"],
        alias="ALLOWED_ORIGINS"
    )

    # JWT
    jwt_secret: str = Field(default="CHANGE_THIS_IN_PRODUCTION", alias="JWT_SECRET")
    jwt_algorithm: str = Field(default="HS256", alias="JWT_ALGORITHM")
    jwt_expire_minutes: int = Field(default=1440, alias="JWT_EXPIRE_MINUTES")  # 24시간

    # Rate Limiting
    rate_limit_per_minute: int = Field(default=100, alias="RATE_LIMIT_PER_MINUTE")


# ─────────────────────────────────────────────
# 메인 설정 (통합)
# ─────────────────────────────────────────────
class Settings(BaseSettings):
    """
    중앙 집중식 설정 관리.
    모든 하위 설정을 통합하여 단일 진입점 제공.
    """
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="allow",
    )

    # 환경 설정
    environment: Environment = Field(default=Environment.DEVELOPMENT)
    debug: bool = Field(default=False)
    log_level: LogLevel = Field(default=LogLevel.INFO)
    app_name: str = Field(default="AI Quant Platform")
    app_version: str = Field(default="1.0.0")

    # 프로젝트 루트
    base_dir: Path = Field(default_factory=lambda: Path(__file__).parent.parent.parent)

    # 하위 설정 (nested)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    api_keys: APIKeySettings = Field(default_factory=APIKeySettings)
    ml: MLSettings = Field(default_factory=MLSettings)
    backtest: BacktestSettings = Field(default_factory=BacktestSettings)
    risk: RiskSettings = Field(default_factory=RiskSettings)
    server: ServerSettings = Field(default_factory=ServerSettings)

    @property
    def is_production(self) -> bool:
        return self.environment == Environment.PRODUCTION

    @property
    def is_development(self) -> bool:
        return self.environment == Environment.DEVELOPMENT

    @property
    def data_dir(self) -> Path:
        return self.base_dir / "data"

    @property
    def log_dir(self) -> Path:
        return self.base_dir / "logs"


# ─────────────────────────────────────────────
# 싱글턴 패턴 (캐싱)
# ─────────────────────────────────────────────
@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    설정 싱글턴 반환.
    lru_cache로 앱 실행 중 한 번만 초기화.
    """
    return Settings()


# 모듈 레벨 편의 접근
settings = get_settings()
