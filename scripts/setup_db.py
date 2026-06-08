"""
=============================================================================
MySQL 데이터베이스 초기화 스크립트 (동기 방식 — pymysql)
=============================================================================
실행:
    python scripts/setup_db.py

- MySQL 9.x의 caching_sha2_password 인증 호환
- 동기 SQLAlchemy(pymysql)로 테이블 생성
- 앱 비동기 코드와는 별개 (초기화 전용)
=============================================================================
"""

from __future__ import annotations

import os
import sys
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pymysql
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.orm import sessionmaker


def get_db_config():
    """환경변수에서 DB 설정 읽기"""
    from dotenv import load_dotenv
    load_dotenv()

    return {
        "host":     os.getenv("MYSQL_HOST", "127.0.0.1"),
        "port":     int(os.getenv("MYSQL_PORT", "3306")),
        "user":     os.getenv("MYSQL_USER", "root"),
        "password": os.getenv("MYSQL_PASSWORD", ""),
        "database": os.getenv("MYSQL_DATABASE", "quant_db"),
    }


def build_sync_engine(cfg: dict):
    """
    SQLAlchemy 동기 엔진 생성.
    Mac 로컬 MySQL 9.x: 유닉스 소켓으로 연결 (caching_sha2 우회).
    """
    from sqlalchemy.engine import URL
    from dotenv import load_dotenv
    load_dotenv()
    socket_path = os.getenv("MYSQL_SOCKET", "/tmp/mysql.sock")

    url = URL.create(
        drivername="mysql+pymysql",
        username=cfg["user"],
        password=cfg["password"],
        database=cfg["database"],
        # host/port 생략 → connect_args의 unix_socket 사용
    )
    return create_engine(
        url,
        connect_args={"unix_socket": socket_path, "charset": "utf8mb4"},
        pool_pre_ping=True,
        echo=False,
    )


# ─────────────────────────────────────────────
# Step 1: 데이터베이스 생성
# ─────────────────────────────────────────────
def step1_create_database(cfg: dict):
    print(f"🔌 MySQL 접속 중... ({cfg['host']}:{cfg['port']})")

    conn = pymysql.connect(
        host=cfg["host"],
        port=cfg["port"],
        user=cfg["user"],
        password=cfg["password"],
        charset="utf8mb4",
    )
    cursor = conn.cursor()
    cursor.execute("SHOW DATABASES")
    existing = [r[0] for r in cursor.fetchall()]

    db_name = cfg["database"]
    if db_name not in existing:
        cursor.execute(
            f"CREATE DATABASE `{db_name}` "
            f"CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
        )
        conn.commit()
        print(f"✅ 데이터베이스 생성: {db_name}")
    else:
        print(f"ℹ️  데이터베이스 이미 존재: {db_name}")

    conn.close()


# ─────────────────────────────────────────────
# Step 2: 테이블 생성 (동기 SQLAlchemy)
# ─────────────────────────────────────────────
def step2_create_tables(cfg: dict):
    from app.database.models import Base

    print(f"\n🏗️  테이블 생성 중...")
    engine = build_sync_engine(cfg)

    # 모든 테이블 생성 (이미 있으면 건너뜀)
    Base.metadata.create_all(engine)

    # 생성된 테이블 확인
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    engine.dispose()

    print(f"✅ 생성된 테이블 ({len(tables)}개):")
    for t in sorted(tables):
        print(f"   📋 {t}")

    return tables


# ─────────────────────────────────────────────
# Step 3: 기본 데이터 삽입
# ─────────────────────────────────────────────
def step3_seed_data(cfg: dict):
    from app.database.models import User, Strategy

    engine = build_sync_engine(cfg)
    Session = sessionmaker(engine)

    with Session() as session:
        # 관리자 계정 존재 여부 확인
        existing = session.query(User).filter_by(email="admin@quant.local").first()

        if not existing:
            admin_id = str(uuid.uuid4())
            admin = User(
                id=admin_id,
                email="admin@quant.local",
                username="admin",
                hashed_password="$2b$12$PLACEHOLDER_change_in_production",
                is_active=True,
                is_superuser=True,
            )
            session.add(admin)
            session.flush()  # admin.id 확정

            # 샘플 전략 3개
            strategies = [
                Strategy(
                    id=str(uuid.uuid4()),
                    strategy_id="ma_cross_v1",
                    name="이동평균 크로스 전략 v1",
                    description="20일/50일 EMA 골든크로스 · 데드크로스",
                    strategy_type="technical",
                    config={"fast": 20, "slow": 50, "ma_type": "EMA"},
                    parameters={"fast_period": 20, "long_period": 50},
                    universe=["AAPL", "MSFT", "005930", "000660"],
                    is_active=True,
                    is_paper_trading=True,
                    user_id=admin_id,
                ),
                Strategy(
                    id=str(uuid.uuid4()),
                    strategy_id="rsi_v1",
                    name="RSI 평균회귀 전략 v1",
                    description="RSI 30/70 기준 과매도·과매수 역추세",
                    strategy_type="technical",
                    config={"period": 14, "oversold": 30, "overbought": 70},
                    parameters={"period": 14},
                    universe=["BTCUSDT", "ETHUSDT", "AAPL"],
                    is_active=True,
                    is_paper_trading=True,
                    user_id=admin_id,
                ),
                Strategy(
                    id=str(uuid.uuid4()),
                    strategy_id="ensemble_v1",
                    name="앙상블 전략 v1",
                    description="MA크로스 + RSI + MACD 3개 전략 다수결",
                    strategy_type="ensemble",
                    config={"min_agreement": 0.5},
                    parameters={"strategies": ["ma_cross_v1", "rsi_v1"]},
                    universe=["AAPL", "005930", "BTCUSDT"],
                    is_active=False,
                    is_paper_trading=True,
                    user_id=admin_id,
                ),
            ]
            session.add_all(strategies)
            session.commit()

            print("\n✅ 기본 데이터 삽입:")
            print("   👤 관리자: admin@quant.local")
            print("   📊 샘플 전략 3개 (ma_cross_v1, rsi_v1, ensemble_v1)")
        else:
            print("\nℹ️  기본 데이터 이미 존재 (건너뜀)")

    engine.dispose()


# ─────────────────────────────────────────────
# Step 4: 최종 확인
# ─────────────────────────────────────────────
def step4_verify(cfg: dict):
    engine = build_sync_engine(cfg)

    with engine.connect() as conn:
        users    = conn.execute(text("SELECT COUNT(*) FROM users")).scalar()
        strats   = conn.execute(text("SELECT COUNT(*) FROM strategies")).scalar()
        ohlcv    = conn.execute(text("SELECT COUNT(*) FROM ohlcv_data")).scalar()
        signals  = conn.execute(text("SELECT COUNT(*) FROM signals")).scalar()
        bt       = conn.execute(text("SELECT COUNT(*) FROM backtest_results")).scalar()

    engine.dispose()

    print(f"\n🔍 테이블 현황:")
    print(f"   users:            {users:>5}행")
    print(f"   strategies:       {strats:>5}행")
    print(f"   ohlcv_data:       {ohlcv:>5}행  ← 데이터 수집 후 채워짐")
    print(f"   signals:          {signals:>5}행  ← 전략 실행 후 채워짐")
    print(f"   backtest_results: {bt:>5}행  ← 백테스팅 후 채워짐")


# ─────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  AI Quant 플랫폼 — MySQL DB 초기화")
    print("=" * 55)

    cfg = get_db_config()

    step1_create_database(cfg)
    step2_create_tables(cfg)
    step3_seed_data(cfg)
    step4_verify(cfg)

    print("\n" + "=" * 55)
    print("  ✅ DB 초기화 완료!")
    print("=" * 55)
    print(f"""
  MySQL 연결 정보:
    호스트:   {cfg['host']}:{cfg['port']}
    DB명:     {cfg['database']}
    사용자:   {cfg['user']}

  다음 단계:
    python scripts/quick_start.py        ← 데이터 수집 테스트
    uvicorn app.api.main:app --port 8000  ← API 서버 시작
""")


if __name__ == "__main__":
    main()
