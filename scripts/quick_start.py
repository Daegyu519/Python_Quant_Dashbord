"""
=============================================================================
Quick Start — 모든 무료 데이터 소스 통합 데모
=============================================================================
API 키 없이 즉시 실행 가능한 데모.
Yahoo Finance + KRX + Binance + CoinGecko 전부 테스트.

실행:
    pip install -r requirements.txt
    python scripts/quick_start.py
=============================================================================
"""

from __future__ import annotations

import asyncio
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime
import warnings
warnings.filterwarnings("ignore")


async def check_api_status():
    """API 설정 현황 출력"""
    from app.config.settings import get_settings
    cfg = get_settings()

    print("\n📋 API 키 현황:")
    print("─" * 50)
    for name, status in cfg.api_keys.summary().items():
        print(f"  {name:<20} {status}")
    print("─" * 50)


async def test_yahoo():
    """Yahoo Finance 테스트 (키 불필요)"""
    print("\n📈 [1] Yahoo Finance 테스트...")
    from app.data.collectors.yahoo_collector import YahooFinanceCollector
    from app.core.types import Timeframe

    collector = YahooFinanceCollector()
    start = datetime(2023, 1, 1)

    # 미국 주식
    aapl = await collector.fetch_ohlcv("AAPL", Timeframe.ONE_DAY, start)
    print(f"   ✅ AAPL (애플):       {aapl.n_bars}개 봉, 최근가 ${aapl.close_prices[-1]:.2f}")

    # 한국 주식 (야후에서도 가능)
    samsung = await collector.fetch_ohlcv("005930.KS", Timeframe.ONE_DAY, start)
    print(f"   ✅ 005930.KS (삼성전자): {samsung.n_bars}개 봉, 최근가 ₩{samsung.close_prices[-1]:,.0f}")

    # 지수
    sp500 = await collector.fetch_ohlcv("^GSPC", Timeframe.ONE_DAY, start)
    print(f"   ✅ S&P 500 지수:      {sp500.n_bars}개 봉, 최근가 ${sp500.close_prices[-1]:,.2f}")

    # 코스피 지수
    kospi = await collector.fetch_ohlcv("^KS11", Timeframe.ONE_DAY, start)
    if kospi.n_bars > 0:
        print(f"   ✅ KOSPI 지수:        {kospi.n_bars}개 봉, 최근가 {kospi.close_prices[-1]:,.2f}")

    return {"AAPL": aapl, "SAMSUNG_YF": samsung, "SPY": sp500}


async def test_krx():
    """KRX 한국 주식 테스트 (키 불필요)"""
    print("\n🇰🇷 [2] KRX 한국 주식 테스트...")

    try:
        from app.data.collectors.krx_collector import KRXCollector
        from app.core.types import Timeframe

        collector = KRXCollector()

        if not collector._fdr_available and not collector._pykrx_available:
            print("   ⚠️  FinanceDataReader 또는 pykrx가 필요합니다.")
            print("      pip install finance-datareader pykrx")
            return {}

        start = datetime(2023, 1, 1)

        # 삼성전자
        samsung = await collector.fetch_ohlcv("005930", Timeframe.ONE_DAY, start)
        if samsung.n_bars > 0:
            print(f"   ✅ 삼성전자 (005930):   {samsung.n_bars}개 봉, 최근가 ₩{samsung.close_prices[-1]:,.0f}")

        # SK하이닉스
        hynix = await collector.fetch_ohlcv("000660", Timeframe.ONE_DAY, start)
        if hynix.n_bars > 0:
            print(f"   ✅ SK하이닉스 (000660): {hynix.n_bars}개 봉, 최근가 ₩{hynix.close_prices[-1]:,.0f}")

        # 코스피 지수
        kospi = await collector.fetch_ohlcv("KS11", Timeframe.ONE_DAY, start)
        if kospi.n_bars > 0:
            print(f"   ✅ KOSPI 지수 (KS11):   {kospi.n_bars}개 봉, 최근가 {kospi.close_prices[-1]:,.2f}")

        return {"005930": samsung, "000660": hynix}

    except Exception as e:
        print(f"   ❌ KRX 오류: {e}")
        print("      pip install finance-datareader 후 재시도하세요.")
        return {}


async def test_binance():
    """Binance 공개 API 테스트 (키 불필요)"""
    print("\n₿ [3] Binance 암호화폐 테스트 (키 불필요)...")

    try:
        from app.data.collectors.binance_collector import BinanceCollector
        from app.core.types import Timeframe

        # API 키 없이 공개 데이터 사용
        collector = BinanceCollector(api_key=None, secret_key=None)
        await collector.connect()

        start = datetime(2023, 1, 1)

        btc = await collector.fetch_ohlcv("BTCUSDT", Timeframe.ONE_DAY, start)
        print(f"   ✅ BTC/USDT:  {btc.n_bars}개 봉, 최근가 ${btc.close_prices[-1]:,.2f}")

        eth = await collector.fetch_ohlcv("ETHUSDT", Timeframe.ONE_DAY, start)
        print(f"   ✅ ETH/USDT:  {eth.n_bars}개 봉, 최근가 ${eth.close_prices[-1]:,.2f}")

        await collector.disconnect()
        return {"BTCUSDT": btc, "ETHUSDT": eth}

    except Exception as e:
        print(f"   ❌ Binance 오류: {e}")
        print("      인터넷 연결을 확인하세요.")
        return {}


async def test_coingecko():
    """CoinGecko 테스트 (키 불필요)"""
    print("\n🦎 [4] CoinGecko 테스트 (키 불필요)...")

    try:
        from app.data.collectors.coingecko_collector import CoinGeckoCollector
        from app.core.types import Timeframe

        collector = CoinGeckoCollector()
        await collector.connect()

        start = datetime(2023, 1, 1)
        btc = await collector.fetch_ohlcv("BTC", Timeframe.ONE_DAY, start)
        print(f"   ✅ Bitcoin (CoinGecko): {btc.n_bars}개 봉")

        # 시장 개요
        overview = await collector.get_market_overview()
        btc_dominance = overview.get("market_cap_percentage", {}).get("btc", 0)
        total_mcap = overview.get("total_market_cap", {}).get("usd", 0)
        print(f"   ✅ BTC 도미넌스: {btc_dominance:.1f}%")
        print(f"   ✅ 전체 시가총액: ${total_mcap/1e12:.2f}T")

        await collector.disconnect()
        return {"BTC_CG": btc}

    except Exception as e:
        print(f"   ❌ CoinGecko 오류: {e}")
        return {}


async def test_data_manager():
    """DataManager 통합 테스트 (자동 소스 감지)"""
    print("\n🔀 [5] DataManager 자동 라우팅 테스트...")

    from app.data.collectors.data_manager import DataManager, detect_source

    symbols = [
        "AAPL",        # → Yahoo Finance (미국 주식)
        "MSFT",        # → Yahoo Finance
        "005930",      # → KRX (한국 주식)
        "BTCUSDT",     # → Binance (암호화폐)
        "BTC",         # → CoinGecko
        "^KS11",       # → Yahoo Finance (코스피 지수)
    ]

    print("   자동 소스 감지:")
    for sym in symbols:
        source = detect_source(sym)
        print(f"     {sym:<12} → {source.value}")


async def test_ensemble_strategy():
    """전략 + 데이터 통합 테스트"""
    print("\n🎯 [6] 앙상블 전략 시그널 테스트...")

    from app.data.collectors.yahoo_collector import YahooFinanceCollector
    from app.core.types import Timeframe
    from app.strategies.technical_strategies import (
        MovingAverageCrossStrategy, RSIStrategy, MACDStrategy, EnsembleStrategy
    )

    collector = YahooFinanceCollector()
    data = await collector.fetch_ohlcv("AAPL", Timeframe.ONE_DAY, datetime(2022, 1, 1))

    ensemble = EnsembleStrategy(
        strategies=[
            MovingAverageCrossStrategy(fast_period=20, long_period=50),
            RSIStrategy(period=14, oversold=30, overbought=70),
            MACDStrategy(fast=12, slow=26, signal_period=9),
        ],
        min_agreement=0.5,
    )

    signals = ensemble.generate_signal(data)

    if signals:
        s = signals[0]
        direction = "🔼 매수(BUY)" if s.is_long else "🔽 매도(SELL)"
        print(f"   ✅ AAPL 앙상블 시그널: {direction}")
        print(f"      강도: {s.strength:.1%} | 신뢰도: {s.confidence:.1%}")
    else:
        print("   📋 AAPL: 현재 명확한 시그널 없음 (중립)")


async def test_feature_engineering():
    """ML 피처 엔지니어링 테스트"""
    print("\n🤖 [7] ML 피처 엔지니어링 테스트...")

    from app.data.collectors.yahoo_collector import YahooFinanceCollector
    from app.core.types import Timeframe, OHLCVFrame
    from app.ml.features.feature_engineering import QuantFeatureEngineer

    collector = YahooFinanceCollector()
    data = await collector.fetch_ohlcv("AAPL", Timeframe.ONE_DAY, datetime(2020, 1, 1))

    # Train/Test 분할
    split = int(len(data.data) * 0.8)
    train = OHLCVFrame("AAPL", Timeframe.ONE_DAY, data.data.iloc[:split])
    test = OHLCVFrame("AAPL", Timeframe.ONE_DAY, data.data.iloc[split:])

    engineer = QuantFeatureEngineer()
    engineer.fit(train)
    features = engineer.transform(test).dropna()

    print(f"   ✅ 피처 수: {len(engineer.get_feature_names())}개")
    print(f"   ✅ 샘플 수: {len(features)}개")
    print(f"   ✅ 주요 피처: {', '.join(engineer.get_feature_names()[:5])}")


async def main():
    print("=" * 65)
    print("  AI Quant Trading Platform — 무료 API 통합 데모")
    print("  (Yahoo Finance + KRX + Binance + CoinGecko)")
    print("=" * 65)

    await check_api_status()

    all_results = {}

    # 각 테스트 실행
    yahoo_data = await test_yahoo()
    all_results.update(yahoo_data)

    krx_data = await test_krx()
    all_results.update(krx_data)

    binance_data = await test_binance()
    all_results.update(binance_data)

    cg_data = await test_coingecko()
    all_results.update(cg_data)

    await test_data_manager()
    await test_ensemble_strategy()
    await test_feature_engineering()

    # 최종 요약
    print("\n" + "=" * 65)
    print("  ✅ 데모 완료! 수집된 데이터:")
    print("─" * 65)
    for name, frame in all_results.items():
        if frame.n_bars > 0:
            start = frame.data.index[0].date()
            end = frame.data.index[-1].date()
            print(f"  {name:<15} {frame.n_bars:>5}개 봉  ({start} ~ {end})")
    print("=" * 65)

    print("""
  🚀 다음 단계:

  [지금 바로 가능]
    python scripts/quick_start.py  ← 이 파일 (API 키 불필요)

  [선택: 무료 키 발급 후 더 많은 데이터]
    Alpha Vantage: https://www.alphavantage.co/support/#api-key
    FRED:          https://fred.stlouisfed.org/docs/api/api_key.html
    → .env 파일에 추가하면 자동 적용

  [전체 스택 실행]
    docker-compose up -d
    http://localhost:8000/docs   ← API 문서
    http://localhost:3000        ← 대시보드
""")


if __name__ == "__main__":
    asyncio.run(main())
