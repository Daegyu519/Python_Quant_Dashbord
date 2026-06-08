# 🔑 API 발급 가이드 — 무료만 사용

## 📊 한눈에 보는 API 분류표

| API | 가입 필요 | 키 필요 | 무료 한도 | 용도 |
|-----|---------|--------|---------|------|
| **Yahoo Finance** | ❌ 불필요 | ❌ 불필요 | 무제한 (비공식) | 미국 주식, ETF, 지수, 환율 |
| **FinanceDataReader** | ❌ 불필요 | ❌ 불필요 | 무제한 | 🇰🇷 한국 주식 (KRX) |
| **pykrx** | ❌ 불필요 | ❌ 불필요 | 무제한 | 🇰🇷 KRX 공식 데이터 |
| **Binance Public** | ❌ 불필요 | ❌ 불필요 | 1200 req/min | 암호화폐 OHLCV, 실시간 WebSocket |
| **CoinGecko** | ❌ 불필요 | ❌ 불필요 | 10~30 req/min | 암호화폐 가격, 시총 |
| **Alpha Vantage** | ✅ 가입 | ✅ 필요 | 25 req/day | 미국 주식 (야후 보완) |
| **FRED** | ✅ 가입 | ✅ 필요 | 무제한 | 🇺🇸 거시경제 지표 (금리, CPI 등) |

---

## ✅ 가입/키 발급 안 해도 되는 것들

### 1. Yahoo Finance — 즉시 사용 가능
```python
# 그냥 설치만 하면 됨 (가입 불필요)
pip install yfinance

# 사용 예시
import yfinance as yf
df = yf.download("AAPL", start="2020-01-01")
df = yf.download("005930.KS", start="2020-01-01")  # 삼성전자
```
- 미국 주식: AAPL, MSFT, TSLA, SPY, QQQ ...
- 한국 주식: `005930.KS` (삼성전자), `000660.KS` (SK하이닉스)
- 암호화폐: BTC-USD, ETH-USD
- 지수: ^KS11 (코스피), ^IXIC (나스닥), ^GSPC (S&P500)
- 환율: USDKRW=X

### 2. FinanceDataReader — 한국 주식 전문 (가입 불필요)
```python
pip install finance-datareader

import FinanceDataReader as fdr
df = fdr.DataReader("005930", "2020-01-01")  # 삼성전자
df = fdr.DataReader("KS11", "2020-01-01")   # 코스피 지수
stocks = fdr.StockListing("KRX")            # 전체 KRX 종목 리스트
```
- 코스피/코스닥 전 종목
- 나스닥, NYSE, S&P500 종목 리스트
- 환율, 원자재

### 3. pykrx — KRX 공식 데이터 (가입 불필요)
```python
pip install pykrx

from pykrx import stock
df = stock.get_market_ohlcv("20200101", "20241231", "005930")
df = stock.get_market_fundamental("20240101", "20241231", "005930")  # PER, PBR
```
- KRX 공식 데이터 (가장 정확)
- 시가총액, PER, PBR, DIV 포함

### 4. Binance 공개 API — 암호화폐 (가입 불필요)
```python
# API 키 없이 시장 데이터 사용 가능
# 1200 req/min 제한 (매우 넉넉)
import aiohttp

url = "https://api.binance.com/api/v3/klines"
# BTCUSDT, ETHUSDT, SOLUSDT 등 모든 페어
```

### 5. CoinGecko — 암호화폐 (가입 불필요)
```python
pip install pycoingecko

from pycoingecko import CoinGeckoAPI
cg = CoinGeckoAPI()
data = cg.get_coin_market_chart_by_id("bitcoin", vs_currency="usd", days=365)
```

---

## 📝 무료 가입 후 키 발급 (선택사항)

### 1. Alpha Vantage — 미국 주식 보완 데이터
> 야후파이낸스만으로 부족할 때 사용. 없어도 플랫폼 완전 동작.

**발급 방법:**
1. https://www.alphavantage.co/support/#api-key 접속
2. 이름, 이메일만 입력 → 즉시 발급 (확인 메일 불필요)
3. `.env` 파일에 입력

**무료 한도:** 25 req/일, 500 req/월
**데이터:** 주가, 기술적 지표, 재무제표

### 2. FRED (Federal Reserve Economic Data) — 거시경제 지표
> 금리, CPI, GDP, 실업률 등 미국 거시경제 데이터

**발급 방법:**
1. https://fred.stlouisfed.org/docs/api/api_key.html 접속
2. 계정 생성 → API Key 신청 → 즉시 발급
3. `.env` 파일에 입력

**무료 한도:** 무제한 (일반적 사용)
**데이터:** 연방기금금리, CPI, 실업률, GDP, 채권금리

---

## ❌ 사용하지 않을 것들 (유료 또는 불필요)

| API | 이유 |
|-----|------|
| Polygon.io | 무료 티어 너무 제한적 (EOD만, 지연 데이터) |
| Bloomberg Terminal | 연간 2,000만원+ 유료 |
| Refinitiv Eikon | 유료 |
| Bybit | Binance로 대체 |
| KIS (한국투자증권) | 실거래 연동용, 지금 단계에서 불필요 |

---

## 🛠️ 최종 .env 설정 가이드

아래 키들만 `.env` 파일에 설정하면 됩니다:

```bash
# ── 필수 (가입 불필요, 설정 불필요) ──
# Yahoo Finance, FinanceDataReader, pykrx, Binance Public은 키 없이 동작

# ── 선택 (무료 가입) ──
ALPHA_VANTAGE_KEY=여기에_발급받은_키_입력    # alphavantage.co
FRED_API_KEY=여기에_발급받은_키_입력         # fred.stlouisfed.org

# ── 암호화폐 실시간 거래용 (현재 불필요) ──
# BINANCE_API_KEY=  ← 시장 데이터만 쓸 때는 불필요
# BINANCE_SECRET_KEY=
```

**결론: 지금 당장 아무것도 발급 안 해도 플랫폼 완전 동작 가능합니다.**
Alpha Vantage와 FRED는 있으면 데이터가 풍부해지는 보너스입니다.
