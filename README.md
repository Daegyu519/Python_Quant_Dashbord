# 📈 Python Quant Dashboard

> **실시간 미국 주식 퀀트 분석 대시보드** — Streamlit 한 줄로 띄우는 8개 분석 페이지
> Real-time quant analysis dashboard for US equities, built with Python & Streamlit.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.58-FF4B4B?logo=streamlit&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?logo=plotly&logoColor=white)
![License](https://img.shields.io/badge/License-Educational-blue)

**🔗 라이브 데모 →** https://daegyu-quant-dashboard.streamlit.app  · 아무 기기·모바일에서 바로 접속

종목(티커)을 직접 고르면 **실시간 시세 · 기술적 지표 · 포트폴리오 최적화 · 가치/ML 분석 ·
애널리스트 목표가 · 퀀트 전략 스코어**까지 한 화면에서 분석합니다. Docker·별도 서버 없이
파이썬 프로세스 하나로 동작합니다.

---

## ⚡ 빠른 시작

```bash
git clone https://github.com/Daegyu519/Python_Quant_Dashbord.git
cd Python_Quant_Dashbord
pip install -r requirements.txt
streamlit run dashboard.py        # → http://localhost:8501
```

API 키가 없어도 동작합니다 (무료 Yahoo Finance 데이터, 약 15분 지연).

---

## 🎯 핵심 기능 — 대시보드 8개 페이지

| 페이지 | 설명 |
|--------|------|
| 📈 **실시간** | 현재가·등락률 + 캔들차트(MA20/50, **볼린저밴드**, 거래량, **RSI**) · N초 자동 갱신 |
| 📊 **수익률·낙폭** | 종목별 누적 수익률 + 최대낙폭(MDD) 비교 |
| 🔗 **상관관계** | 수익률 상관 히트맵 + 분포 (분산투자 적합도) |
| 🎯 **포트폴리오 3D** | Markowitz 효율적 프론티어 (몬테카를로 5,000개, 최적 샤프 비중 산출) |
| 🔬 **전략 최적화 3D** | MA 크로스 파라미터별 샤프 지형도 |
| 🌊 **변동성 서피스 3D** | 종목·시간별 실현 변동성 지형도 |
| 💰 **가치·ML 분석** | 가치투자 점수(피오트로스키·버핏·DCF·그레이엄·알트만) + ML 예측 + **애널리스트 목표가·매수/매도 추천·신뢰도** |
| 🎯 **눌림목 스코어** | 하나증권 '과열주 눌림목' 전략 응용 — 3개월 강세 + 1개월 눌림 + 실적/목표가 확인 |

> 각 페이지에 **"📖 이 차트 읽는 법"** 해설을 내장해, 지표 해석 가이드를 함께 제공합니다.

---

## 🖼️ 스크린샷

### 📈 실시간 대시보드 — 캔들 + MA · 볼린저밴드 · RSI
<img width="1092" alt="실시간 캔들 차트" src="https://github.com/user-attachments/assets/5850722b-7243-44a5-8754-6dc1e648b249" />

### 🎯 3D 포트폴리오 최적화 — Markowitz 효율적 프론티어
<img width="1013" alt="3D 효율적 프론티어" src="https://github.com/user-attachments/assets/28ece42e-4f0f-4dbd-a48e-8493058d5f12" />

### 💰 가치·ML 분석 — 애널리스트 목표가 · 매수/매도 추천 · 신뢰도
<img width="1040" alt="가치·ML 분석 요약" src="https://github.com/user-attachments/assets/93743b39-c55b-4a09-bd3d-bd8598a88fb3" />
<img width="1068" alt="가치·ML 분석 차트" src="https://github.com/user-attachments/assets/2bb532b7-611b-4bbe-86fa-945b1066316f" />
<img width="1045" alt="ML 신뢰도·레짐 분석" src="https://github.com/user-attachments/assets/83be0429-c359-4619-8cfb-fe8a5837f5ef" />

---

## 🛠️ 기술 스택

| 분류 | 사용 기술 |
|------|----------|
| **언어** | Python 3.10+ |
| **대시보드 / 시각화** | Streamlit, Plotly (2D·3D) |
| **데이터** | yfinance (Yahoo Finance), pandas, NumPy |
| **머신러닝** | XGBoost, LightGBM, scikit-learn, hmmlearn (레짐 감지) |
| **퀀트** | Markowitz 포트폴리오 최적화, 백테스팅, 팩터 스코어링 |
| **(확장) 백엔드** | FastAPI, Celery, Redis, TimescaleDB, ClickHouse, Docker |

---

## 🧠 엔지니어링 하이라이트

- **표준 지표 정확도** — RSI를 단순이동평균이 아닌 **Wilder 평활(RMA)** 로 구현해 TradingView/Yahoo와 동일한 값 산출
- **포트폴리오 최적화** — 몬테카를로 5,000개 시뮬레이션으로 효율적 프론티어를 3D로 시각화하고 최적 샤프 비중 도출
- **다중 신호 결합** — 가치투자(재무 5종) + ML(상승확률·레짐) + 기술적 점수를 가중 종합해 투자 판단 산출
- **애널리스트 컨센서스 통합** — Yahoo 목표가·투자의견·EPS 추정치 변화(어닝 리비전)를 추천에 반영
- **리서치 기반 팩터 구현** — 하나증권 퀀트 리포트의 '과열주 눌림목' 전략을 분석적으로 재현(가격·목표가·실적 팩터). 원전의 한국 수급 팩터는 데이터 제약을 고려해 의도적으로 제외
- **방어적 설계** — 종목별 try/except, 캐시(TTL) 기반 호출 최소화, 결측치 보정

---

## 📁 프로젝트 구조

```
Python_Quant_Dashbord/
├── dashboard.py            # ⭐ 실시간 대시보드 (메인 진입점)
├── app/                    # 퀀트 엔진 (확장 플랫폼)
│   ├── data/collectors/    # Yahoo·Binance·KRX·CoinGecko 수집기
│   ├── strategies/         # MA·RSI·MACD·볼린저·모멘텀·앙상블
│   ├── backtesting/        # Numba JIT 벡터화 백테스팅
│   ├── ml/                 # 피처 엔지니어링 + 앙상블 모델
│   ├── value_investing/    # 가치투자 스크리너
│   ├── portfolio/          # Mean-Variance·Risk Parity·Black-Litterman
│   ├── risk/               # VaR·CVaR·포지션 한도
│   └── api/                # FastAPI (확장용)
├── scripts/
│   ├── quick_start.py      # 무료 데이터 소스 점검
│   ├── visualize.py        # 정적 차트 HTML 생성
│   └── advanced_analysis.py# 가치 + ML 종합 분석
├── docs/ARCHITECTURE.md    # 아키텍처 문서
├── 설명서.md               # 코드 전체 상세 설명서(한국어)
└── requirements.txt
```

---

## 📊 데이터 소스 (전부 무료)

| 소스 | 자산 | API 키 |
|------|------|--------|
| Yahoo Finance | 미국 주식·ETF·지수, 애널리스트 목표가/추정치 | 불필요 |
| Binance | 암호화폐 (BTCUSDT 등) | 불필요 |
| KRX / FinanceDataReader | 한국 주식 | 불필요 |
| CoinGecko | 암호화폐 OHLCV | 불필요 |

---

## ⚠️ 면책

본 프로젝트는 **교육·포트폴리오 목적**으로 제작되었습니다. 표시되는 점수·추천·목표가는
**투자 조언이 아니며**, 실제 투자 판단의 근거로 사용해서는 안 됩니다. 무료 데이터는
약 15분 지연되며 정확성을 보장하지 않습니다.

---

<p align="center">
  <sub>Built with Python · Streamlit · Plotly — by <a href="https://github.com/Daegyu519">Daegyu519</a></sub>
</p>
