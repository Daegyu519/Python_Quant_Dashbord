# 📈 Python Quant Dashboard

> **실시간 미국 주식 퀀트 분석 대시보드** — Streamlit 한 줄로 띄우는 17개 분석 페이지
> Real-time quant analysis dashboard for US equities, built with Python & Streamlit.
> 토스(Toss) 스타일 UI(라이트/다크 토글) + 스크리너 + AI 추천 + 암호화폐·옵션 + 모의 포트폴리오 + 거시지표 + 전략 백테스트 + Gemini 뉴스 심리.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.58-FF4B4B?logo=streamlit&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?logo=plotly&logoColor=white)
![License](https://img.shields.io/badge/License-Educational-blue)

**🔗 라이브 데모 →** https://daegyu519-quant.streamlit.app  · 아무 기기·모바일에서 바로 접속

<a href="https://daegyu519-quant.streamlit.app">
  <img src="docs/qr_live_demo.png" width="150" alt="라이브 데모 QR — 모바일로 스캔" />
</a>

> 📱 위 QR을 휴대폰 카메라로 스캔하면 대시보드가 바로 열립니다.

종목(티커)을 직접 고르면 **실시간 시세 · 기술적 지표 · 포트폴리오 최적화 · 가치/ML 분석 ·
애널리스트 목표가 · 퀀트 전략 스코어 · AI 뉴스 심리**까지 한 화면에서 분석합니다.
Docker·별도 서버 없이 파이썬 프로세스 하나로 동작합니다.

---

## ⚡ 빠른 시작

```bash
git clone https://github.com/Daegyu519/Python_Quant_Dashbord.git
cd Python_Quant_Dashbord
pip install -r requirements.txt
streamlit run dashboard.py        # → http://localhost:8501
```

API 키가 없어도 동작합니다 (무료 Yahoo Finance 데이터, 약 15분 지연).

**(선택) 뉴스 AI 심리 분석 활성화** — Gemini API 키(무료)를 설정하면
📰 페이지에서 Gemini 3단계 뉴스 분석이 켜집니다(무료 발급: aistudio.google.com/apikey):

```bash
export GEMINI_API_KEY=...        # 또는 .streamlit/secrets.toml 에 추가
```

---

## 🎯 핵심 기능 — 대시보드 17개 페이지

| 페이지 | 설명 |
|--------|------|
| 📈 **실시간** | 현재가·등락률 + 캔들차트(MA, **볼린저밴드·일목 구름대**, 거래량, **RSI·MACD** — 토글 가능) · N초 자동 갱신 |
| ⭐ **AI 추천** | 모델 기반 **예상 수익률·상승확률** 미리보기 — 확률 가중 기대수익, 톱픽 랭킹, 목표가 업사이드 |
| 🧪 **모델 검증** | 앙상블 모델의 **out-of-sample 성적표** — AUC·방향적중률·엣지·칼리브레이션·구간수익으로 "예측이 실제로 맞는가" 측정 |
| 🔎 **스크리너** | 유니버스(대형주 49)를 일괄 스캔 → **조건(RSI·모멘텀·MA·변동성) 필터·랭킹**으로 종목 발굴 |
| 📊 **수익률·낙폭** | 누적 수익률·MDD + **전문 리스크 지표**(샤프·소르티노·칼마·VaR/CVaR·베타) |
| 🔗 **상관관계** | 수익률 상관 히트맵 + 분포 (분산투자 적합도) |
| 🎯 **포트폴리오 3D** | Markowitz 효율적 프론티어 (몬테카를로 5,000개, 최적 샤프 비중 산출) |
| 🔬 **전략 최적화 3D** | MA 크로스 파라미터별 샤프 지형도 |
| 🌊 **변동성 서피스 3D** | 종목·시간별 실현 변동성 지형도 |
| 🤖 **전략 백테스트** | MA크로스·RSI·MACD·볼린저·모멘텀 5개 전략 — 룩어헤드 방지·수수료 반영·바이앤홀드 비교 |
| 💰 **가치·ML 분석** | 가치투자 점수(피오트로스키·버핏·DCF·그레이엄·알트만) + ML 예측(**펀더멘털 피처: PEG·지속ROE·FCF 통합**) + **애널리스트 목표가·매수/매도 추천·신뢰도** |
| 🎯 **눌림목 스코어** | 하나증권 '과열주 눌림목' 전략 응용 — 3개월 강세 + 1개월 눌림 + 실적/목표가 확인 |
| 💎 **크립토** | 암호화폐 전용 페이지 — 24시간 거래, 캔들·지표(MA·볼린저·일목·RSI·MACD) + 코인별 누적수익률 비교 |
| 📉 **옵션** | 옵션 체인 분석 — **변동성 스마일(IV)**, 행사가별 미결제약정, 풋/콜 비율, 최대 고통(Max Pain) |
| 💼 **모의 포트폴리오** | 페이퍼 트레이딩 — 매수 기록·미실현 손익·**SPY 대비 알파**, 보유비중·손익률 차트 (로컬 영속 저장) |
| 🌐 **거시·경제** | **FRED** 금리·물가·VIX·**장단기 금리차(침체 신호)**·실업률 (API 키 불필요) |
| 📰 **뉴스·AI 심리** | 야후 뉴스 수집 → **Gemini 3단계 파이프라인**(잡음 필터 → 핵심 이벤트 요약 → 심리점수 −1.0~+1.0) + 심리 게이지 |

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
| **대시보드 / 시각화** | Streamlit, Plotly (2D·3D) — 토스 스타일 테마 (라이트/다크 토글, 메인색 피커, Pretendard, 한국형 빨강↑/파랑↓) |
| **데이터** | yfinance (Yahoo Finance 시세·재무·뉴스), pandas, NumPy |
| **머신러닝** | XGBoost, LightGBM, scikit-learn, hmmlearn (레짐 감지), PyTorch (선택 — Apple Silicon **MPS** 추론) |
| **LLM** | Google Gemini (무료 티어) — 다단계 뉴스 분석 파이프라인 (structured outputs) |
| **퀀트** | Markowitz 포트폴리오 최적화, 백테스팅, 팩터 스코어링 |
| **(확장) 백엔드** | FastAPI, Celery, Redis, TimescaleDB, ClickHouse, Docker |

---

## 🧠 엔지니어링 하이라이트

- **표준 지표 정확도** — RSI를 단순이동평균이 아닌 **Wilder 평활(RMA)** 로 구현해 TradingView/Yahoo와 동일한 값 산출
- **포트폴리오 최적화** — 몬테카를로 5,000개 시뮬레이션으로 효율적 프론티어를 3D로 시각화하고 최적 샤프 비중 도출
- **다중 신호 결합** — 가치투자(재무 5종) + ML(상승확률·레짐) + 기술적 점수를 가중 종합해 투자 판단 산출
- **애널리스트 컨센서스 통합** — Yahoo 목표가·투자의견·EPS 추정치 변화(어닝 리비전)를 추천에 반영
- **리서치 기반 팩터 구현** — 하나증권 퀀트 리포트의 '과열주 눌림목' 전략을 분석적으로 재현(가격·목표가·실적 팩터). 원전의 한국 수급 팩터는 데이터 제약을 고려해 의도적으로 제외
- **다단계 LLM 뉴스 파이프라인** — Gemini structured outputs로 ①잡음 필터 → ②핵심 이벤트 요약 → ③정량 심리점수(−1.0~+1.0)를 3단계 분리 실행. 파싱 실패 없는 기계가독 출력 보장
- **암호화폐·옵션 분석** — 동일 무료 인프라(yfinance)로 크립토(24시간 캔들·지표) + 옵션 체인(변동성 스마일·미결제약정·풋콜비율·맥스페인)을 전용 페이지로 제공
- **모델 검증 레이어** — walk-forward OOF로 예측의 AUC·적중률·칼리브레이션을 채점하는 성적표 페이지 + 금융 로직 단위 테스트(pytest 23종)로 신뢰성 확보
- **발굴→결정→추적 워크플로** — 스크리너로 후보 발굴, 모의 포트폴리오로 매매 기록·SPY 대비 성과 추적(로컬 영속 저장), FRED 거시지표로 시장 국면 파악
- **펀더멘털 통합 ML** — PEG·지속 ROE(4년 평균+일관성)·FCF 수익률·내재가치 갭을 ML 피처에 결합하고, 종합 점수를 확률 틸트(±6%p)로 반영해 기술적 신호를 보정
- **Apple Silicon 최적화** — PyTorch 추론 디바이스를 `mps` 우선으로 명시 구성, MC-Dropout으로 점예측 대신 확률+불확실성 출력
- **방어적 설계** — 종목별 try/except, 캐시(TTL) 기반 호출 최소화, 결측치 보정

---

## 📁 프로젝트 구조

```
Python_Quant_Dashbord/
├── dashboard.py            # ⭐ 실시간 대시보드 (메인 진입점)
├── app/                    # 퀀트 엔진 (확장 플랫폼)
│   ├── ui/                 # 테마(라이트/다크·헤더바·등폭숫자) + Plotly 공통 스타일
│   ├── analytics/          # 순수 금융 계산 함수 (RSI·일목·max pain·기대수익, 테스트 대상)
│   ├── storage/            # 로컬 JSON 영속 저장 (모의 포트폴리오)
│   ├── news/               # 뉴스 수집기 + Gemini 3단계 분석 파이프라인
│   ├── data/collectors/    # Yahoo·Binance·KRX·CoinGecko 수집기
│   ├── strategies/         # MA·RSI·MACD·볼린저·모멘텀·앙상블
│   ├── backtesting/        # Numba JIT 벡터화 백테스팅
│   ├── ml/                 # 피처 엔지니어링(기술+펀더멘털) + 앙상블 + MPS 딥러닝
│   ├── value_investing/    # 가치투자 스크리너
│   ├── portfolio/          # Mean-Variance·Risk Parity·Black-Litterman
│   ├── risk/               # VaR·CVaR·포지션 한도
│   └── api/                # FastAPI (확장용)
├── scripts/
│   ├── quick_start.py      # 무료 데이터 소스 점검
│   ├── visualize.py        # 정적 차트 HTML 생성
│   └── advanced_analysis.py# 가치 + ML 종합 분석
├── .streamlit/config.toml  # 네이티브 테마 설정 (토스 라이트)
├── docs/ARCHITECTURE.md    # 아키텍처 문서
├── 설명서.md               # 코드 전체 상세 설명서(한국어)
└── requirements.txt
```

---

## 📊 데이터 소스

| 소스 | 자산 | API 키 |
|------|------|--------|
| Yahoo Finance | 미국 주식·ETF·지수, **암호화폐(BTC-USD)**, **옵션 체인**, 애널리스트 목표가/추정치, 재무제표, 종목 뉴스 | 불필요 |
| Binance | 암호화폐 (BTCUSDT 등) | 불필요 |
| KRX / FinanceDataReader | 한국 주식 | 불필요 |
| CoinGecko | 암호화폐 OHLCV | 불필요 |
| FRED (세인트루이스 연준) | 거시지표 — 금리·CPI·VIX·장단기금리차·실업률 (🌐 페이지) | 불필요 (공개 CSV) |
| Google Gemini | 뉴스 AI 분석 (📰 페이지 전용, 무료 티어, 미설정 시 뉴스 목록만 표시) | `GEMINI_API_KEY` (선택) |

---

## ⚠️ 면책

본 프로젝트는 **교육·포트폴리오 목적**으로 제작되었습니다. 표시되는 점수·추천·목표가는
**투자 조언이 아니며**, 실제 투자 판단의 근거로 사용해서는 안 됩니다. 무료 데이터는
약 15분 지연되며 정확성을 보장하지 않습니다.

---

<p align="center">
  <sub>Built with Python · Streamlit · Plotly — by <a href="https://github.com/Daegyu519">Daegyu519</a></sub>
</p>
