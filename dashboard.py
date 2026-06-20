"""
=============================================================================
실시간 미국 주식 대시보드 (Streamlit)
=============================================================================
Docker / frontend 불필요. 파이썬 프로세스 하나로 localhost 에서 실시간 갱신.

실행:
    pip install streamlit                # 최초 1회
    streamlit run dashboard.py           # → http://localhost:8501 자동 열림

페이지:
    📈 실시간          현재가 · 등락률 · 캔들차트(MA·볼린저·일목구름대/거래량/RSI/MACD)
    ⭐ AI 추천         모델 기반 예상 수익률·상승확률 미리보기 + 톱픽 랭킹
    🧪 모델 검증       앙상블 모델의 과거 적중률·칼리브레이션 성적표(out-of-sample)
    🔎 스크리너        유니버스 일괄 스캔 → 조건(RSI·모멘텀·MA) 필터·랭킹으로 종목 발굴
    📊 수익률·낙폭     종목별 누적 수익률 + 드로다운 비교
    🔗 상관관계        수익률 상관 히트맵 + 분포
    🎯 포트폴리오 3D   Markowitz 효율적 프론티어 (몬테카를로)
    🔬 전략 최적화 3D  MA 크로스 파라미터별 샤프 지형도
    🌊 변동성 서피스 3D 종목·시간별 실현 변동성
    🤖 전략 백테스트   규칙 기반 알고 트레이딩 전략 검증 (수수료·룩어헤드 방지)
    💰 가치·ML 분석    가치투자 점수 + ML 예측 (버튼 실행, 종목당 ~6초)
    💎 크립토          암호화폐 전용 (24시간 거래, 캔들·지표·누적수익률 비교)
    📉 옵션            옵션 체인 분석 (변동성 스마일·미결제약정·풋콜비율·맥스페인)
    💼 모의 포트폴리오 페이퍼 트레이딩 — 매수 기록·손익·SPY 대비 성과 (로컬 저장)
    🌐 거시·경제       FRED 금리·물가·VIX·장단기금리차 (키 불필요)
    📰 뉴스·AI 심리    종목 뉴스 수집 + Gemini 3단계 분석(필터→요약→심리점수)

각 페이지의 "📖 이 차트 읽는 법"을 펼치면 해석 가이드가 나옵니다.

📰 뉴스 AI 분석은 GEMINI_API_KEY 환경변수(또는 .streamlit/secrets.toml)가
   설정된 경우에만 활성화됩니다(무료 발급: aistudio.google.com/apikey).
   미설정 시 뉴스 목록만 표시.

⚠️ 무료 Yahoo 데이터는 약 15분 지연이며, 미국 장 시간(한국 밤~새벽)에만 가격이 움직입니다.
⚠️ 가치·ML 분석은 교육용 참고 지표이며 투자 조언이 아닙니다.
=============================================================================
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from datetime import datetime, time as dtime, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # app 패키지 임포트용

from app.ui.theme import (  # noqa: E402 — sys.path 설정 후 임포트
    ACCENT_PRESETS, BASE_LAYOUT, COLORS, DEFAULT_ACCENT, HEAT_SCALE, PALETTE,
    SCORE_SCALE, hex_to_rgba, inject_terminal_css, render_header, set_theme,
    ticker_tile,
)
from app.analytics.indicators import (  # noqa: E402 — 순수 계산 함수(테스트 대상)
    ichimoku, max_pain, model_forecast, tech_score,
)

# ─────────────────────────────────────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="실시간 주식 대시보드", page_icon="📈", layout="wide")

# CSS·테마 적용은 사이드바의 🎨 테마 컨트롤에서 (다크모드/메인색 반영 후 주입)

PRESET_TICKERS = {
    "AAPL": "애플", "MSFT": "마이크로소프트", "NVDA": "엔비디아",
    "GOOGL": "구글", "AMZN": "아마존", "TSLA": "테슬라",
    "META": "메타", "AMD": "AMD", "NFLX": "넷플릭스", "SPY": "S&P500 ETF",
}

# 암호화폐 프리셋 — yfinance 의 'BTC-USD' 형식 티커(24시간 거래)
CRYPTO_PRESETS = {
    "BTC-USD": "비트코인", "ETH-USD": "이더리움", "SOL-USD": "솔라나",
    "XRP-USD": "리플", "BNB-USD": "BNB", "DOGE-USD": "도지코인",
    "ADA-USD": "에이다", "AVAX-USD": "아발란체", "LINK-USD": "체인링크",
    "MATIC-USD": "폴리곤",
}

# 옵션 페이지에서 자주 보는 기초자산 프리셋 (유동성 큰 미국 주식·ETF)
OPTION_PRESETS = ["AAPL", "NVDA", "TSLA", "SPY", "QQQ", "AMD", "MSFT", "AMZN"]

# AI 자동 추천용 섹터별 큐레이션 종목 (GICS 기반, 섹터당 유동성 큰 대표주).
# 섹터 내 비교는 같은 거시 영향을 통제 → 상대 랭킹이 더 타당.
SECTOR_UNIVERSE: dict[str, list[str]] = {
    "기술 (Tech)": ["AAPL", "MSFT", "ORCL", "CRM", "ADBE", "ACN", "NOW", "INTU",
                   "IBM", "CSCO", "PLTR", "AMD", "NVDA", "AVGO", "QCOM", "TXN", "MU"],
    "반도체 (Semis)": ["NVDA", "AMD", "AVGO", "TSM", "QCOM", "MU", "AMAT", "LRCX",
                     "ASML", "INTC", "TXN", "KLAC", "MRVL", "ARM"],
    "통신·미디어 (Comm)": ["GOOGL", "META", "NFLX", "DIS", "CMCSA", "T", "VZ", "TMUS"],
    "임의소비재 (Cons. Disc)": ["AMZN", "TSLA", "HD", "MCD", "NKE", "SBUX", "LOW",
                            "BKNG", "TJX"],
    "필수소비재 (Staples)": ["WMT", "COST", "PG", "KO", "PEP", "PM", "MDLZ", "CL"],
    "헬스케어 (Health)": ["UNH", "JNJ", "LLY", "MRK", "ABBV", "PFE", "TMO", "ABT",
                       "DHR", "AMGN", "BMY"],
    "금융 (Financials)": ["JPM", "BAC", "WFC", "GS", "MS", "V", "MA", "AXP", "BLK",
                       "SCHW", "C"],
    "산업재 (Industrials)": ["CAT", "BA", "GE", "HON", "UPS", "RTX", "DE", "LMT",
                         "UNP", "MMM"],
    "에너지 (Energy)": ["XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX"],
}

ET = ZoneInfo("America/New_York")
KST = ZoneInfo("Asia/Seoul")

_ALL_NAMES = {**PRESET_TICKERS, **CRYPTO_PRESETS}


def name_of(sym: str) -> str:
    return _ALL_NAMES.get(sym, sym)


def is_crypto(sym: str) -> bool:
    """yfinance 크립토 티커 판별 (예: BTC-USD, ETH-USD)."""
    return sym.upper().endswith("-USD")


def help_box(md: str) -> None:
    """페이지마다 '이 차트 읽는 법' 해설 박스"""
    with st.expander("📖 이 차트 읽는 법"):
        st.markdown(md)


# ─────────────────────────────────────────────────────────────────────────────
# 데이터
# ─────────────────────────────────────────────────────────────────────────────

def us_market_status() -> tuple[bool, str]:
    now_et = datetime.now(ET)
    is_open = now_et.weekday() < 5 and dtime(9, 30) <= now_et.time() <= dtime(16, 0)
    label = (f"🟢 미국 장 열림 (뉴욕 {now_et:%H:%M})" if is_open
             else f"🔴 미국 장 닫힘 (뉴욕 {now_et:%H:%M})")
    return is_open, label


@st.cache_data(show_spinner=False, ttl=10)
def fetch_intraday(symbol: str, mode: str, _bust: int):
    tk = yf.Ticker(symbol)
    if mode == "인트라데이 (1분봉)":
        df = tk.history(period="1d", interval="1m")
    else:
        df = _yf_history(symbol, "6mo", "1d")   # 재시도+대체경로로 공백 최소화
    if df is None or df.empty:
        return None, None, None
    df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
    # 당일 미체결(장중) 봉이 NaN 으로 들어오면 가격이 NaN 이 됨 → NaN 종가 행 제거
    df = df.dropna(subset=["close"])
    if df.empty:
        return None, None, None
    last_price = float(df["close"].iloc[-1])
    try:
        prev_close = float(tk.fast_info["previous_close"])
    except Exception:
        prev_close = None
    if not prev_close or prev_close != prev_close:        # None 또는 NaN
        prev_close = float(df["open"].iloc[0]) if len(df) else last_price
    return df, last_price, prev_close


def _yf_history(symbol: str, period: str = "2y", interval: str = "1d"):
    """
    신뢰 소스(Yahoo)에서 가격을 강건하게 조회 — 공백 최소화.
      ① Ticker.history 재시도 → ② 대체 엔드포인트 yf.download.
    둘 다 실패하면 None (없는 값을 지어내지 않음).
    """
    import time as _t
    for attempt in range(2):
        try:
            df = yf.Ticker(symbol).history(period=period, interval=interval)
            if df is not None and not df.empty:
                return df
        except Exception:
            pass
        _t.sleep(0.4 * (attempt + 1))
    try:
        df = yf.download(symbol, period=period, interval=interval,
                         auto_adjust=True, progress=False, threads=False)
        if df is not None and not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            return df
    except Exception:
        pass
    return None


@st.cache_data(show_spinner=False, ttl=300)
def fetch_daily(symbol: str) -> pd.DataFrame | None:
    df = _yf_history(symbol, "2y", "1d")
    if df is None or df.empty:
        return None
    df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
    # 장중/당일 미체결 봉이 NaN 으로 들어오는 경우 제거 → iloc[-1] 가 NaN 되는 문제 방지
    df = df.dropna(subset=["close"])
    if df.empty:
        return None
    df.index = pd.to_datetime(df.index)
    return df


@st.cache_data(show_spinner=False, ttl=600)
def fetch_daily_batch(symbols: tuple[str, ...]) -> dict[str, pd.DataFrame]:
    """
    여러 종목 일봉을 yfinance 로 '일괄' 다운로드(1회 요청) → 종목별 dict.

    스크리너처럼 수십 종목을 훑을 때 종목당 .history() 를 순차 호출하면 야후가
    대량 요청을 throttle 해 대부분 빈 응답이 온다. yf.download(리스트) 는 한 번의
    배치 요청이라 훨씬 안정적이다.
    """
    if not symbols:
        return {}
    try:
        raw = yf.download(list(symbols), period="1y", interval="1d",
                          auto_adjust=True, group_by="ticker", threads=True,
                          progress=False)
    except Exception:
        return {}
    out: dict[str, pd.DataFrame] = {}
    single = len(symbols) == 1
    for s in symbols:
        try:
            df = raw if single else raw[s]
            df = (df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
                  .dropna())
            if len(df) >= 60:
                out[s] = df
        except Exception:
            continue
    return out


def load_frames(symbols: list[str]) -> dict[str, pd.DataFrame]:
    frames = {}
    for s in symbols:
        df = fetch_daily(s)
        if df is not None and len(df) > 50:
            frames[s] = df
    return frames


def run_async(coro):
    try:
        return asyncio.run(coro)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()


class _Frame:
    """ML 모델(ImprovedEnsembleModel)이 기대하는 최소 인터페이스(.data/.symbol)"""
    def __init__(self, df: pd.DataFrame, symbol: str):
        self.data = df
        self.symbol = symbol


def verdict(final: float) -> str:
    if final >= 65:   return "🟢 매수 추천"
    if final >= 50:   return "🟡 보유/관망"
    if final >= 35:   return "🟠 관망"
    return "🔴 회피"


RECO_KOR = {
    "strong_buy": "적극 매수", "buy": "매수", "hold": "보유",
    "underperform": "비중축소", "sell": "매도", "strong_sell": "적극 매도",
}


def reco_kor(key) -> str:
    return RECO_KOR.get(key, key or "-")


@st.cache_data(show_spinner=False, ttl=600)
def fetch_analyst(symbol: str) -> dict:
    """Yahoo 애널리스트 목표가·투자의견(전문가 컨센서스)."""
    try:
        info = yf.Ticker(symbol).info
    except Exception:
        return {}
    keys = ["currentPrice", "targetMeanPrice", "targetHighPrice", "targetLowPrice",
            "recommendationKey", "recommendationMean", "numberOfAnalystOpinions"]
    return {k: info.get(k) for k in keys}


@st.cache_data(show_spinner=False, ttl=600)
def fetch_eps_revision(symbol: str):
    """올해(0y) EPS 추정치의 30일 전 대비 변화율(%). 양수=상향(어닝 리비전 업)."""
    try:
        et = yf.Ticker(symbol).eps_trend
        cur = float(et.loc["0y", "current"])
        ago = float(et.loc["0y", "30daysAgo"])
        if ago not in (0, None) and ago == ago:
            return (cur - ago) / abs(ago) * 100
    except Exception:
        pass
    return None


# ─────────────────────────────────────────────────────────────────────────────
# 차트 — 실시간 캔들
# ─────────────────────────────────────────────────────────────────────────────

def chart_candlestick(df: pd.DataFrame, symbol: str, name: str,
                      show_bb: bool = True, show_ichimoku: bool = True,
                      show_macd: bool = True) -> go.Figure:
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    df["ma20"] = df["close"].rolling(20).mean()
    df["ma50"] = df["close"].rolling(50).mean()

    # 볼린저 밴드 (20일 SMA ±2σ). 중심선 = MA20.
    bb_std = df["close"].rolling(20).std()
    df["bb_up"] = df["ma20"] + 2 * bb_std
    df["bb_dn"] = df["ma20"] - 2 * bb_std

    # RSI(14) — 표준 Wilder 평활(RMA, alpha=1/14). TradingView/Yahoo 등과 동일.
    delta = df["close"].diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    roll_down = down.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    rs = roll_up / roll_down
    rsi = 100 - (100 / (1 + rs))
    df["rsi"] = rsi.where(roll_down != 0, 100.0)

    # MACD(12, 26, 9)
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    macd_sig = macd_line.ewm(span=9, adjust=False).mean()
    macd_hist = macd_line - macd_sig

    n_rows = 4 if show_macd else 3
    row_heights = [0.50, 0.14, 0.18, 0.18] if show_macd else [0.6, 0.2, 0.2]
    # 가격 패널은 차트 제목과 중복이라 소제목 생략 → 범례 영역 깔끔하게
    titles = ["", "거래량", "RSI(14)"] + (["MACD(12,26,9)"] if show_macd else [])
    fig = make_subplots(rows=n_rows, cols=1, shared_xaxes=True,
                        row_heights=row_heights, vertical_spacing=0.045,
                        subplot_titles=titles)

    # ── 일목균형표 구름대 (가장 뒤에 깔리도록 먼저 추가) ────────────────────
    if show_ichimoku and len(df) >= 52:
        ichi, ext_idx = ichimoku(df)
        a, b = ichi["span_a"].values, ichi["span_b"].values
        # 두 색 구름: A≥B(양운, 빨강) / A<B(음운, 파랑) — 마스킹 트릭으로 분리 채색
        a_up = np.where(a >= b, a, b)   # 양운 구간만 폭이 생김
        a_dn = np.where(a < b, a, b)    # 음운 구간만 폭이 생김
        fig.add_trace(go.Scatter(x=ext_idx, y=b, line=dict(width=0),
            hoverinfo="skip", showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=ext_idx, y=a_up, line=dict(width=0),
            fill="tonexty", fillcolor=hex_to_rgba(COLORS["up"], 0.14),
            name="일목 구름", hoverinfo="skip", showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=ext_idx, y=b, line=dict(width=0),
            hoverinfo="skip", showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=ext_idx, y=a_dn, line=dict(width=0),
            fill="tonexty", fillcolor=hex_to_rgba(COLORS["down"], 0.14),
            hoverinfo="skip", showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=ext_idx, y=ichi["tenkan"],
            line=dict(color=hex_to_rgba(COLORS["up"], 0.65), width=1, dash="dot"),
            name="전환선", hoverinfo="skip", showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=ext_idx, y=ichi["kijun"],
            line=dict(color=hex_to_rgba(COLORS["down"], 0.65), width=1, dash="dot"),
            name="기준선", hoverinfo="skip", showlegend=False), row=1, col=1)

    # ── 볼린저 밴드 (은은하지만 보이게) ─────────────────────────────────────
    if show_bb:
        bb_col = hex_to_rgba(COLORS["blue"], 0.45)
        fig.add_trace(go.Scatter(x=df.index, y=df["bb_up"],
            line=dict(color=bb_col, width=1),
            name="BB 상단", showlegend=False, hoverinfo="skip"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["bb_dn"],
            line=dict(color=bb_col, width=1),
            fill="tonexty", fillcolor=hex_to_rgba(COLORS["blue"], 0.05),
            name="볼린저밴드", showlegend=False, hoverinfo="skip"), row=1, col=1)

    fig.add_trace(go.Candlestick(
        x=df.index, open=df["open"], high=df["high"], low=df["low"], close=df["close"],
        increasing_line_color=COLORS["up"], decreasing_line_color=COLORS["down"], name="가격",
    ), row=1, col=1)
    for ma, color, label in [("ma20", COLORS["warn"], "MA20"), ("ma50", COLORS["purple"], "MA50")]:
        fig.add_trace(go.Scatter(x=df.index, y=df[ma], line=dict(color=color, width=1.4),
                                 name=label), row=1, col=1)
    vc = [COLORS["up"] if c >= o else COLORS["down"] for c, o in zip(df["close"], df["open"])]
    fig.add_trace(go.Bar(x=df.index, y=df["volume"], marker_color=vc, opacity=0.7,
                         name="거래량", showlegend=False), row=2, col=1)
    # RSI — 과매수/과매도 영역 음영 + 밝은 라인으로 시인성 강화
    fig.add_hrect(y0=70, y1=100, fillcolor=hex_to_rgba(COLORS["up"], 0.07),
                  line_width=0, row=3, col=1)
    fig.add_hrect(y0=0, y1=30, fillcolor=hex_to_rgba(COLORS["down"], 0.07),
                  line_width=0, row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["rsi"],
                             line=dict(color="#c4b5fd", width=1.9),
                             name="RSI", showlegend=False), row=3, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color=COLORS["up"], opacity=0.55, row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color=COLORS["down"], opacity=0.55, row=3, col=1)

    # ── MACD 패널 ───────────────────────────────────────────────────────────
    if show_macd:
        hc = [hex_to_rgba(COLORS["up"], 0.8) if v >= 0 else hex_to_rgba(COLORS["down"], 0.8)
              for v in macd_hist]
        fig.add_trace(go.Bar(x=df.index, y=macd_hist, marker_color=hc,
                             name="히스토그램", showlegend=False), row=4, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=macd_line,
            line=dict(color="#38bdf8", width=1.8),
            name="MACD", showlegend=False), row=4, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=macd_sig,
            line=dict(color=COLORS["warn"], width=1.5),
            name="시그널(9)", showlegend=False), row=4, col=1)
        fig.add_hline(y=0, line_dash="dot", line_color=COLORS["muted"],
                      opacity=0.4, row=4, col=1)

    fig.update_layout(**BASE_LAYOUT,
        title=dict(text=f"{name} ({symbol})", font=dict(size=16, color=COLORS["accent"])),
        height=700 if show_macd else 600, xaxis_rangeslider_visible=False,
        margin=dict(l=10, r=10, t=60, b=10),
        legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", y=1.02))
    fig.update_xaxes(gridcolor=COLORS["grid"])
    fig.update_yaxes(gridcolor=COLORS["grid"])
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 차트 — 수익률·낙폭 / 상관관계 / 3D
# ─────────────────────────────────────────────────────────────────────────────

def chart_equity(frames: dict) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3],
                        vertical_spacing=0.05, subplot_titles=["누적 수익률 (%)", "낙폭 (Drawdown %)"])
    for i, (sym, df) in enumerate(frames.items()):
        ret = df["close"].pct_change().fillna(0)
        cum = (1 + ret).cumprod()
        dd = (cum - cum.cummax()) / cum.cummax() * 100
        color = PALETTE[i % len(PALETTE)]
        fig.add_trace(go.Scatter(x=df.index, y=(cum - 1) * 100, line=dict(color=color, width=2),
                                 name=name_of(sym)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=dd, line=dict(color=color, width=1),
                                 fill="tozeroy", fillcolor=hex_to_rgba(color, 0.08),
                                 name=sym, showlegend=False), row=2, col=1)
    fig.add_hline(y=0, line_dash="dot", line_color=COLORS["muted"], opacity=0.3, row=1, col=1)
    fig.update_layout(**BASE_LAYOUT,
        title=dict(text="종목별 누적 수익률 & 낙폭 (최근 2년)", font=dict(size=17, color=COLORS["accent"])),
        height=650, legend=dict(bgcolor="rgba(0,0,0,0)"))
    fig.update_yaxes(ticksuffix="%", row=1, col=1, gridcolor=COLORS["grid"])
    fig.update_yaxes(ticksuffix="%", row=2, col=1, gridcolor=COLORS["grid"])
    fig.update_xaxes(gridcolor=COLORS["grid"])
    return fig


def chart_correlation(frames: dict) -> go.Figure:
    d = {}
    for sym, df in frames.items():
        ret = df["close"].pct_change().dropna()
        ret.index = pd.to_datetime(ret.index).tz_localize(None).normalize()
        d[name_of(sym)] = ret.groupby(ret.index).last()
    ret_df = pd.DataFrame(d).dropna()
    corr = ret_df.corr()
    fig = make_subplots(rows=1, cols=2, column_widths=[0.5, 0.5],
                        subplot_titles=["수익률 상관관계 행렬", "수익률 분포 비교"])
    fig.add_trace(go.Heatmap(
        z=corr.values, x=corr.columns.tolist(), y=corr.index.tolist(),
        colorscale=[[0.0, COLORS["down"]], [0.5, COLORS["bg"]], [1.0, COLORS["up"]]],
        zmin=-1, zmax=1, text=np.round(corr.values, 2), texttemplate="%{text}",
        textfont=dict(size=12), colorbar=dict(title="상관계수", x=0.45, thickness=12)), row=1, col=1)
    for i, col in enumerate(ret_df.columns):
        fig.add_trace(go.Histogram(x=ret_df[col] * 100, name=col, opacity=0.65, nbinsx=60,
                                   marker_color=PALETTE[i % len(PALETTE)],
                                   histnorm="probability density"), row=1, col=2)
    fig.update_layout(**BASE_LAYOUT,
        title=dict(text="종목 간 상관관계 & 수익률 분포", font=dict(size=17, color=COLORS["accent"])),
        height=550, barmode="overlay")
    fig.update_xaxes(title_text="일별 수익률 (%)", row=1, col=2, ticksuffix="%")
    return fig


def chart_frontier(frames: dict):
    returns_dict = {}
    for sym, df in frames.items():
        ret = df["close"].pct_change().dropna()
        ret.index = pd.to_datetime(ret.index).tz_localize(None).normalize()
        returns_dict[sym] = ret.groupby(ret.index).last()
    ret_df = pd.DataFrame(returns_dict).dropna()
    mu = ret_df.mean().values * 252
    cov = ret_df.cov().values * 252
    n = len(mu)

    np.random.seed(42)
    N_SIM = 5000
    vols, rets, sharpes, weights = [], [], [], []
    for _ in range(N_SIM):
        w = np.random.dirichlet(np.ones(n))
        r = float(w @ mu)
        v = float(np.sqrt(w @ cov @ w))
        vols.append(v * 100); rets.append(r * 100)
        sharpes.append((r - 0.02) / v if v > 0 else 0); weights.append(w)
    vols, rets, sharpes = np.array(vols), np.array(rets), np.array(sharpes)
    bi = int(np.argmax(sharpes))
    labels = list(frames.keys())
    hover = ["<br>".join(f"{labels[j]}: {weights[i][j]:.1%}" for j in range(n)) for i in range(N_SIM)]

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=vols, y=rets, z=sharpes, mode="markers",
        marker=dict(size=2.5, color=sharpes, colorscale="Viridis",
                    colorbar=dict(title="샤프", thickness=15, x=1.02), opacity=0.7),
        text=hover,
        hovertemplate="변동성:%{x:.1f}%<br>수익률:%{y:.1f}%<br>샤프:%{z:.2f}<br>%{text}<extra></extra>",
        name="포트폴리오"))
    fig.add_trace(go.Scatter3d(x=[vols[bi]], y=[rets[bi]], z=[sharpes[bi]], mode="markers+text",
        marker=dict(size=10, color=COLORS["down"], symbol="diamond"),
        text=["★ 최적"], textfont=dict(color=COLORS["accent"], size=13),
        name=f"최적 샤프 ({sharpes[bi]:.2f})"))
    fig.update_layout(**BASE_LAYOUT,
        title=dict(text="3D 효율적 프론티어 — Markowitz 포트폴리오 최적화",
                   font=dict(size=17, color=COLORS["accent"])),
        scene=dict(
            xaxis=dict(title="변동성(연,%)", backgroundcolor=COLORS["panel"], gridcolor=COLORS["grid"]),
            yaxis=dict(title="기대수익(연,%)", backgroundcolor=COLORS["panel"], gridcolor=COLORS["grid"]),
            zaxis=dict(title="샤프 비율", backgroundcolor=COLORS["panel"], gridcolor=COLORS["grid"]),
            bgcolor=COLORS["bg"], camera=dict(eye=dict(x=1.8, y=-1.6, z=0.9))),
        height=720)
    alloc = "  ·  ".join(f"{labels[j]} {weights[bi][j]:.0%}" for j in range(n))
    return fig, alloc


def chart_param_surface(df: pd.DataFrame, sym: str) -> tuple[go.Figure, dict]:
    """MA 크로스 파라미터별 샤프 지형도 + 최적 조합 정보.

    Returns:
        (fig, best) — best = {fast, slow, sharpe, n_combos}
    """
    closes = df["close"].values
    fv, sv, zv = [], [], []
    for fast in range(5, 60, 5):
        for slow in range(20, 200, 10):
            if fast >= slow:
                continue
            ma_f = pd.Series(closes).rolling(fast).mean().values
            ma_s = pd.Series(closes).rolling(slow).mean().values
            pos = np.where(ma_f > ma_s, 1.0, -1.0); pos = np.roll(pos, 1); pos[0] = 0
            ret = np.diff(closes) / closes[:-1]
            strat = pos[:-1] * ret
            sh = 0.0 if strat.std() == 0 else float((strat.mean() / strat.std()) * np.sqrt(252))
            fv.append(fast); sv.append(slow); zv.append(np.clip(sh, -3, 4))
    fv, sv, zv = map(np.array, (fv, sv, zv))
    bi = int(np.argmax(zv))
    best = dict(fast=int(fv[bi]), slow=int(sv[bi]), sharpe=float(zv[bi]), n_combos=int(len(zv)))

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=fv, y=sv, z=zv, mode="markers",
        marker=dict(size=5, color=zv,
                    colorscale=[[0.0, COLORS["down"]], [0.5, "#e5e8eb"], [1.0, COLORS["up"]]],
                    # 컬러바: 오른쪽 가운데, 높이 60% — 범례와 안 겹치게 분리
                    colorbar=dict(title=dict(text="샤프", side="right"),
                                  thickness=14, len=0.6, x=1.02, y=0.5),
                    opacity=0.85),
        hovertemplate="단기:%{x}일<br>장기:%{y}일<br>샤프:%{z:.3f}<extra></extra>", name="파라미터 조합"))
    fig.add_trace(go.Scatter3d(x=[fv[bi]], y=[sv[bi]], z=[zv[bi]], mode="markers+text",
        marker=dict(size=14, color=COLORS["accent"], symbol="diamond", line=dict(color="white", width=2)),
        text=[f"★ {fv[bi]}/{sv[bi]}"], textposition="top center",
        textfont=dict(color=COLORS["accent"], size=12),
        name=f"★ 최적 (단기{fv[bi]}·장기{sv[bi]})"))
    fig.update_layout(**BASE_LAYOUT,
        title=dict(text=f"3D MA 크로스 파라미터 최적화 — {name_of(sym)} ({sym})<br>"
                        f"<sub>★ 최적: 단기 {fv[bi]}일 · 장기 {sv[bi]}일 · 샤프 {zv[bi]:.2f}</sub>",
                   font=dict(size=16, color=COLORS["accent"]), x=0.0, xanchor="left"),
        # 범례를 좌측 하단 가로 배치 → 우측 컬러바와 충돌 제거
        legend=dict(orientation="h", yanchor="bottom", y=0.0, xanchor="left", x=0.0,
                    bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
        scene=dict(
            xaxis=dict(title="단기 MA(일)", backgroundcolor=COLORS["panel"], gridcolor=COLORS["grid"]),
            yaxis=dict(title="장기 MA(일)", backgroundcolor=COLORS["panel"], gridcolor=COLORS["grid"]),
            zaxis=dict(title="샤프 비율", backgroundcolor=COLORS["panel"], gridcolor=COLORS["grid"]),
            bgcolor=COLORS["bg"], camera=dict(eye=dict(x=2.0, y=-1.8, z=1.0))),
        margin=dict(l=0, r=0, t=70, b=0), height=720)
    return fig, best


def chart_vol_surface(frames: dict) -> go.Figure:
    vol_data = {}
    for sym, df in frames.items():
        vol = (df["close"].pct_change().rolling(30).std() * np.sqrt(252) * 100).dropna()
        vol.index = pd.to_datetime(vol.index).tz_localize(None).normalize()
        vol_data[sym] = vol
    common = None
    for vol in vol_data.values():
        s = set(vol.index.date)
        common = s if common is None else common & s
    if not common:
        return go.Figure()
    date_list = sorted(common)
    date_list = date_list[::5] if len(date_list) > 400 else date_list
    date_strs = [str(d) for d in date_list]
    labels = list(vol_data.keys())
    n_sym, n_time = len(labels), len(date_list)
    # 금융 데이터 무결성: 결측을 보간/채우지 않음 — 실제 데이터만 사용하고
    # 없는 칸은 NaN(서피스에서 빈 곳)으로 둔다.
    Z = np.full((n_sym, n_time), np.nan)
    for i, sym in enumerate(labels):
        by_date = {d.date(): v for d, v in vol_data[sym].items()}
        for j, d in enumerate(date_list):
            Z[i, j] = by_date.get(d, np.nan)
    step = max(1, n_time // 8)
    tick_vals = list(range(0, n_time, step))
    tick_text = [date_strs[i][:7] for i in tick_vals]
    fig = go.Figure(go.Surface(
        x=list(range(n_time)), y=list(range(n_sym)), z=Z,
        colorscale=HEAT_SCALE,
        colorbar=dict(title=dict(text="변동성(%)", side="right"), thickness=16, x=1.02),
        opacity=0.92,
        contours=dict(z=dict(show=True, usecolormap=True, highlightcolor="white", project_z=True)),
        hovertemplate="시간:%{x}<br>종목:%{y}<br>변동성:%{z:.1f}%<extra></extra>"))
    fig.update_layout(**BASE_LAYOUT,
        title=dict(text="3D 실현 변동성 서피스 — 종목별 리스크 지형도",
                   font=dict(size=17, color=COLORS["accent"])),
        scene=dict(
            xaxis=dict(title="시간", tickvals=tick_vals, ticktext=tick_text,
                       backgroundcolor=COLORS["panel"], gridcolor=COLORS["grid"]),
            yaxis=dict(title="종목", tickvals=list(range(n_sym)), ticktext=[name_of(s) for s in labels],
                       backgroundcolor=COLORS["panel"], gridcolor=COLORS["grid"]),
            zaxis=dict(title="연환산 변동성(%)", backgroundcolor=COLORS["panel"], gridcolor=COLORS["grid"]),
            bgcolor=COLORS["bg"], camera=dict(eye=dict(x=2.0, y=-2.2, z=1.3))),
        height=720)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 차트 — 가치·ML (advanced)
# ─────────────────────────────────────────────────────────────────────────────

def chart_value_radar(value_results: dict) -> go.Figure:
    cats = ["피오트로스키", "버핏체크", "DCF", "그레이엄", "알트만Z"]
    fig = go.Figure()
    for i, (sym, vs) in enumerate(value_results.items()):
        if vs is None:
            continue
        vals = [vs.piotroski_score / 9 * 100, vs.buffett_score / 10 * 100,
                float(np.clip(vs.dcf_upside + 50, 0, 100)),
                float(np.clip(vs.graham_upside + 50, 0, 100)),
                float(np.clip(vs.altman_z / 5 * 100, 0, 100))]
        vals.append(vals[0])
        c = PALETTE[i % len(PALETTE)]
        fig.add_trace(go.Scatterpolar(r=vals, theta=cats + [cats[0]], fill="toself",
            fillcolor=hex_to_rgba(c, 0.18), line=dict(color=c, width=2),
            name=f"{name_of(sym)} ({vs.total_score:.0f})"))
    fig.update_layout(paper_bgcolor=COLORS["bg"], plot_bgcolor=COLORS["panel"],
        font=dict(color=COLORS["text"], family="monospace"),
        polar=dict(bgcolor=COLORS["panel"],
            radialaxis=dict(visible=True, range=[0, 100], gridcolor=COLORS["grid"], tickfont=dict(size=9)),
            angularaxis=dict(gridcolor=COLORS["grid"])),
        title=dict(text="가치투자 레이더 (5개 축, 100=최우량)", font=dict(size=16, color=COLORS["accent"])),
        legend=dict(bgcolor="rgba(255,255,255,0.6)"), height=520)
    return fig


def chart_ranking(res: dict) -> go.Figure:
    syms = sorted(res.keys(), key=lambda s: res[s]["final"], reverse=True)
    finals = [res[s]["final"] for s in syms]
    colors = [COLORS["up"] if f >= 65 else COLORS["warn"] if f >= 50 else COLORS["down"] for f in finals]
    fig = make_subplots(rows=1, cols=2, column_widths=[0.55, 0.45],
                        subplot_titles=["📊 종합 투자 점수 랭킹", "💰 가치점수 vs ML 상승확률"])
    fig.add_trace(go.Bar(x=[name_of(s) for s in syms], y=finals, marker_color=colors,
        text=[f"{f:.0f}" for f in finals], textposition="outside", showlegend=False), row=1, col=1)
    fig.add_hline(y=65, line_dash="dash", line_color=COLORS["up"], opacity=0.5, row=1, col=1)
    fig.add_hline(y=50, line_dash="dash", line_color=COLORS["warn"], opacity=0.5, row=1, col=1)
    val_sc = [res[s]["vs"].total_score if res[s]["vs"] else 50 for s in syms]
    ml_sc = [res[s]["pred"].up_probability * 100 if res[s]["pred"] else 50 for s in syms]
    fig.add_trace(go.Scatter(x=val_sc, y=ml_sc, mode="markers+text",
        marker=dict(size=16, color=finals, colorscale=SCORE_SCALE, cmin=0, cmax=100,
                    line=dict(color="white", width=1)),
        text=syms, textposition="top center", showlegend=False), row=1, col=2)
    fig.add_hline(y=50, line_dash="dot", line_color=COLORS["muted"], opacity=0.3, row=1, col=2)
    fig.add_vline(x=50, line_dash="dot", line_color=COLORS["muted"], opacity=0.3, row=1, col=2)
    fig.update_layout(**BASE_LAYOUT, height=520,
        title=dict(text="종합 투자 랭킹", font=dict(size=16, color=COLORS["accent"])))
    fig.update_yaxes(range=[0, 100], row=1, col=1, gridcolor=COLORS["grid"])
    fig.update_xaxes(title_text="가치 점수", row=1, col=2, gridcolor=COLORS["grid"])
    fig.update_yaxes(title_text="ML 상승확률(%)", row=1, col=2, gridcolor=COLORS["grid"])
    return fig


def chart_confidence(res: dict) -> go.Figure:
    syms = [s for s in res if res[s]["pred"] is not None]
    fig = make_subplots(rows=1, cols=2, subplot_titles=["🤖 ML 신뢰도", "🌊 레짐 확률 분포"])
    fig.add_trace(go.Bar(x=[name_of(s) for s in syms],
        y=[res[s]["pred"].confidence * 100 for s in syms],
        marker_color=[PALETTE[i % len(PALETTE)] for i in range(len(syms))],
        name="신뢰도", showlegend=False), row=1, col=1)
    for reg, c in {"상승장": COLORS["up"], "횡보장": COLORS["warn"], "하락장": COLORS["down"]}.items():
        fig.add_trace(go.Bar(x=[name_of(s) for s in syms],
            y=[res[s]["pred"].regime_proba.get(reg, 0) * 100 for s in syms],
            name=reg, marker_color=c), row=1, col=2)
    fig.update_layout(**BASE_LAYOUT, height=460, barmode="stack",
        title=dict(text="ML 신뢰도 & 시장 레짐", font=dict(size=16, color=COLORS["accent"])),
        legend=dict(bgcolor="rgba(255,255,255,0.6)"))
    fig.update_yaxes(ticksuffix="%", gridcolor=COLORS["grid"])
    return fig


@st.cache_data(ttl=21600, show_spinner=False)   # 6시간 캐시 → 같은 종목 재분석은 즉시
def analyze_one(symbol: str) -> dict | None:
    """종목 1개의 가치투자 + ML + 애널리스트 분석 (무거움). 결과를 캐싱한다."""
    from app.value_investing.screener import ValueInvestingScreener
    from app.ml.models.improved_ensemble import ImprovedEnsembleModel
    from app.ml.features.fundamental_features import fetch_fundamental_features

    df = fetch_daily(symbol)
    if df is None or len(df) <= 50:
        return None
    try:
        vs = run_async(ValueInvestingScreener().analyze(symbol))
    except Exception:
        vs = None
    # 펀더멘털 피처 (PEG·지속ROE·FCF 등) — ML 피처 + 확률 틸트에 사용
    try:
        fund = fetch_fundamental_features(symbol)
    except Exception:
        fund = {}
    if vs and vs.dcf_upside is not None:
        fund["dcf_upside"] = float(vs.dcf_upside)   # 내재가치 갭(%) 주입
    try:
        model = ImprovedEnsembleModel()
        model.fit(_Frame(df, symbol), fundamentals=fund)
        pred = model.predict(_Frame(df, symbol),
                             value_score=(vs.total_score if vs else 50.0),
                             fundamentals=fund)
    except Exception:
        pred = None
    ts, _ = tech_score(df)
    val = vs.total_score if vs else 50.0
    mlp = (pred.up_probability * 100) if pred else 50.0
    final = float(np.clip(ts * 0.20 + val * 0.40 + mlp * 0.40, 0, 100))
    analyst = fetch_analyst(symbol)
    closes = df["close"].dropna()              # 마지막 유효 종가 사용 (NaN 봉 방지)
    cur = float(closes.iloc[-1])
    ann_vol = float(df["close"].pct_change().dropna().std() * np.sqrt(252))
    move = cur * ann_vol * np.sqrt(21 / 252)   # 1개월(21거래일) ±1σ 예상 변동폭

    # v2: 펀더멘털 디테일 + 종합 점수 포함 (레짐 수정 반영을 위한 캐시 무효화 겸용)
    from app.ml.features.fundamental_features import fundamental_score
    return dict(vs=vs, pred=pred, tech=ts, final=final,
                analyst=analyst, cur=cur, band=(cur - move, cur + move),
                fund=fund, fund_score=fundamental_score(fund))


def run_advanced(symbols: list[str], frames: dict) -> dict:
    """종목별 가치·ML 분석. analyze_one(캐시)을 호출 → 재분석 시 즉시."""
    res = {}
    prog = st.progress(0.0, text="분석 준비 중...")
    for i, sym in enumerate(symbols):
        if sym not in frames:
            continue
        prog.progress(i / len(symbols),
                      text=f"{name_of(sym)} 분석 중… (첫 실행은 종목당 수십 초, 이후 캐시되면 즉시)")
        r = analyze_one(sym)
        if r is not None:
            res[sym] = r
        prog.progress((i + 1) / len(symbols), text=f"{name_of(sym)} 완료")
    prog.empty()
    return res


# ─────────────────────────────────────────────────────────────────────────────
# 차트 — 눌림목 스코어 (하나증권 '과열주 눌림목' 전략 응용)
#   원전: 하나증권 이경수, 실전 퀀트(Quant MP) 2026.05.27 "모멘텀 발산기, 과열주 눌림목"
#   핵심: 3개월 강세(과열) + 1개월 단기 눌림 종목을, 실적·목표주가 상향으로 확인.
#   ※ 원 리포트의 수급(개인/기관/외인) 팩터는 미국 무료데이터로 불가 → 의도적으로 제외.
# ─────────────────────────────────────────────────────────────────────────────

def _lin(x, lo, hi) -> float:
    """lo→0, hi→100 선형 매핑(클립). 결측이면 중립 50."""
    if x is None or x != x:
        return 50.0
    if hi == lo:
        return 50.0
    return float(np.clip((x - lo) / (hi - lo) * 100, 0, 100))


def compute_pullback(df: pd.DataFrame, sym: str) -> dict:
    c = df["close"]; v = df["volume"]
    cur = float(c.iloc[-1])
    n = len(c)
    ret_3m = (cur / float(c.iloc[-64]) - 1) * 100 if n > 64 else (cur / float(c.iloc[0]) - 1) * 100
    ret_1m = (cur / float(c.iloc[-22]) - 1) * 100 if n > 22 else (cur / float(c.iloc[0]) - 1) * 100
    dollar = c * v
    turn_1m = float(dollar.iloc[-21:].mean())
    turn_3m = float(dollar.iloc[-63:].mean()) if n > 63 else float(dollar.mean())
    turnover_chg = (turn_1m / turn_3m - 1) * 100 if turn_3m else 0.0
    above_ma50 = cur > float(c.iloc[-50:].mean()) if n >= 50 else cur > float(c.mean())

    a = fetch_analyst(sym) or {}
    tp = a.get("targetMeanPrice")
    tp_up = (tp - cur) / cur * 100 if tp else None
    eps_rev = fetch_eps_revision(sym)

    # 팩터별 0~100 점수
    s_mom = _lin(ret_3m, -30, 30)                       # 3개월 강세
    s_dip = float(np.clip(100 - abs(ret_1m + 8) * 6, 0, 100))  # 1개월 눌림(−8%에서 최고)
    s_turn = _lin(turnover_chg, -30, 60)                # 거래대금 증가
    s_tp = _lin(tp_up, -10, 40)                         # 목표주가 상승여력
    s_eps = _lin(eps_rev, -5, 10)                       # 실적 추정치 상향

    total = (0.30 * s_mom + 0.25 * s_dip + 0.18 * s_tp + 0.15 * s_eps + 0.12 * s_turn)
    total = float(np.clip(total, 0, 100))

    # 가드레일 포함 판정 (리포트 로직 충실)
    if ret_3m < 5:
        v_txt = "⚪ 추세 미형성"
    elif ret_1m <= -25 and not above_ma50:
        v_txt = "🔴 추세 이탈(낙폭과대)"
    elif ret_1m >= 5:
        v_txt = "🟠 눌림 없음(과열 추격 주의)"
    elif total >= 62:
        v_txt = "🟢 강세 눌림목(매수 후보)"
    else:
        v_txt = "🟡 관찰"

    return dict(sym=sym, cur=cur, ret_3m=ret_3m, ret_1m=ret_1m, turnover_chg=turnover_chg,
                tp_up=tp_up, eps_rev=eps_rev, total=total, verdict=v_txt, above_ma50=above_ma50)


def chart_pullback(rows: list) -> go.Figure:
    fig = go.Figure()
    # '눌림목 존' 음영 (3M 강세 + 1M 적정 눌림)
    fig.add_shape(type="rect", x0=10, x1=max(40, max(r["ret_3m"] for r in rows) + 5),
                  y0=-18, y1=-3, fillcolor="rgba(49,130,246,0.08)",
                  line=dict(color="rgba(49,130,246,0.45)", width=1))
    fig.add_annotation(x=10, y=-3, text="  ◀ 눌림목 존(강세+눌림)", showarrow=False,
                       font=dict(color=COLORS["accent"], size=11), xanchor="left", yanchor="bottom")
    fig.add_hline(y=0, line_dash="dot", line_color=COLORS["muted"], opacity=0.3)
    fig.add_vline(x=0, line_dash="dot", line_color=COLORS["muted"], opacity=0.3)
    fig.add_trace(go.Scatter(
        x=[r["ret_3m"] for r in rows], y=[r["ret_1m"] for r in rows],
        mode="markers+text", text=[r["sym"] for r in rows], textposition="top center",
        marker=dict(size=18, color=[r["total"] for r in rows], colorscale=SCORE_SCALE,
                    cmin=0, cmax=100, line=dict(color="white", width=1),
                    colorbar=dict(title="눌림목<br>점수", thickness=14)),
        hovertemplate="%{text}<br>3개월:%{x:.1f}%<br>1개월:%{y:.1f}%<extra></extra>"))
    fig.update_layout(**BASE_LAYOUT, height=560,
        title=dict(text="눌림목 맵 — 3개월 강세(가로) vs 1개월 눌림(세로)",
                   font=dict(size=16, color=COLORS["accent"])))
    fig.update_xaxes(title_text="3개월 수익률 (%) — 오른쪽=강세", ticksuffix="%", gridcolor=COLORS["grid"])
    fig.update_yaxes(title_text="1개월 수익률 (%) — 아래=눌림", ticksuffix="%", gridcolor=COLORS["grid"])
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 전문 리스크·성과 지표 (수익률·낙폭 페이지)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def benchmark_returns() -> pd.Series | None:
    """베타 계산용 벤치마크(S&P500=SPY) 일별 수익률."""
    df = fetch_daily("SPY")
    if df is None:
        return None
    ret = df["close"].pct_change().dropna()
    ret.index = pd.to_datetime(ret.index).tz_localize(None).normalize()
    return ret.groupby(ret.index).last()


def risk_metrics_table(frames: dict) -> pd.DataFrame:
    """
    기관 수준 리스크·성과 지표 테이블 (최근 2년 일봉 기준, 무위험수익률 2% 가정).

    CAGR / 연변동성 / 샤프 / 소르티노 / 칼마 / MDD / VaR·CVaR(95%, 1일) /
    베타(vs S&P500) / 승률
    """
    RF = 0.02
    bench = benchmark_returns()
    rows = []
    for sym, df in frames.items():
        ret = df["close"].pct_change().dropna()
        if len(ret) < 60:
            continue
        ann_ret = float(ret.mean() * 252)
        vol = float(ret.std() * np.sqrt(252))
        cagr = float((1 + ret).prod() ** (252 / len(ret)) - 1)
        sharpe = (ann_ret - RF) / vol if vol > 0 else 0.0

        downside = ret[ret < 0]
        dvol = float(downside.std() * np.sqrt(252)) if len(downside) > 1 else 0.0
        sortino = (ann_ret - RF) / dvol if dvol > 0 else 0.0

        cum = (1 + ret).cumprod()
        mdd = float(((cum - cum.cummax()) / cum.cummax()).min())
        calmar = cagr / abs(mdd) if mdd != 0 else 0.0

        var95 = float(ret.quantile(0.05))
        tail = ret[ret <= var95]
        cvar95 = float(tail.mean()) if len(tail) else var95

        beta = None
        if bench is not None and sym != "SPY":
            r = ret.copy()
            r.index = pd.to_datetime(r.index).tz_localize(None).normalize()
            r = r.groupby(r.index).last()
            joined = pd.concat([r, bench], axis=1, join="inner").dropna()
            if len(joined) > 60 and joined.iloc[:, 1].var() > 0:
                beta = float(joined.iloc[:, 0].cov(joined.iloc[:, 1])
                             / joined.iloc[:, 1].var())
        elif sym == "SPY":
            beta = 1.0

        rows.append({
            "종목": f"{name_of(sym)} ({sym})",
            "CAGR": f"{cagr:+.1%}",
            "연변동성": f"{vol:.1%}",
            "샤프": round(sharpe, 2),
            "소르티노": round(sortino, 2),
            "칼마": round(calmar, 2),
            "MDD": f"{mdd:.1%}",
            "VaR95(1일)": f"{var95:.2%}",
            "CVaR95(1일)": f"{cvar95:.2%}",
            "베타(vs S&P)": round(beta, 2) if beta is not None else "-",
            "승률(일)": f"{float((ret > 0).mean()):.0%}",
        })
    out = pd.DataFrame(rows)
    return out.sort_values("샤프", ascending=False) if len(out) else out


# ─────────────────────────────────────────────────────────────────────────────
# ⭐ AI 추천 — 모델 기반 예상 수익률·상승확률 미리보기
# ─────────────────────────────────────────────────────────────────────────────

def chart_forecast(rows: list[dict]) -> go.Figure:
    syms = [r["sym"] for r in rows]
    exp5 = [r["exp5"] * 100 for r in rows]
    probs = [r["p_up"] * 100 for r in rows]
    bar_colors = [COLORS["up"] if v >= 0 else COLORS["down"] for v in exp5]
    fig = make_subplots(rows=1, cols=2, column_widths=[0.55, 0.45],
                        subplot_titles=["모델 예상 수익률 (5일, %)",
                                        "상승확률(가로) vs 예상수익률(세로)"])
    fig.add_trace(go.Bar(x=[name_of(s) for s in syms], y=exp5, marker_color=bar_colors,
        text=[f"{v:+.2f}%" for v in exp5], textposition="outside",
        showlegend=False), row=1, col=1)
    fig.add_hline(y=0, line_color=COLORS["muted"], opacity=0.5, row=1, col=1)
    fig.add_trace(go.Scatter(x=probs, y=exp5, mode="markers+text",
        text=syms, textposition="top center",
        marker=dict(size=14, color=bar_colors,
                    line=dict(color=COLORS["panel"], width=1)),
        hovertemplate="%{text}<br>상승확률 %{x:.0f}%<br>예상 %{y:+.2f}%<extra></extra>",
        showlegend=False), row=1, col=2)
    fig.add_vline(x=50, line_dash="dot", line_color=COLORS["muted"], opacity=0.4, row=1, col=2)
    fig.add_hline(y=0, line_dash="dot", line_color=COLORS["muted"], opacity=0.4, row=1, col=2)
    fig.update_layout(**BASE_LAYOUT, height=440,
        title=dict(text="모델 기반 예측 미리보기",
                   font=dict(size=16, color=COLORS["accent"])))
    fig.update_xaxes(title_text="상승확률(%)", row=1, col=2, gridcolor=COLORS["grid"])
    fig.update_yaxes(ticksuffix="%", gridcolor=COLORS["grid"])
    fig.update_xaxes(gridcolor=COLORS["grid"])
    return fig


def _render_reco_results(res: dict, frames: dict) -> None:
    """추천 결과 공통 렌더 — 톱픽 카드 + 예측 차트 + 상세 표 (자동/관심종목 공용)."""
    rows = []
    for s, r in res.items():
        p = r["pred"]
        p_up = float(p.up_probability) if p else 0.5
        df = frames.get(s)
        fc = (model_forecast(df, p_up) if df is not None and len(df)
              else {"exp5": 0.0, "exp21": 0.0})
        a = r.get("analyst") or {}
        tgt, cur = a.get("targetMeanPrice"), r["cur"]
        tgt_ok = (tgt is not None and cur and not pd.isna(tgt) and not pd.isna(cur))
        rows.append(dict(
            sym=s, final=r["final"], verdict=verdict(r["final"]),
            p_up=p_up, conf=(float(p.confidence) if p else 0.0),
            regime=(p.regime if p else "-"),
            exp5=fc["exp5"], exp21=fc["exp21"], band=r["band"], cur=cur,
            tgt_up=((tgt - cur) / cur if tgt_ok else None),
        ))
    rows.sort(key=lambda x: (x["final"], x["exp5"]), reverse=True)

    st.subheader("오늘의 톱픽")
    top = rows[:3]
    cols = st.columns(len(top))
    for col, medal, r0 in zip(cols, ["🥇", "🥈", "🥉"], top):
        col.metric(
            f"{medal} {name_of(r0['sym'])} ({r0['sym']}) — {r0['verdict']}",
            f"종합 {r0['final']:.0f}점 · 상승확률 {r0['p_up']:.0%}",
            f"{r0['exp5']:+.2%} (5일 예상수익)")

    st.plotly_chart(chart_forecast(rows), width="stretch")

    st.subheader("종목별 예측 상세")
    tbl = pd.DataFrame([{
        "순위": i + 1,
        "종목": f"{name_of(r['sym'])} ({r['sym']})",
        "추천": r["verdict"],
        "종합점수": round(r["final"], 1),
        "상승확률": f"{r['p_up']:.0%}",
        "예상수익(5일)": f"{r['exp5']:+.2%}",
        "예상수익(1개월·환산)": f"{r['exp21']:+.1%}",
        "1개월 예상범위(±1σ)": f"${r['band'][0]:,.2f}~${r['band'][1]:,.2f}",
        "목표가 업사이드": f"{r['tgt_up']:+.1%}" if r["tgt_up"] is not None else "-",
        "신뢰도": f"{r['conf']:.0%}",
        "레짐": r["regime"],
    } for i, r in enumerate(rows)])
    st.dataframe(tbl, width="stretch", hide_index=True)
    st.caption("⚠️ 교육용 참고 지표 — 투자 조언이 아닙니다. "
               "예상수익은 확률 가중 추정치이며 실제 결과와 다를 수 있습니다.")


def render_ai_reco_page(symbols: list[str]) -> None:
    """AI 추천 — ① 유니버스 자동 추천(AI가 종목 선정) ② 내 관심종목 분석."""
    st.caption("앙상블 ML(상승확률) + 가치 점수 + 펀더멘털 틸트 + 애널리스트 컨센서스로 "
               "**예상 수익률·상승확률**을 산출해 추천합니다.")
    help_box(
        "**🤖 AI 자동 추천** — 종목 선택까지 AI에게 맡깁니다. **섹터를 고르면 그 안에서** "
        "2단계로 종목을 선정합니다:\n"
        "1. **1차 스크리닝**: 섹터 종목을 일괄 스캔해 기술 팩터(추세·모멘텀)로 유망 후보를 추립니다.\n"
        "2. **정밀 분석**: 후보에만 가치투자+ML 앙상블+예측을 돌려 **종합점수·예상수익**으로 랭킹.\n"
        "   *같은 섹터 내 비교라 거시 영향이 통제돼 상대 랭킹이 더 타당합니다.*\n\n"
        "**예상 수익률** = 상승확률 × 과거 상승 시 평균수익 + 하락확률 × 과거 하락 시 평균손실.\n"
        "**종합점수** = 기술 20% + 가치 40% + ML 40% (65↑ 매수 / 50~65 보유 / 그 이하 관망).\n\n"
        "⚠️ 과거 데이터 기반 추정이며 투자 조언이 아닙니다. 후보 정밀분석은 종목당 ~6초(이후 캐시).")

    mode = st.radio("추천 방식", ["🤖 AI 자동 추천 (유니버스 스캔)", "⭐ 내 관심종목 분석"],
                    horizontal=True, key="reco_mode")

    if mode.startswith("🤖"):
        c = st.columns([0.52, 0.3, 0.18])
        sector = c[0].selectbox("섹터 / 시장", list(SECTOR_UNIVERSE) + ["전체 대형주"],
                                key="reco_sector",
                                help="섹터를 고르면 그 안에서 AI가 종목을 선정합니다.")
        topk = c[1].slider("정밀분석 후보 수", 5, 20, 8, key="reco_topk",
                           help="1차 스크리닝 통과 후 ML 정밀분석할 상위 종목 수")
        universe = (SCREENER_UNIVERSE if sector == "전체 대형주"
                    else SECTOR_UNIVERSE[sector])
        st.caption(f"대상 유니버스: **{sector}** ({len(universe)}종목) → 1차 스크리닝 "
                   f"→ 상위 {topk}개 정밀분석")
        if c[2].button("🚀 AI 추천", type="primary"):
            with st.spinner(f"1차 스크리닝 — {len(universe)}종목 일괄 다운로드·팩터 계산..."):
                batch = fetch_daily_batch(tuple(universe))
                ranked = sorted(
                    ((s, screen_one(df)) for s, df in batch.items()),
                    key=lambda kv: kv[1]["tech"] * 0.5 + _lin(kv[1]["ret_3m"], -20, 30) * 0.5,
                    reverse=True)
                finalists = [s for s, _ in ranked[:topk]]
            res = run_advanced(finalists, batch)   # 정밀 분석(가치+ML), 진행바 표시
            st.session_state.ai_reco = dict(
                res=res, frames={s: batch[s] for s in finalists if s in batch})
        bundle = st.session_state.get("ai_reco")
        if not bundle or not bundle.get("res"):
            st.info("위 **🚀 AI 추천**을 눌러 유니버스에서 자동 추천을 생성하세요.")
            return
        _render_reco_results(bundle["res"], bundle["frames"])

    else:   # 내 관심종목 분석
        if not symbols:
            st.info("사이드바에서 종목을 먼저 선택하거나, 위에서 **🤖 AI 자동 추천**을 사용하세요.")
            return
        if st.button("⭐ 분석 실행 / 갱신", type="primary", key="reco_watch_btn"):
            with st.spinner("관심종목 정밀 분석 중..."):
                frames = load_frames(symbols)
                st.session_state.ai_reco = dict(
                    res=run_advanced(symbols, frames), frames=frames)
        bundle = st.session_state.get("ai_reco")
        if not bundle or not bundle.get("res"):
            st.info("위 **⭐ 분석 실행**을 눌러주세요.")
            return
        _render_reco_results(bundle["res"], bundle["frames"])


# ─────────────────────────────────────────────────────────────────────────────
# 🧪 모델 검증 — walk-forward OOF 성적표 (적중률·칼리브레이션·구간수익)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=21600, show_spinner=False)
def validate_one(symbol: str) -> dict | None:
    """
    종목 모델을 학습하고 out-of-sample 검증 지표를 반환 (무거움, 6시간 캐시).

    학습 전에 워크포워드 그리드로 XGBoost 하이퍼파라미터를 최적화하고(성능 우선),
    찾은 최적값으로 모델을 학습한다. 반환 dict 에는 검증 지표 + "tuning" 리포트
    (최적 파라미터·기본값 대비 향상도·리더보드)가 함께 담긴다.
    """
    from app.ml.models.improved_ensemble import ImprovedEnsembleModel
    df = fetch_daily(symbol)
    if df is None or len(df) <= 320:   # walk-forward 검증엔 ~1.5년+ 필요
        return None
    try:
        frame = _Frame(df, symbol)
        model = ImprovedEnsembleModel()
        tuning = model.optimize_hyperparams(frame)   # 최적 XGB 파라미터 탐색·저장
        model.fit(frame)                             # 최적값으로 학습
        metrics = model.validation_metrics()
        if metrics is None:
            return None
        metrics["tuning"] = tuning                   # 없으면 None
        return metrics
    except Exception:
        return None


def chart_calibration(calib: list) -> go.Figure:
    """칼리브레이션 — 예측확률(가로) vs 실제 상승빈도(세로). 대각선=완벽 보정."""
    if not calib:
        return go.Figure()
    px = [c[0] * 100 for c in calib]
    py = [c[1] * 100 for c in calib]
    sizes = [min(40, 8 + c[2] / 3) for c in calib]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 100], y=[0, 100], mode="lines",
        line=dict(color=COLORS["muted"], dash="dash", width=1),
        name="완벽 보정", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=px, y=py, mode="markers+lines",
        marker=dict(size=sizes, color=COLORS["accent"],
                    line=dict(color=COLORS["panel"], width=1)),
        line=dict(color=COLORS["accent"], width=2), name="모델",
        hovertemplate="예측 %{x:.0f}% → 실제 %{y:.0f}%<extra></extra>"))
    fig.update_layout(**BASE_LAYOUT, height=360,
        title=dict(text="칼리브레이션 (예측확률 vs 실제빈도)",
                   font=dict(size=15, color=COLORS["accent"])),
        legend=dict(bgcolor="rgba(0,0,0,0)"))
    fig.update_xaxes(title_text="예측 상승확률(%)", range=[0, 100], gridcolor=COLORS["grid"])
    fig.update_yaxes(title_text="실제 상승빈도(%)", range=[0, 100], gridcolor=COLORS["grid"])
    return fig


def chart_ret_by_bin(ret_by_bin: list) -> go.Figure:
    """확률 구간별 평균 N일 수익률 — 우상향이면 모델에 정보가 있다는 뜻."""
    if not ret_by_bin:
        return go.Figure()
    labels = [r[0] for r in ret_by_bin]
    vals = [r[1] for r in ret_by_bin]
    colors = [COLORS["up"] if v >= 0 else COLORS["down"] for v in vals]
    fig = go.Figure(go.Bar(x=labels, y=vals, marker_color=colors,
        text=[f"{v:+.2f}%" for v in vals], textposition="outside"))
    fig.add_hline(y=0, line_color=COLORS["muted"], opacity=0.5)
    fig.update_layout(**BASE_LAYOUT, height=360,
        title=dict(text="예측확률 구간별 실제 5일 수익률",
                   font=dict(size=15, color=COLORS["accent"])))
    fig.update_xaxes(title_text="모델 예측 상승확률 구간", gridcolor=COLORS["grid"])
    fig.update_yaxes(title_text="평균 5일 수익률(%)", ticksuffix="%", gridcolor=COLORS["grid"])
    return fig


def render_tuning_panel(t: dict | None) -> None:
    """검증 카드 안: XGBoost 최적 하이퍼파라미터를 바로 보이게 표시.

    워크포워드 OOF AUC 로 18개 조합을 채점해 고른 최적값과, 기본(고정)값 대비
    향상도(Δ AUC), 상위 조합 리더보드를 함께 보여준다."""
    if not t:
        st.caption("⚙️ 최적 하이퍼파라미터: 데이터 부족 등으로 탐색을 건너뛰었습니다.")
        return
    bp   = t["best_params"]
    imp  = t.get("improvement", float("nan"))
    with st.expander("⚙️ 최적 하이퍼파라미터 (XGBoost · 워크포워드 AUC 기준)", expanded=True):
        cc = st.columns(4)
        cc[0].metric("max_depth",     bp.get("max_depth", "-"))
        cc[1].metric("learning_rate", f"{bp.get('learning_rate', 0):.3g}")
        cc[2].metric("n_estimators",  bp.get("n_estimators", "-"))
        cc[3].metric("최적 AUC", f"{t['best_auc']:.3f}",
                     f"{imp:+.3f} vs 기본" if imp == imp else None,
                     delta_color="normal" if (imp == imp and imp >= 0) else "inverse")

        lb = pd.DataFrame([{
            "max_depth":     c.get("max_depth"),
            "learning_rate": c.get("learning_rate"),
            "n_estimators":  c.get("n_estimators"),
            "AUC":           f"{a:.3f}",
        } for c, a in t.get("leaderboard", [])])
        if not lb.empty:
            st.markdown("**상위 조합 리더보드**")
            st.dataframe(lb, hide_index=True, width="stretch")

        base     = t.get("base_params", {})
        base_auc = t.get("base_auc", float("nan"))
        cap = (f"평가 {t['n_combos']}개 조합 · OOF 표본 {t['n_samples']}일 · "
               f"기본값(max_depth={base.get('max_depth')}, lr={base.get('learning_rate')}, "
               f"n_estimators={base.get('n_estimators')})")
        if base_auc == base_auc:
            cap += f" AUC {base_auc:.3f} 와 비교"
        st.caption(cap)


_GRADE_DESC = {
    "A": "🟢 우수 — 통계적으로 의미 있는 예측력",
    "B": "🟢 양호 — 약하지만 유효한 신호",
    "C": "🟡 보통 — 신중히 참고",
    "D": "🟠 약함 — 거의 동전 던지기",
    "F": "🔴 무의미 — 예측력 없음 (무작위 이하)",
    "N/A": "⚪ 판정 불가",
}


def render_validation_page(symbols: list[str]) -> None:
    st.caption("⭐ AI 추천·💰 가치ML 에 쓰는 **앙상블 모델이 과거에 실제로 맞았는지**를 "
               "walk-forward out-of-sample(룩어헤드 없음)으로 채점한 **성적표**입니다.")
    help_box(
        "**왜 중요한가** — 모델이 '70% 상승'이라 말할 때, 실제로 70% 정도 올랐는지 검증 "
        "안 하면 그 숫자는 의미가 없습니다. 이 페이지가 그걸 측정합니다.\n\n"
        "- **AUC**: 0.5=동전 던지기, **0.55↑면 의미 있는 예측력**, 0.6↑면 우수. 가장 핵심 지표.\n"
        "- **방향 적중률**: proba>0.5 기준 맞힌 비율. 단, 시장이 원래 상승편향이라 "
        "**기준선(그냥 '항상 상승' 찍기)** 과 비교해야 공정합니다 → **엣지**가 양수여야 진짜 정보.\n"
        "- **Brier**: 확률 예측 오차(낮을수록 좋음, 0.25=무작위).\n"
        "- **칼리브레이션**: 점들이 대각선에 가까울수록 확률이 정직합니다.\n"
        "- **구간별 수익률**: 예측확률이 높은 구간일수록 실제 수익이 높으면(우상향) 정보가 있는 것.\n"
        "- **⚙️ 최적 하이퍼파라미터**: 각 종목마다 XGBoost 설정(트리 깊이·학습률·트리 수) "
        "18개 조합을 워크포워드 AUC 로 채점해 **가장 성능 좋은 조합**을 골라 학습에 씁니다. "
        "기본값 대비 AUC가 얼마나 올랐는지(Δ)도 함께 표시됩니다.\n\n"
        "⚠️ 과거 검증이 미래를 보장하지 않습니다. 무료 데이터·단일종목 학습의 한계가 있습니다.")

    if st.button("🧪 모델 검증 실행 / 갱신", type="primary"):
        res = {}
        prog = st.progress(0.0, text="검증 준비 중...")
        for i, sym in enumerate(symbols):
            prog.progress(i / len(symbols),
                          text=f"{name_of(sym)} 하이퍼파라미터 탐색·학습·검증 중… (종목당 ~20–40초)")
            m = validate_one(sym)
            if m:
                res[sym] = m
            prog.progress((i + 1) / len(symbols))
        prog.empty()
        st.session_state.val_results = res

    res = {s: st.session_state.get("val_results", {}).get(s)
           for s in symbols if st.session_state.get("val_results", {}).get(s)}
    if not res:
        st.info("위 **🧪 모델 검증 실행** 버튼을 눌러주세요. "
                "(하이퍼파라미터 탐색 포함 종목당 ~20–40초, 데이터 1.5년+ 필요. 6시간 캐시)")
        return

    # 종목별 성적표 카드
    for sym, m in sorted(res.items(), key=lambda kv: -(kv[1]["auc"] if kv[1]["auc"] == kv[1]["auc"] else 0)):
        with st.container(border=True):
            st.markdown(f"#### {name_of(sym)} ({sym}) — 성적 **{m['grade']}**  ·  "
                        f"{_GRADE_DESC.get(m['grade'], '')}")
            k = st.columns(5)
            auc = m["auc"]
            k[0].metric("AUC", f"{auc:.3f}" if auc == auc else "-",
                        "예측력 있음" if (auc == auc and auc > 0.55) else "무작위 수준" if auc == auc else None)
            k[1].metric("방향 적중률", f"{m['accuracy']:.1%}")
            k[2].metric("엣지(기준선 대비)", f"{m['edge']:+.1%}p",
                        "정보 있음" if m["edge"] > 0.01 else "미미",
                        delta_color="normal" if m["edge"] > 0 else "inverse")
            k[3].metric("Brier", f"{m['brier']:.3f}" if m["brier"] == m["brier"] else "-",
                        help="낮을수록 좋음(0.25=무작위)")
            k[4].metric("롱 신호 엣지", f"{m['long_edge_pct']:+.2f}%p",
                        help="proba>55% 일 때 평균수익 − 전체 평균")
            c1, c2 = st.columns(2)
            c1.plotly_chart(chart_calibration(m["calibration"]), width="stretch")
            c2.plotly_chart(chart_ret_by_bin(m["ret_by_bin"]), width="stretch")
            render_tuning_panel(m.get("tuning"))
            st.caption(f"검증 표본 {m['n']}일 · 실제 상승비율(기준선) {m['base_rate']:.1%}")

    # 종합 요약 표
    st.subheader("종합 성적표")
    tbl = pd.DataFrame([{
        "종목": f"{name_of(s)} ({s})",
        "성적": m["grade"],
        "AUC": f"{m['auc']:.3f}" if m["auc"] == m["auc"] else "-",
        "적중률": f"{m['accuracy']:.1%}",
        "엣지": f"{m['edge']:+.1%}p",
        "Brier": f"{m['brier']:.3f}" if m["brier"] == m["brier"] else "-",
        "롱엣지": f"{m['long_edge_pct']:+.2f}%p",
        "표본": m["n"],
    } for s, m in res.items()])
    st.dataframe(tbl, width="stretch", hide_index=True)
    st.caption("⚠️ AUC가 0.5 근처면 그 종목에선 모델 신호를 신뢰하지 마세요. "
               "검증은 과거 기준이며 미래 수익을 보장하지 않습니다.")


# ─────────────────────────────────────────────────────────────────────────────
# 전략 백테스트 (알고리즘 트레이딩) — 다음날 체결 · 수수료 반영 · 롱/현금
# ─────────────────────────────────────────────────────────────────────────────

def _rsi_arr(c: np.ndarray, period: int = 14) -> np.ndarray:
    d = np.diff(c, prepend=c[0])
    up = np.where(d > 0, d, 0.0)
    dn = np.where(d < 0, -d, 0.0)
    au = pd.Series(up).ewm(alpha=1 / period, adjust=False).mean().values
    ad = pd.Series(dn).ewm(alpha=1 / period, adjust=False).mean().values
    return 100 - 100 / (1 + au / (ad + 1e-9))


def strategy_positions(df: pd.DataFrame, strategy: str, p: dict) -> np.ndarray:
    """전략별 목표 포지션 (1=보유, 0=현금). 룩어헤드 방지 시프트는 run_backtest 에서."""
    c = df["close"].values
    n = len(c)
    pos = np.zeros(n)

    if strategy == "MA 크로스":
        fast = pd.Series(c).rolling(p["fast"]).mean().values
        slow = pd.Series(c).rolling(p["slow"]).mean().values
        pos = np.where(fast > slow, 1.0, 0.0)

    elif strategy == "RSI 역추세":
        rsi = _rsi_arr(c, p["rsi_period"])
        hold = 0.0
        for i in range(n):                       # 진입/청산이 상태 의존 → 루프
            if hold == 0.0 and rsi[i] < p["buy_th"]:
                hold = 1.0
            elif hold == 1.0 and rsi[i] > p["sell_th"]:
                hold = 0.0
            pos[i] = hold

    elif strategy == "MACD":
        ema12 = pd.Series(c).ewm(span=12).mean().values
        ema26 = pd.Series(c).ewm(span=26).mean().values
        macd = ema12 - ema26
        sig = pd.Series(macd).ewm(span=9).mean().values
        pos = np.where(macd > sig, 1.0, 0.0)

    elif strategy == "볼린저 평균회귀":
        mid = pd.Series(c).rolling(20).mean().values
        std = pd.Series(c).rolling(20).std().values
        lower = mid - 2 * std
        hold = 0.0
        for i in range(n):
            if np.isnan(mid[i]):
                pos[i] = 0.0
                continue
            if hold == 0.0 and c[i] < lower[i]:
                hold = 1.0                        # 하단 이탈 → 매수
            elif hold == 1.0 and c[i] > mid[i]:
                hold = 0.0                        # 중심선 회복 → 청산
            pos[i] = hold

    elif strategy == "모멘텀 (12-1)":
        look, skip = p["look"], p["skip"]
        mom = pd.Series(c).shift(skip).pct_change(look - skip).values
        pos = np.where(np.nan_to_num(mom) > 0, 1.0, 0.0)
        pos[np.isnan(mom)] = 0.0

    return np.nan_to_num(pos)


def run_backtest(df: pd.DataFrame, strategy: str, p: dict,
                 cost_bps: float = 10.0) -> dict:
    """
    벡터화 백테스트. 신호 발생 '다음 날' 수익률부터 반영(룩어헤드 방지),
    포지션 변경 시 편도 비용(cost_bps = 수수료+슬리피지) 차감. 롱/현금 전략.
    """
    c = df["close"].values
    ret = np.diff(c) / c[:-1]                              # 일별 수익률 (n-1,)
    pos = strategy_positions(df, strategy, p)
    pos_lag = np.roll(pos, 1)                              # 어제 신호로 오늘 보유
    pos_lag[0] = 0.0

    trades = np.abs(np.diff(pos_lag))                      # 포지션 변경 시점 (n-1,)
    cost = trades * cost_bps / 10000
    strat_ret = pos_lag[1:] * ret - cost

    eq = np.cumprod(1 + strat_ret)
    bh = np.cumprod(1 + ret)

    def _metrics(r: np.ndarray) -> dict:
        ann = float(r.mean() * 252)
        vol = float(r.std() * np.sqrt(252))
        cum = np.cumprod(1 + r)
        peak = np.maximum.accumulate(cum)
        mdd = float(((cum - peak) / peak).min())
        return dict(
            total=float(cum[-1] - 1),
            cagr=float(cum[-1] ** (252 / len(r)) - 1),
            sharpe=(ann - 0.02) / vol if vol > 0 else 0.0,
            mdd=mdd,
        )

    # 트레이드 단위 승률: 보유 구간(진입→청산)별 전략 자산곡선 변화 부호
    wins = closed = 0
    entry_val = None
    for i in range(1, len(pos_lag)):
        if pos_lag[i] == 1.0 and pos_lag[i - 1] == 0.0:
            entry_val = eq[i - 1] if 0 <= i - 1 < len(eq) else 1.0
        elif pos_lag[i] == 0.0 and pos_lag[i - 1] == 1.0 and entry_val is not None:
            exit_val = eq[min(i - 1, len(eq) - 1)]
            closed += 1
            wins += int(exit_val > entry_val)
            entry_val = None

    return dict(
        dates=df.index[1:], equity=eq, buyhold=bh, position=pos_lag[1:],
        strat=_metrics(strat_ret), bench=_metrics(ret),
        n_trades=int(trades.sum()), exposure=float(pos_lag.mean()),
        win_rate=(wins / closed) if closed else None,
    )


def chart_backtest(bt: dict, sym: str, strategy: str) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25],
                        vertical_spacing=0.05,
                        subplot_titles=["전략 vs 바이앤홀드 누적 수익률 (%)", "포지션 (1=보유)"])
    fig.add_trace(go.Scatter(x=bt["dates"], y=(bt["equity"] - 1) * 100,
        line=dict(color=COLORS["accent"], width=2.2), name=f"전략: {strategy}"), row=1, col=1)
    fig.add_trace(go.Scatter(x=bt["dates"], y=(bt["buyhold"] - 1) * 100,
        line=dict(color=COLORS["muted"], width=1.6, dash="dot"), name="바이앤홀드"), row=1, col=1)
    fig.add_hline(y=0, line_dash="dot", line_color=COLORS["muted"], opacity=0.4, row=1, col=1)
    fig.add_trace(go.Scatter(x=bt["dates"], y=bt["position"],
        line=dict(color=COLORS["up"], width=1), fill="tozeroy",
        fillcolor=hex_to_rgba(COLORS["up"], 0.12), name="포지션",
        showlegend=False), row=2, col=1)
    fig.update_layout(**BASE_LAYOUT, height=620,
        title=dict(text=f"{name_of(sym)} ({sym}) — {strategy} 백테스트 (최근 2년)",
                   font=dict(size=16, color=COLORS["accent"])),
        legend=dict(bgcolor="rgba(255,255,255,0.6)", orientation="h", y=1.02))
    fig.update_yaxes(ticksuffix="%", row=1, col=1, gridcolor=COLORS["grid"])
    fig.update_yaxes(range=[-0.1, 1.1], row=2, col=1, gridcolor=COLORS["grid"])
    fig.update_xaxes(gridcolor=COLORS["grid"])
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 뉴스 · AI 심리 분석 (다단계 LLM 파이프라인 — app/news/)
# ─────────────────────────────────────────────────────────────────────────────

def gemini_api_key() -> str | None:
    """GEMINI_API_KEY(또는 GOOGLE_API_KEY) 탐색: 환경변수 → .streamlit/secrets.toml 순."""
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        key = os.environ.get(name)
        if key:
            return key
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        try:
            if st.secrets.get(name):
                return st.secrets[name]
        except Exception:
            pass
    return None


@st.cache_data(ttl=1800, show_spinner=False)    # 뉴스 목록 30분 캐시
def fetch_news_cached(symbol: str) -> list[dict]:
    from app.news.aggregator import fetch_ticker_news
    try:
        return fetch_ticker_news(symbol, limit=12,
                                 newsapi_key=os.environ.get("NEWS_API_KEY"))
    except Exception:
        return []


@st.cache_data(ttl=21600, show_spinner=False)   # AI 분석 6시간 캐시 (API 호출 절약)
def analyze_news_cached(symbol: str, articles: list[dict],
                        has_key: bool, _api_key: str | None) -> dict:
    """Gemini+FinBERT 파이프라인 실행. _api_key 는 캐시 해시에서 제외(보안).

    has_key 는 캐시 해시에 포함 → 키를 새로 넣으면 기존 무키(FinBERT 단독)
    캐시가 자동 무효화되어 Gemini 이벤트·흐름이 갱신된다."""
    from app.news.llm_pipeline import analyze_ticker_news
    return analyze_ticker_news(symbol, articles, api_key=_api_key or None)


@st.cache_resource(show_spinner=False)
def _finbert_ready() -> bool:
    """FinBERT(transformers+torch) 사용 가능 여부 — 1회 확인 후 캐시."""
    try:
        from app.news.finbert_sentiment import finbert_available
        return finbert_available()
    except Exception:
        return False


def chart_sentiment_gauge(score: float, label: str) -> go.Figure:
    """Sentiment Score(-1~+1) 게이지 — 파랑(약세)→오렌지(중립)→빨강(강세). 한국 관례."""
    color = (COLORS["up"] if score > 0.15
             else COLORS["down"] if score < -0.15 else COLORS["warn"])
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number=dict(valueformat="+.2f", font=dict(size=40, color=color)),
        title=dict(text=f"AI 뉴스 심리 — {label}",
                   font=dict(size=15, color=COLORS["accent"])),
        gauge=dict(
            axis=dict(range=[-1, 1], tickvals=[-1, -0.5, 0, 0.5, 1],
                      tickcolor=COLORS["text"]),
            bar=dict(color=color, thickness=0.55),
            bgcolor=COLORS["panel"],
            borderwidth=1, bordercolor=COLORS["grid"],
            steps=[
                dict(range=[-1.0, -0.3], color=hex_to_rgba(COLORS["down"], 0.25)),
                dict(range=[-0.3, 0.3], color=hex_to_rgba(COLORS["warn"], 0.15)),
                dict(range=[0.3, 1.0], color=hex_to_rgba(COLORS["up"], 0.25)),
            ],
            threshold=dict(line=dict(color="#191f28", width=2),
                           thickness=0.8, value=score),
        ),
    ))
    fig.update_layout(**BASE_LAYOUT, height=300, margin=dict(l=30, r=30, t=60, b=10))
    return fig


# 이벤트 방향 배지 — 한국 관례(호재=빨강 상승, 악재=파랑 하락)
_EVENT_DIR_BADGE = {"호재": "🔴", "악재": "🔵", "중립": "⚪"}


def render_news_detail(analysis: dict) -> None:
    """뉴스 분석 상세 — 흐름(서사) + 구조화 이벤트(방향·분류·중요도) + 관전 포인트."""
    nar = analysis.get("narrative")
    if nar:
        st.markdown("##### 📜 흐름 (스토리)")
        st.markdown(nar)

    events = analysis.get("events") or []
    if events:
        st.markdown("##### 🗓️ 핵심 이벤트")
        for e in events:
            badge = _EVENT_DIR_BADGE.get(e.get("direction"), "⚪")
            imp = int(e.get("importance", 0) or 0)
            stars = "●" * imp + "○" * (5 - imp)
            meta = " · ".join(x for x in (e.get("direction", ""),
                                          e.get("category", ""),
                                          f"중요도 {stars}") if x)
            st.markdown(
                f"{badge} **{e.get('title', '')}**  \n"
                f"<span style='color:#8b95a1;font-size:0.78rem'>{meta}</span>",
                unsafe_allow_html=True)
    elif analysis.get("key_events"):   # 구버전 캐시 하위호환
        st.markdown("##### 🗓️ 핵심 이벤트")
        for ev in analysis["key_events"]:
            st.markdown(f"- {ev}")

    wp = analysis.get("watch_points") or []
    if wp:
        st.markdown("##### 👀 관전 포인트 (다음 트리거)")
        for w in wp:
            st.markdown(f"- {w}")

    if analysis.get("impact_summary"):
        st.markdown(f"> **결론:** {analysis['impact_summary']}")
    if analysis.get("drivers"):
        st.markdown("**점수 동인:** " + " · ".join(
            f"`{d}`" for d in analysis["drivers"]))
    if analysis.get("rationale"):
        st.caption(f"근거: {analysis['rationale']}")


def render_news_page(symbols: list[str]) -> None:
    st.caption("야후 파이낸스 뉴스 수집 → **Gemini 3단계 + FinBERT 앙상블** "
               "(①잡음 필터 → ②핵심 이벤트·흐름 요약 → ③심리점수 −1.0~+1.0)")
    help_box(
        "- **심리 게이지**: 오늘 뉴스의 *한계적(marginal)* 가격 영향을 점수화. "
        "+1에 가까울수록 강세 재료, −1에 가까울수록 악재.\n"
        "- **심리점수 = Gemini + FinBERT 앙상블**: "
        "**Gemini**(요약 기반 맥락·거시 판단, 60%)와 **FinBERT**(ProsusAI/finbert, "
        "금융 특화 BERT로 기사별 로컬 감성, 40%)를 섞어 단일 모델보다 robust 하게. "
        "두 점수를 따로도 표시합니다.\n"
        "- **📜 흐름(스토리)**: 최근 뉴스가 시간 순으로 어떻게 전개됐고 사건들이 "
        "원인→결과로 어떻게 이어지는지 **서사**로 설명합니다.\n"
        "- **🗓️ 핵심 이벤트**: 잡음(클릭베이트 등)을 걸러낸 이벤트를 **🔴 호재 / 🔵 악재 / "
        "⚪ 중립** 방향, 분류(실적·규제·애널리스트 등), 중요도(●)와 함께 표시.\n"
        "- **👀 관전 포인트**: 앞으로 주가를 흔들 다음 트리거.\n"
        "- **동인(Drivers)**: 심리점수를 움직인 구체적 요인.\n"
        "- 분석은 6시간 캐시됩니다. 같은 종목 재실행은 즉시.\n\n"
        "⚠️ AI 요약은 참고용이며 투자 조언이 아닙니다. 원문 링크로 교차 확인하세요.")

    api_key = gemini_api_key()
    fb_ready = _finbert_ready()
    can_analyze = bool(api_key) or fb_ready

    if not api_key:
        if fb_ready:
            st.info("🔑 **GEMINI_API_KEY 미설정** — **FinBERT(로컬)** 로 심리점수만 "
                    "산출합니다(요약·필터 없음). 키를 넣으면 이벤트 요약·잡음 필터가 추가됩니다. "
                    "[무료 키 발급](https://aistudio.google.com/apikey)")
        else:
            st.info("🔑 **GEMINI_API_KEY 미설정 · FinBERT 미설치** — 뉴스 목록만 표시합니다. "
                    "[무료 키 발급](https://aistudio.google.com/apikey) 후 환경변수/"
                    "`.streamlit/secrets.toml`에 `GEMINI_API_KEY` 를 넣거나, "
                    "`pip install transformers` 로 로컬 FinBERT 를 활성화하세요.")
    elif fb_ready:
        st.caption("🧠 FinBERT 로컬 모델 사용 가능 — 심리점수가 Gemini와 앙상블됩니다.")

    if "news_analyses" not in st.session_state:
        st.session_state.news_analyses = {}

    tabs = st.tabs([f"{name_of(s)} ({s})" for s in symbols])
    for tab, sym in zip(tabs, symbols):
        with tab:
            with st.spinner(f"{sym} 뉴스 수집 중..."):
                articles = fetch_news_cached(sym)
            if not articles:
                st.warning("수집된 뉴스가 없습니다. (야후 무료 데이터 한계로 "
                           "일부 종목은 비어 있을 수 있습니다)")
                continue

            # ── AI 분석 실행/표시 ────────────────────────────────────────
            if can_analyze:
                btn_label = (f"🤖 {sym} AI 심리 분석 실행" if api_key
                             else f"🧠 {sym} FinBERT 심리 분석 실행")
                spin_msg = ("Gemini 3단계 + FinBERT 앙상블 실행 중… (~10초)" if api_key
                            else "FinBERT 로컬 감성 분석 중… (첫 실행은 모델 다운로드로 느릴 수 있음)")
                run = st.button(btn_label, key=f"news_btn_{sym}", type="primary")
                if run:
                    with st.spinner(spin_msg):
                        try:
                            st.session_state.news_analyses[sym] = \
                                analyze_news_cached(sym, articles, bool(api_key), api_key)
                        except Exception as e:
                            st.error(f"AI 분석 실패: {str(e)[:200]}")

            analysis = st.session_state.news_analyses.get(sym)
            if analysis:
                g, info = st.columns([0.42, 0.58])
                with g:
                    st.plotly_chart(
                        chart_sentiment_gauge(analysis["score"], analysis["label"]),
                        width="stretch")
                    st.caption(f"확신도 {analysis['confidence']:.0%} · "
                               f"기사 {analysis['articles_total']}건 중 "
                               f"{analysis['articles_kept']}건 유효")
                    # 앙상블 분해 — Gemini / FinBERT 개별 점수
                    gs, fs = analysis.get("gemini_score"), analysis.get("finbert_score")
                    parts = []
                    if gs is not None:
                        parts.append(f"Gemini `{gs:+.2f}`")
                    if fs is not None:
                        parts.append(f"FinBERT `{fs:+.2f}`")
                    method = analysis.get("method", "")
                    if parts:
                        st.caption(f"{' · '.join(parts)} → 종합 `{analysis['score']:+.2f}`"
                                   + (f"  ({method})" if method else ""))
                    elif method and method != "없음":
                        st.caption(method)
                with info:
                    render_news_detail(analysis)
                st.divider()

            # ── 뉴스 피드 ────────────────────────────────────────────────
            st.markdown("##### 뉴스 피드")
            kept = set(analysis["kept_indices"]) if analysis else set()
            for i, a in enumerate(articles):
                badge = "🟢 유효" if i in kept else ("⚪ 잡음" if analysis else "")
                with st.container(border=True):
                    title_md = (f"**[{a['title']}]({a['url']})**" if a.get("url")
                                else f"**{a['title']}**")
                    st.markdown(f"{title_md}  {badge}")
                    meta = " · ".join(x for x in (
                        a.get("source", ""),
                        a.get("published", "")[:16].replace("T", " "),
                    ) if x)
                    if a.get("summary"):
                        st.caption(a["summary"][:220] + ("…" if len(a["summary"]) > 220 else ""))
                    if meta:
                        st.caption(meta)


# ─────────────────────────────────────────────────────────────────────────────
# 💎 크립토 전용 페이지
# ─────────────────────────────────────────────────────────────────────────────

def chart_crypto_returns(frames: dict) -> go.Figure:
    """여러 코인의 정규화 누적수익률 비교 (시작=0%)."""
    fig = go.Figure()
    for i, (sym, df) in enumerate(frames.items()):
        cum = ((1 + df["close"].pct_change().fillna(0)).cumprod() - 1) * 100
        fig.add_trace(go.Scatter(x=df.index, y=cum, name=name_of(sym),
            line=dict(color=PALETTE[i % len(PALETTE)], width=2)))
    fig.add_hline(y=0, line_dash="dot", line_color=COLORS["muted"], opacity=0.4)
    fig.update_layout(**BASE_LAYOUT, height=420,
        title=dict(text="코인별 누적 수익률 (최근 2년, %)",
                   font=dict(size=16, color=COLORS["accent"])),
        legend=dict(bgcolor="rgba(0,0,0,0)"))
    fig.update_yaxes(ticksuffix="%", gridcolor=COLORS["grid"])
    fig.update_xaxes(gridcolor=COLORS["grid"])
    return fig


def render_crypto_page(syms: list[str], show_bb: bool, show_ichimoku: bool,
                       show_macd: bool) -> None:
    st.caption("암호화폐는 **24시간·연중무휴** 거래됩니다 (미국 장 시간과 무관). "
               "데이터: 무료 Yahoo Finance, 약 15분 지연.")
    now_utc = datetime.now(timezone.utc)
    st.success(f"🟢 암호화폐 시장 24시간 거래 중 (UTC {now_utc:%H:%M} · "
               f"한국 {datetime.now(KST):%H:%M})")
    help_box(
        "- 암호화폐 티커는 `BTC-USD` 형식입니다. 기존 캔들·지표(MA·볼린저·일목·RSI·MACD)를 "
        "그대로 사용하되, 주식과 달리 **주말에도 거래**되어 캔들이 연속입니다.\n"
        "- **누적 수익률 비교**: 2년 전을 0%로 맞춰 코인 간 상대 성과를 비교합니다.\n"
        "- 변동성이 주식보다 훨씬 크므로 RSI·볼린저 과매수/과매도가 더 자주 발생합니다.\n\n"
        "⚠️ 교육용 참고 지표이며 투자 조언이 아닙니다. 암호화폐는 고위험 자산입니다.")

    if not syms:
        st.info("👈 사이드바에서 코인을 하나 이상 선택하세요.")
        return

    bust = int(time.time() // 30)
    cols = st.columns(len(syms))
    for col, sym in zip(cols, syms):
        try:
            df, last, prev = fetch_intraday(sym, "일봉 (6개월)", bust)
            if df is None:
                col.markdown(ticker_tile(name_of(sym), sym, None, None, None),
                             unsafe_allow_html=True)
                continue
            change = last - prev
            pct = (change / prev * 100) if prev else 0.0
            col.markdown(ticker_tile(name_of(sym), sym, last, change, pct),
                         unsafe_allow_html=True)
        except Exception:
            col.markdown(ticker_tile(name_of(sym), sym, None, None, None),
                         unsafe_allow_html=True)

    st.divider()
    with st.spinner("일봉 불러오는 중..."):
        frames = load_frames(syms)
    if not frames:
        st.error("데이터를 불러오지 못했습니다. 코인 티커를 확인하세요 (예: BTC-USD).")
        return

    tabs = st.tabs([f"{name_of(s)} ({s})" for s in frames])
    for tab, (sym, df) in zip(tabs, frames.items()):
        tab.plotly_chart(
            chart_candlestick(df, sym, name_of(sym), show_bb=show_bb,
                              show_ichimoku=show_ichimoku, show_macd=show_macd),
            width="stretch")

    if len(frames) >= 2:
        st.divider()
        st.plotly_chart(chart_crypto_returns(frames), width="stretch")


# ─────────────────────────────────────────────────────────────────────────────
# 📉 옵션 전용 페이지 — yfinance 옵션 체인 (변동성 스마일·미결제약정·풋콜)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False, ttl=600)
def fetch_option_expiries(symbol: str) -> list[str]:
    try:
        return list(yf.Ticker(symbol).options or [])
    except Exception:
        return []


@st.cache_data(show_spinner=False, ttl=600)
def fetch_option_chain(symbol: str, expiry: str):
    """(calls, puts, spot) 반환. 실패 시 (None, None, None)."""
    try:
        tk = yf.Ticker(symbol)
        ch = tk.option_chain(expiry)
        calls = ch.calls.copy()
        puts = ch.puts.copy()
        try:
            spot = float(tk.fast_info["last_price"])
        except Exception:
            hist = tk.history(period="1d")
            spot = float(hist["Close"].iloc[-1]) if len(hist) else float("nan")
        return calls, puts, spot
    except Exception:
        return None, None, None


def chart_vol_smile(calls: pd.DataFrame, puts: pd.DataFrame, spot: float) -> go.Figure:
    """변동성 스마일 — 행사가별 내재변동성(IV). 노이즈(IV~0, 원거리 행사가) 제거."""
    lo, hi = spot * 0.6, spot * 1.4
    fig = go.Figure()
    for df, color, label in [(calls, COLORS["up"], "콜(Call)"),
                             (puts, COLORS["down"], "풋(Put)")]:
        d = df[(df["impliedVolatility"] > 0.005) &
               (df["strike"].between(lo, hi))].sort_values("strike")
        fig.add_trace(go.Scatter(x=d["strike"], y=d["impliedVolatility"] * 100,
            mode="lines+markers", name=label,
            line=dict(color=color, width=2), marker=dict(size=5)))
    fig.add_vline(x=spot, line_dash="dash", line_color=COLORS["accent"],
                  annotation_text=f"현재가 ${spot:,.2f}",
                  annotation_font_color=COLORS["accent"])
    fig.update_layout(**BASE_LAYOUT, height=420,
        title=dict(text="변동성 스마일 — 행사가별 내재변동성(IV)",
                   font=dict(size=16, color=COLORS["accent"])),
        legend=dict(bgcolor="rgba(0,0,0,0)"))
    fig.update_xaxes(title_text="행사가($)", gridcolor=COLORS["grid"])
    fig.update_yaxes(title_text="내재변동성(%)", ticksuffix="%", gridcolor=COLORS["grid"])
    return fig


def chart_open_interest(calls: pd.DataFrame, puts: pd.DataFrame,
                        spot: float) -> go.Figure:
    """행사가별 미결제약정(OI) — 콜 vs 풋 막대."""
    lo, hi = spot * 0.6, spot * 1.4
    c = calls[calls["strike"].between(lo, hi)]
    p = puts[puts["strike"].between(lo, hi)]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=c["strike"], y=c["openInterest"], name="콜 OI",
                         marker_color=COLORS["up"], opacity=0.75))
    fig.add_trace(go.Bar(x=p["strike"], y=p["openInterest"], name="풋 OI",
                         marker_color=COLORS["down"], opacity=0.75))
    fig.add_vline(x=spot, line_dash="dash", line_color=COLORS["accent"],
                  annotation_text=f"현재가 ${spot:,.2f}",
                  annotation_font_color=COLORS["accent"])
    fig.update_layout(**BASE_LAYOUT, height=420, barmode="overlay",
        title=dict(text="행사가별 미결제약정(Open Interest)",
                   font=dict(size=16, color=COLORS["accent"])),
        legend=dict(bgcolor="rgba(0,0,0,0)"))
    fig.update_xaxes(title_text="행사가($)", gridcolor=COLORS["grid"])
    fig.update_yaxes(title_text="미결제약정(계약)", gridcolor=COLORS["grid"])
    return fig


def render_options_page(symbol: str) -> None:
    st.caption("미국 주식·ETF **옵션 체인**을 분석합니다 — 변동성 스마일, 행사가별 "
               "미결제약정, 풋/콜 비율, 최대 고통(max pain). 데이터: 무료 Yahoo Finance.")
    help_box(
        "- **변동성 스마일(IV)**: 행사가별 시장이 매긴 미래 변동성. U자(스마일)나 한쪽으로 "
        "기운 스큐(skew)가 흔하며, **IV가 높을수록 옵션이 비싸고 시장이 큰 변동을 예상**한다는 뜻.\n"
        "- **미결제약정(OI)**: 아직 청산되지 않은 계약 수 = 그 행사가의 관심·지지/저항. "
        "콜 OI가 몰린 위쪽 행사가는 저항, 풋 OI가 몰린 아래쪽은 지지로 보기도 합니다.\n"
        "- **풋/콜 비율**: 1보다 크면 풋(하락 베팅) 우위 = 방어적/약세 심리, 1보다 작으면 강세 심리.\n"
        "- **최대 고통(Max Pain)**: 만기 시 옵션 매도자의 총 지급액이 최소가 되는 행사가. "
        "만기일 가격이 이 부근으로 수렴하는 경향이 있다는 *가설*(참고용).\n\n"
        "⚠️ 옵션은 레버리지·시간가치 소멸로 **원금 전액 손실**이 가능한 고위험 상품입니다. "
        "교육용 참고 지표이며 투자 조언이 아닙니다.")

    if not symbol:
        st.info("👈 사이드바에서 기초자산 티커를 입력하세요 (예: AAPL).")
        return

    expiries = fetch_option_expiries(symbol)
    if not expiries:
        st.warning(f"**{symbol}** 의 옵션 데이터가 없습니다. 옵션이 상장된 미국 "
                   f"주식·ETF 티커인지 확인하세요 (예: AAPL, NVDA, SPY). "
                   f"암호화폐·일부 해외종목은 옵션 체인이 제공되지 않습니다.")
        return

    expiry = st.selectbox("만기일 선택", expiries,
                          format_func=lambda d: f"{d}  ({_dte(d)}일 후)")
    with st.spinner("옵션 체인 불러오는 중..."):
        calls, puts, spot = fetch_option_chain(symbol, expiry)
    if calls is None or (calls.empty and puts.empty):
        st.error("옵션 체인을 불러오지 못했습니다. 다른 만기를 선택해 보세요.")
        return

    call_oi = float(calls["openInterest"].fillna(0).sum())
    put_oi = float(puts["openInterest"].fillna(0).sum())
    pcr = (put_oi / call_oi) if call_oi else float("nan")
    mp = max_pain(calls, puts)

    m = st.columns(5)
    m[0].metric("현재가", f"${spot:,.2f}" if spot == spot else "-")
    m[1].metric("만기까지", f"{_dte(expiry)}일")
    m[2].metric("풋/콜 비율(OI)", f"{pcr:.2f}" if pcr == pcr else "-",
                "약세 심리" if (pcr == pcr and pcr > 1) else "강세 심리" if pcr == pcr else None,
                delta_color="inverse")
    m[3].metric("최대 고통(Max Pain)", f"${mp:,.2f}" if mp else "-",
                f"{(mp - spot) / spot * 100:+.1f}%" if (mp and spot == spot) else None)
    m[4].metric("총 미결제약정", f"{int(call_oi + put_oi):,}",
                f"콜 {int(call_oi):,} / 풋 {int(put_oi):,}")

    c1, c2 = st.columns(2)
    c1.plotly_chart(chart_vol_smile(calls, puts, spot), width="stretch")
    c2.plotly_chart(chart_open_interest(calls, puts, spot), width="stretch")

    st.subheader("옵션 체인 (현재가 ±40% 행사가)")
    lo, hi = spot * 0.6, spot * 1.4
    show_cols = {"strike": "행사가", "lastPrice": "현재가", "bid": "매수호가",
                 "ask": "매도호가", "impliedVolatility": "IV", "openInterest": "미결제",
                 "volume": "거래량"}
    tc, tp = st.tabs([f"콜 (Call) {len(calls)}건", f"풋 (Put) {len(puts)}건"])
    for tab, df in [(tc, calls), (tp, puts)]:
        d = df[df["strike"].between(lo, hi)][list(show_cols)].rename(columns=show_cols)
        d["IV"] = (d["IV"] * 100).round(1)
        tab.dataframe(d, width="stretch", hide_index=True)


def _dte(expiry: str) -> int:
    """만기까지 남은 일수."""
    try:
        exp = datetime.strptime(expiry, "%Y-%m-%d").date()
        return max(0, (exp - datetime.now(ET).date()).days)
    except ValueError:
        return 0


# ─────────────────────────────────────────────────────────────────────────────
# 🔎 종목 스크리너 — 유니버스 스캔 후 조건 필터·랭킹 (빠른 기술적 팩터)
# ─────────────────────────────────────────────────────────────────────────────

# 스캔 유니버스 — 섹터별 대형 유동주 (빠른 일봉 스캔용). 직접 입력으로 확장 가능.
SCREENER_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "AVGO", "ORCL", "AMD",
    "CRM", "ADBE", "NFLX", "INTC", "QCOM", "CSCO", "TXN", "IBM", "NOW", "UBER",
    "JPM", "BAC", "WFC", "GS", "V", "MA", "AXP",
    "UNH", "JNJ", "LLY", "PFE", "MRK", "ABBV",
    "WMT", "COST", "HD", "MCD", "NKE", "SBUX", "PG", "KO", "PEP",
    "XOM", "CVX", "CAT", "BA", "GE", "DIS", "T", "VZ",
]


def screen_one(df: pd.DataFrame) -> dict:
    """일봉만으로 계산하는 빠른 스크리닝 팩터 (네트워크 추가 호출 없음)."""
    c = df["close"]
    cur = float(c.iloc[-1])
    n = len(c)
    ts, rsi = tech_score(df)
    ma20 = float(c.iloc[-20:].mean()) if n >= 20 else cur
    ma50 = float(c.iloc[-50:].mean()) if n >= 50 else cur
    ret_1m = (cur / float(c.iloc[-22]) - 1) * 100 if n > 22 else 0.0
    ret_3m = (cur / float(c.iloc[-64]) - 1) * 100 if n > 64 else 0.0
    ret_6m = (cur / float(c.iloc[-126]) - 1) * 100 if n > 126 else 0.0
    wk52_high = float(c.iloc[-252:].max()) if n >= 60 else float(c.max())
    from_high = (cur / wk52_high - 1) * 100
    ann_vol = float(c.pct_change().dropna().std() * np.sqrt(252) * 100)
    return dict(cur=cur, rsi=rsi, tech=ts, ret_1m=ret_1m, ret_3m=ret_3m,
                ret_6m=ret_6m, from_high=from_high, ann_vol=ann_vol,
                above_ma50=cur > ma50, above_ma20=cur > ma20)


def render_screener_page(universe: list[str], rsi_lo: float, rsi_hi: float,
                         min_ret3m: float, only_above_ma50: bool,
                         sort_key: str) -> None:
    st.caption("유니버스를 일괄 스캔해 **조건에 맞는 종목을 발굴**합니다 (빠른 기술적 팩터 — "
               "RSI·이동평균·모멘텀·52주 고점 대비). 무거운 ML은 발굴 후 💰 페이지에서.")
    help_box(
        "- **스캔 방식**: 일봉만으로 계산해 빠릅니다(종목당 수백 ms, 캐시되면 즉시).\n"
        "- **RSI 범위**: 예) 30~45 로 좁히면 과매도 반등 후보, 55~70 은 강세 추세 종목.\n"
        "- **3개월 모멘텀**: 추세 강도. 양수만 보려면 0 이상으로.\n"
        "- **MA50 위**: 중기 상승 추세인 종목만.\n"
        "- **기술점수**: RSI+이동평균 배열 종합(0~100). 50 중립, 높을수록 매수 우위.\n\n"
        "⚠️ 발굴은 아이디어 출발점일 뿐, 매매 신호가 아닙니다. 가치·뉴스로 교차 확인하세요.")

    if st.button("🔎 스캔 실행", type="primary"):
        with st.spinner(f"{len(universe)}개 종목 일괄 스캔 중... (배치 다운로드)"):
            frames = fetch_daily_batch(tuple(universe))
            rows = []
            for sym, df in frames.items():
                try:
                    r = screen_one(df)
                    r["sym"] = sym
                    rows.append(r)
                except Exception:
                    continue
        st.session_state.screen_rows = rows
        miss = [s for s in universe if s not in frames]
        if miss:
            st.caption(f"데이터 미수신 {len(miss)}개: "
                       f"{', '.join(miss[:12])}{'…' if len(miss) > 12 else ''} "
                       f"(잠시 후 다시 스캔하면 채워질 수 있어요)")

    rows = st.session_state.get("screen_rows", [])
    if not rows:
        st.info("위 **🔎 스캔 실행** 버튼을 눌러 유니버스를 스캔하세요. "
                f"(현재 유니버스 {len(universe)}종목, 첫 스캔은 ~십수 초)")
        return

    # 필터 적용
    filt = [r for r in rows
            if rsi_lo <= r["rsi"] <= rsi_hi
            and r["ret_3m"] >= min_ret3m
            and (r["above_ma50"] if only_above_ma50 else True)]
    if not filt:
        st.warning("조건에 맞는 종목이 없습니다. 사이드바에서 조건을 완화하세요.")
        return

    key_map = {"기술점수": "tech", "RSI": "rsi", "3개월 수익률": "ret_3m",
               "1개월 수익률": "ret_1m", "52주 고점대비": "from_high",
               "변동성(낮은순)": "ann_vol"}
    sk = key_map.get(sort_key, "tech")
    reverse = sk != "ann_vol"   # 변동성만 오름차순(낮을수록 위)
    filt.sort(key=lambda r: r[sk], reverse=reverse)

    st.success(f"조건 통과: **{len(filt)}종목** / 스캔 {len(rows)}종목")

    # 산점도: 3개월 모멘텀(가로) vs RSI(세로), 색=기술점수
    fig = go.Figure(go.Scatter(
        x=[r["ret_3m"] for r in filt], y=[r["rsi"] for r in filt],
        mode="markers+text", text=[r["sym"] for r in filt], textposition="top center",
        marker=dict(size=14, color=[r["tech"] for r in filt], colorscale="RdYlGn",
                    cmin=0, cmax=100, line=dict(color=COLORS["panel"], width=1),
                    colorbar=dict(title="기술<br>점수", thickness=14)),
        hovertemplate="%{text}<br>3M:%{x:.1f}%<br>RSI:%{y:.0f}<extra></extra>"))
    fig.add_hline(y=70, line_dash="dash", line_color=COLORS["up"], opacity=0.4)
    fig.add_hline(y=30, line_dash="dash", line_color=COLORS["down"], opacity=0.4)
    fig.update_layout(**BASE_LAYOUT, height=460,
        title=dict(text="스크리너 — 3개월 모멘텀 vs RSI",
                   font=dict(size=16, color=COLORS["accent"])))
    fig.update_xaxes(title_text="3개월 수익률(%)", ticksuffix="%", gridcolor=COLORS["grid"])
    fig.update_yaxes(title_text="RSI(14)", gridcolor=COLORS["grid"])
    st.plotly_chart(fig, width="stretch")

    tbl = pd.DataFrame([{
        "종목": f"{name_of(r['sym'])} ({r['sym']})",
        "현재가": f"${r['cur']:,.2f}",
        "기술점수": round(r["tech"], 0),
        "RSI": round(r["rsi"], 0),
        "1개월": f"{r['ret_1m']:+.1f}%",
        "3개월": f"{r['ret_3m']:+.1f}%",
        "6개월": f"{r['ret_6m']:+.1f}%",
        "52주고점대비": f"{r['from_high']:+.1f}%",
        "연변동성": f"{r['ann_vol']:.1f}%",
        "MA50위": "✅" if r["above_ma50"] else "—",
    } for r in filt])
    st.dataframe(tbl, width="stretch", hide_index=True)
    st.caption("⚠️ 교육용 참고 지표 — 투자 조언이 아닙니다.")


# ─────────────────────────────────────────────────────────────────────────────
# 💼 모의 포트폴리오 (페이퍼 트레이딩) — 로컬 영속 저장
# ─────────────────────────────────────────────────────────────────────────────

_PORTFOLIO_FILE = "portfolio.json"


def _load_positions() -> list[dict]:
    from app.storage import load_json
    data = load_json(_PORTFOLIO_FILE, default=[])
    return data if isinstance(data, list) else []


def _save_positions(positions: list[dict]) -> bool:
    from app.storage import save_json
    return save_json(_PORTFOLIO_FILE, positions)


@st.cache_data(show_spinner=False, ttl=300)
def _last_close(symbol: str) -> float | None:
    df = fetch_daily(symbol)
    if df is None or df.empty:
        return None
    return float(df["close"].iloc[-1])


@st.cache_data(show_spinner=False, ttl=300)
def _benchmark_return(start_date: str, symbol: str = "SPY") -> float | None:
    """start_date(YYYY-MM-DD) 이후 벤치마크 총수익률(%)."""
    df = fetch_daily(symbol)
    if df is None or df.empty:
        return None
    idx = pd.to_datetime(df.index).tz_localize(None)
    try:
        start = pd.Timestamp(start_date)
    except ValueError:
        return None
    sub = df[idx >= start]
    if len(sub) < 2:
        return None
    return (float(sub["close"].iloc[-1]) / float(sub["close"].iloc[0]) - 1) * 100


def chart_alloc(rows: list[dict]) -> go.Figure:
    fig = go.Figure(go.Pie(
        labels=[f"{name_of(r['ticker'])}" for r in rows],
        values=[r["value"] for r in rows], hole=0.55,
        marker=dict(colors=[PALETTE[i % len(PALETTE)] for i in range(len(rows))]),
        textinfo="label+percent"))
    fig.update_layout(**BASE_LAYOUT, height=360,
        title=dict(text="보유 비중", font=dict(size=15, color=COLORS["accent"])),
        showlegend=False)
    return fig


def chart_pnl(rows: list[dict]) -> go.Figure:
    syms = [name_of(r["ticker"]) for r in rows]
    pnl = [r["pnl_pct"] for r in rows]
    colors = [COLORS["up"] if v >= 0 else COLORS["down"] for v in pnl]
    fig = go.Figure(go.Bar(x=syms, y=pnl, marker_color=colors,
        text=[f"{v:+.1f}%" for v in pnl], textposition="outside"))
    fig.add_hline(y=0, line_color=COLORS["muted"], opacity=0.5)
    fig.update_layout(**BASE_LAYOUT, height=360,
        title=dict(text="종목별 손익률", font=dict(size=15, color=COLORS["accent"])))
    fig.update_yaxes(ticksuffix="%", gridcolor=COLORS["grid"])
    return fig


def render_paper_page() -> None:
    st.caption("실제 주문 없이 **'이 가격에 샀다'를 기록**하고 손익·벤치마크 대비 성과를 "
               "추적합니다. 데이터는 로컬 `.data/portfolio.json` 에 저장됩니다(깃 제외).")
    help_box(
        "- **페이퍼 트레이딩**: 가상 매매 기록. 실제 체결이 아닙니다.\n"
        "- **미실현 손익**: (현재가 − 매수가) × 수량. 현재가는 무료 일봉 종가 기준(약 15분 지연).\n"
        "- **벤치마크 대비**: 같은 기간 SPY(S&P500)를 들고 있었을 때와 비교. "
        "양수면 시장을 이긴 것(알파).\n"
        "- ⚠️ 로컬 파일 저장이라 **Streamlit Cloud 등에서는 재시작 시 초기화**될 수 있습니다.")

    positions = _load_positions()

    # ── 매수 기록 추가 폼 ────────────────────────────────────────────────────
    with st.expander("➕ 보유 종목 추가", expanded=not positions):
        f = st.columns([1.2, 1, 1, 1.2])
        t_in = f[0].text_input("티커", placeholder="예: AAPL").strip().upper()
        sh_in = f[1].number_input("수량", min_value=0.0, value=10.0, step=1.0)
        default_price = _last_close(t_in) if t_in else None
        pr_in = f[2].number_input("매수 단가($)", min_value=0.0,
                                  value=float(default_price or 0.0), step=1.0,
                                  help="비우면 현재가가 기본값")
        d_in = f[3].date_input("매수일", value=datetime.now(ET).date())
        if st.button("추가", type="primary"):
            if t_in and sh_in > 0 and pr_in > 0:
                positions.append(dict(ticker=t_in, shares=float(sh_in),
                                      price=float(pr_in), date=str(d_in)))
                if _save_positions(positions):
                    st.success(f"{t_in} {sh_in:g}주 @ ${pr_in:,.2f} 추가됨")
                    st.rerun()
                else:
                    st.error("저장 실패 (.data/ 쓰기 권한 확인)")
            else:
                st.warning("티커·수량·단가를 올바르게 입력하세요.")

    if not positions:
        st.info("아직 보유 종목이 없습니다. 위 **➕ 보유 종목 추가**로 기록을 시작하세요.")
        return

    # ── 현재가·손익 계산 ─────────────────────────────────────────────────────
    rows, total_cost, total_val = [], 0.0, 0.0
    earliest = min(p["date"] for p in positions)
    for p in positions:
        cur = _last_close(p["ticker"])
        cost = p["shares"] * p["price"]
        if cur is None:
            rows.append(dict(**p, cur=None, value=0.0, cost=cost,
                             pnl=0.0, pnl_pct=0.0, unavailable=True))
            total_cost += cost
            continue
        value = p["shares"] * cur
        pnl = value - cost
        rows.append(dict(**p, cur=cur, value=value, cost=cost, pnl=pnl,
                         pnl_pct=(pnl / cost * 100) if cost else 0.0,
                         unavailable=False))
        total_cost += cost
        total_val += value

    total_pnl = total_val - total_cost
    total_pct = (total_pnl / total_cost * 100) if total_cost else 0.0
    bench = _benchmark_return(earliest, "SPY")
    alpha = (total_pct - bench) if bench is not None else None

    # ── 요약 메트릭 ──────────────────────────────────────────────────────────
    m = st.columns(4)
    m[0].metric("총 평가금액", f"${total_val:,.2f}")
    m[1].metric("총 손익", f"${total_pnl:,.2f}", f"{total_pct:+.2f}%")
    m[2].metric("벤치마크(SPY)", f"{bench:+.2f}%" if bench is not None else "-",
                help=f"{earliest} 이후")
    m[3].metric("알파(초과수익)", f"{alpha:+.2f}%p" if alpha is not None else "-",
                "시장 상회" if (alpha is not None and alpha > 0) else "시장 하회" if alpha is not None else None)

    valid = [r for r in rows if not r["unavailable"] and r["value"] > 0]
    if valid:
        c1, c2 = st.columns(2)
        c1.plotly_chart(chart_alloc(valid), width="stretch")
        c2.plotly_chart(chart_pnl(valid), width="stretch")

    # ── 보유 종목 표 ─────────────────────────────────────────────────────────
    st.subheader("보유 종목")
    tbl = pd.DataFrame([{
        "종목": f"{name_of(r['ticker'])} ({r['ticker']})",
        "수량": f"{r['shares']:g}",
        "매수가": f"${r['price']:,.2f}",
        "현재가": f"${r['cur']:,.2f}" if r["cur"] else "데이터없음",
        "평가금액": f"${r['value']:,.2f}",
        "손익": f"${r['pnl']:,.2f}" if not r["unavailable"] else "-",
        "손익률": f"{r['pnl_pct']:+.1f}%" if not r["unavailable"] else "-",
        "비중": f"{(r['value']/total_val*100):.1f}%" if (total_val and not r["unavailable"]) else "-",
        "매수일": r["date"],
    } for r in rows])
    st.dataframe(tbl, width="stretch", hide_index=True)

    # ── 삭제 ─────────────────────────────────────────────────────────────────
    with st.expander("🗑️ 종목 삭제"):
        labels = [f"{i}: {p['ticker']} {p['shares']:g}주 @ ${p['price']:,.2f} ({p['date']})"
                  for i, p in enumerate(positions)]
        sel = st.selectbox("삭제할 기록", labels) if labels else None
        if sel and st.button("삭제", type="secondary"):
            idx = int(sel.split(":")[0])
            removed = positions.pop(idx)
            _save_positions(positions)
            st.success(f"{removed['ticker']} 기록 삭제됨")
            st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# 🌐 거시·경제 (FRED) — 키 불필요(공개 CSV), 금리·물가·변동성
# ─────────────────────────────────────────────────────────────────────────────

# FRED 시리즈 ID → 표시명
_FRED_SERIES = {
    "T10Y2Y":   "장단기 금리차 (10년−2년, %)",
    "DGS10":    "미국채 10년 금리 (%)",
    "DGS2":     "미국채 2년 금리 (%)",
    "DFF":      "연방기금금리 (%)",
    "VIXCLS":   "VIX 변동성지수",
    "CPIAUCSL": "소비자물가지수 (CPI)",
    "UNRATE":   "실업률 (%)",
    # 시장·리스크 지표
    "DTWEXBGS":         "달러지수 (무역가중, 브로드)",
    "DCOILWTICO":       "WTI 원유 ($/배럴)",
    "GOLDPMGBD228NLBM": "금 ($/온스, LBMA)",
    "BAMLH0A0HYM2":     "하이일드 신용 스프레드 (%p)",
    "T10YIE":           "10년 기대인플레 (BEI, %)",
}


def _fred_raw(series_id: str) -> pd.Series:
    """FRED 공개 CSV 단일 시리즈 조회 (키 불필요·재시도). 실패 시 예외. 스레드 안전."""
    import io
    import time as _t
    import requests

    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    last_err: Exception | None = None
    for attempt in range(3):
        try:
            resp = requests.get(url, timeout=10,
                                headers={"User-Agent": "Mozilla/5.0 (quant-dashboard)"})
            resp.raise_for_status()
            df = pd.read_csv(io.StringIO(resp.text))
            if df.shape[1] < 2:
                raise ValueError("예상치 못한 CSV 형식")
            date_col, val_col = df.columns[0], df.columns[1]
            df[val_col] = pd.to_numeric(df[val_col], errors="coerce")   # "." → NaN
            s = pd.Series(df[val_col].values,
                          index=pd.to_datetime(df[date_col], errors="coerce")).dropna()
            if len(s) == 0:
                raise ValueError("빈 시리즈")
            return s
        except Exception as e:
            last_err = e
            _t.sleep(0.4 * (attempt + 1))
    raise RuntimeError(f"FRED 조회 실패: {series_id}") from last_err


def _fred_raw_safe(series_id: str) -> pd.Series | None:
    try:
        return _fred_raw(series_id)
    except Exception:
        return None


def _get_fred_bundle(ids: tuple[str, ...], ttl: int = 3600,
                     force: bool = False) -> dict[str, pd.Series | None]:
    """
    여러 FRED 시리즈를 '병렬'로 조회 (순차 7회 → 동시 1회 수준으로 단축).

    - 성공분만 st.session_state 에 (시각, 시리즈)로 캐시 → 재실행 시 즉시.
    - 실패는 캐시하지 않음 → 다음 호출/🔄 버튼에서 자동 재시도.
    - @st.cache_data 를 스레드에서 호출할 때의 컨텍스트 경고를 피하려 세션 캐시 사용.
    """
    import time as _t
    from concurrent.futures import ThreadPoolExecutor

    store: dict = st.session_state.setdefault("_fred_cache", {})
    now = _t.time()
    need = [s for s in ids
            if force or s not in store or (now - store[s][0]) > ttl]
    if need:
        with ThreadPoolExecutor(max_workers=min(8, len(need))) as ex:
            for sid, series in zip(need, ex.map(_fred_raw_safe, need)):
                if series is not None:
                    store[sid] = (now, series)        # 성공만 저장
    return {sid: (store[sid][1] if sid in store else None) for sid in ids}


def _fred_line(s: pd.Series, title: str, years: int = 5,
               zero_line: bool = False, suffix: str = "") -> go.Figure:
    s = s[s.index >= (pd.Timestamp.now() - pd.DateOffset(years=years))]
    color = COLORS["accent"]
    fig = go.Figure(go.Scatter(x=s.index, y=s.values, mode="lines",
        line=dict(color=color, width=2), fill="tozeroy",
        fillcolor=hex_to_rgba(color, 0.08)))
    if zero_line:
        fig.add_hline(y=0, line_dash="dash", line_color=COLORS["down"], opacity=0.6)
    fig.update_layout(**BASE_LAYOUT, height=300,
        title=dict(text=title, font=dict(size=14, color=COLORS["accent"])),
        margin=dict(l=10, r=10, t=40, b=10))
    fig.update_yaxes(ticksuffix=suffix, gridcolor=COLORS["grid"])
    fig.update_xaxes(gridcolor=COLORS["grid"])
    return fig


def render_macro_page() -> None:
    st.caption("미국 거시·시장 지표 — 금리·물가·변동성 + **달러·원유·금·신용 스프레드**. "
               "데이터: **FRED**(세인트루이스 연준, 공개 CSV·키 불필요). "
               "시장 전반의 위험·국면을 읽는 배경 지표입니다.")
    help_box(
        "**거시 지표**\n"
        "- **장단기 금리차(10년−2년)**: 0 아래(역전)면 경기침체 선행 신호로 자주 인용됩니다.\n"
        "- **VIX**: '공포지수'. 보통 20 이하 안정, 30↑ 공포·급락 국면.\n"
        "- **CPI(소비자물가)**: 전년동월비(YoY)로 인플레이션 추세를 봅니다. 연준 목표는 ~2%.\n"
        "- **연방기금금리/국채금리**: 금리가 오르면 성장주·고밸류 자산에 역풍.\n"
        "- **실업률**: 낮을수록 경기 호조이나, 급반등은 경기 둔화 신호.\n\n"
        "**💵 시장·리스크 지표**\n"
        "- **달러지수**: 강달러는 미국 다국적기업 수출·실적과 신흥국에 부담, 약세는 반대로 우호적.\n"
        "- **🛢️ 유가(WTI) / 🥇 금**: 유가는 인플레·에너지주에 직결, 금은 안전자산·인플레 헤지 심리를 반영.\n"
        "- **🚨 하이일드 스프레드**: 고수익채−국채 금리차. **벌어지면** 신용위험·위험회피(risk-off) "
        "신호로 증시에 악재, 좁아지면 위험선호(risk-on).\n"
        "- **기대인플레(BEI)**: 채권시장이 보는 향후 10년 평균 물가. 연준 정책·실질금리의 가늠자.\n\n"
        "⚠️ 거시 지표는 후행·발표지연이 있으며 투자 조언이 아닙니다.")

    force = st.button("🔄 다시 불러오기", help="일부 지표가 비어 있으면 눌러 재시도하세요.")

    with st.spinner("FRED 지표 병렬 로딩 중..."):
        series = _get_fred_bundle(
            ("T10Y2Y", "DGS10", "DGS2", "DFF", "VIXCLS", "CPIAUCSL", "UNRATE",
             "DTWEXBGS", "DCOILWTICO", "GOLDPMGBD228NLBM", "BAMLH0A0HYM2", "T10YIE"),
            force=force)
    spread, dgs10, dgs2, dff, vix, cpi, unrate = (
        series["T10Y2Y"], series["DGS10"], series["DGS2"], series["DFF"],
        series["VIXCLS"], series["CPIAUCSL"], series["UNRATE"])
    usd, oil, gold, hy, bei = (
        series["DTWEXBGS"], series["DCOILWTICO"], series["GOLDPMGBD228NLBM"],
        series["BAMLH0A0HYM2"], series["T10YIE"])

    failed = [sid for sid, s in series.items() if s is None]
    if len(failed) == len(series):
        st.error("FRED 데이터를 불러오지 못했습니다. 네트워크 확인 후 **🔄 다시 불러오기**를 눌러주세요.")
        return
    if failed:
        st.warning(f"일부 지표를 불러오지 못했습니다({', '.join(failed)}). "
                   f"네트워크 일시 지연일 수 있어요 — **🔄 다시 불러오기**로 재시도하면 채워집니다.")

    def _last(s):
        return float(s.iloc[-1]) if s is not None and len(s) else None

    cpi_yoy = None
    if cpi is not None and len(cpi) > 13:
        cpi_yoy = (cpi.iloc[-1] / cpi.iloc[-13] - 1) * 100

    # ── 요약 메트릭 ──────────────────────────────────────────────────────────
    m = st.columns(4)
    sp = _last(spread)
    m[0].metric("장단기 금리차(10Y−2Y)", f"{sp:+.2f}%p" if sp is not None else "-",
                "⚠️ 역전(침체신호)" if (sp is not None and sp < 0) else "정상" if sp is not None else None,
                delta_color="inverse")
    vx = _last(vix)
    m[1].metric("VIX(공포지수)", f"{vx:.1f}" if vx is not None else "-",
                "공포" if (vx is not None and vx > 30) else "안정" if vx is not None else None,
                delta_color="inverse")
    m[2].metric("CPI 전년비(YoY)", f"{cpi_yoy:+.1f}%" if cpi_yoy is not None else "-",
                help="연준 목표 ~2%")
    fed = _last(dff)
    m[3].metric("연방기금금리", f"{fed:.2f}%" if fed is not None else "-")

    m2 = st.columns(3)
    t10 = _last(dgs10); t2 = _last(dgs2); ur = _last(unrate)
    m2[0].metric("미국채 10년", f"{t10:.2f}%" if t10 is not None else "-")
    m2[1].metric("미국채 2년", f"{t2:.2f}%" if t2 is not None else "-")
    m2[2].metric("실업률", f"{ur:.1f}%" if ur is not None else "-")

    st.divider()
    st.subheader("💵 시장·리스크 지표")
    m3 = st.columns(5)
    ud = _last(usd)
    m3[0].metric("달러지수(브로드)", f"{ud:.1f}" if ud is not None else "-",
                 help="무역가중 미 달러 가치. 강달러는 미 다국적기업 실적·신흥국에 부담.")
    ol = _last(oil)
    m3[1].metric("WTI 원유", f"${ol:.1f}" if ol is not None else "-",
                 help="유가는 인플레이션·에너지주에 직결.")
    gd = _last(gold)
    m3[2].metric("금(온스)", f"${gd:,.0f}" if gd is not None else "-",
                 help="안전자산·인플레이션 헤지 수요의 척도.")
    hys = _last(hy)
    m3[3].metric("하이일드 스프레드", f"{hys:.2f}%p" if hys is not None else "-",
                 "⚠️ 신용 경계" if (hys is not None and hys > 5)
                 else "안정" if hys is not None else None,
                 delta_color="inverse",
                 help="고수익채−국채 금리차. 벌어지면 위험회피(risk-off) 신호.")
    bv = _last(bei)
    m3[4].metric("10년 기대인플레", f"{bv:.2f}%" if bv is not None else "-",
                 help="채권시장이 보는 향후 10년 평균 물가상승률(BEI).")

    st.divider()
    # ── 거시 차트 ────────────────────────────────────────────────────────────
    if spread is not None:
        st.plotly_chart(_fred_line(spread, "📉 장단기 금리차 (10년−2년) — 0 아래면 역전",
                                   zero_line=True, suffix="%p"), width="stretch")
    c1, c2 = st.columns(2)
    if vix is not None:
        c1.plotly_chart(_fred_line(vix, "😱 VIX 변동성지수", years=3), width="stretch")
    if cpi is not None:
        cpi_yoy_series = (cpi / cpi.shift(12) - 1) * 100
        c2.plotly_chart(_fred_line(cpi_yoy_series.dropna(), "🔥 CPI 전년비(YoY, %)",
                                   suffix="%"), width="stretch")
    c3, c4 = st.columns(2)
    if dgs10 is not None:
        c3.plotly_chart(_fred_line(dgs10, "🏦 미국채 10년 금리", suffix="%"), width="stretch")
    if unrate is not None:
        c4.plotly_chart(_fred_line(unrate, "👷 실업률", years=7, suffix="%"), width="stretch")

    # ── 시장·리스크 차트 ─────────────────────────────────────────────────────
    if any(s is not None for s in (usd, oil, gold, hy)):
        st.divider()
        st.markdown("##### 💵 시장·리스크 지표 추이")
        c5, c6 = st.columns(2)
        if usd is not None:
            c5.plotly_chart(_fred_line(usd, "💵 달러지수 (무역가중, 브로드)", years=3),
                            width="stretch")
        if hy is not None:
            c6.plotly_chart(_fred_line(hy, "🚨 하이일드 신용 스프레드 (%p) — 벌어지면 위험회피",
                                       years=3, suffix="%p"), width="stretch")
        c7, c8 = st.columns(2)
        if oil is not None:
            c7.plotly_chart(_fred_line(oil, "🛢️ WTI 원유 ($/배럴)", years=3),
                            width="stretch")
        if gold is not None:
            c8.plotly_chart(_fred_line(gold, "🥇 금 ($/온스)", years=3), width="stretch")

    st.caption("📚 출처: Federal Reserve Economic Data (FRED), St. Louis Fed.")


# ─────────────────────────────────────────────────────────────────────────────
# 💵 시장 지표 띠 — 헤더 아래 '항상' 보이는 달러·VIX·금리·유가·금·신용 (FRED, 캐시)
# ─────────────────────────────────────────────────────────────────────────────

# (yfinance 티커, 라벨, 값 포맷) — 종목차트와 같은 소스라 안정적으로 받아진다.
_RIBBON_TICKERS = [
    ("DX-Y.NYB", "💵 달러",      "{:.2f}"),
    ("^VIX",     "😱 VIX",       "{:.1f}"),
    ("^TNX",     "🏦 미국채10Y",  "{:.2f}%"),
    ("CL=F",     "🛢️ WTI",      "${:.1f}"),
    ("GC=F",     "🥇 금",        "${:,.0f}"),
    ("BTC-USD",  "₿ 비트코인",    "${:,.0f}"),
]


@st.cache_data(ttl=300, show_spinner=False)   # 5분 캐시 (재실행 시 즉시)
def fetch_market_ribbon() -> dict[str, tuple[float, float]]:
    """시장 지표 마지막값 + 전일대비% — yfinance 일괄 1회 조회. {sym: (last, chg%)}."""
    syms = [t for t, _, _ in _RIBBON_TICKERS]
    out: dict[str, tuple[float, float]] = {}
    try:
        raw = yf.download(syms, period="5d", interval="1d",
                          progress=False, threads=True)
        close = raw["Close"]
        for s in syms:
            try:
                ser = close[s].dropna()
                if len(ser):
                    last = float(ser.iloc[-1])
                    prev = float(ser.iloc[-2]) if len(ser) > 1 else last
                    chg = (last - prev) / prev * 100 if prev else 0.0
                    out[s] = (last, chg)
            except Exception:
                continue
    except Exception:
        pass
    return out


def render_market_ribbon() -> None:
    """모든 페이지 상단에 고정 노출되는 시장 지표 한 줄 (yfinance, 5분 캐시).

    종목 차트와 동일한 데이터 소스라 안정적이며, 캐시 결과라 재실행은 즉시.
    한국 관례: 상승=빨강, 하락=파랑."""
    data = fetch_market_ribbon()
    # 각 항목을 통째로 nowrap 처리 → 좁은 화면에서 단어 중간이 아니라 '항목 단위'로 줄바꿈.
    chips = [f"<span style='color:{COLORS['muted']};font-size:0.76rem;"
             f"white-space:nowrap'>시장 지표</span>"]
    for sym, label, fmt in _RIBBON_TICKERS:
        v = data.get(sym)
        if v is None:
            chips.append(f"<span style='white-space:nowrap'><b>{label}</b> –</span>")
            continue
        last, chg = v
        col = COLORS["up"] if chg >= 0 else COLORS["down"]
        arrow = "▲" if chg >= 0 else "▼"
        chips.append(
            f"<span style='white-space:nowrap'><b>{label}</b> {fmt.format(last)} "
            f"<span style='color:{col};font-size:0.76rem'>{arrow}{abs(chg):.1f}%</span></span>")
    inner = "".join(chips) if data else "<span>시세를 불러오지 못했습니다</span>"
    # flex-wrap + gap: 항목이 화면을 넘으면 깔끔히 다음 줄로 (단어 잘림 없음).
    st.markdown(
        f"<div style='background:{COLORS['panel']};border:1px solid {COLORS['grid']};"
        f"border-radius:10px;padding:7px 14px;margin:0 0 12px;font-size:0.85rem;"
        f"color:{COLORS['text']};display:flex;flex-wrap:wrap;justify-content:center;"
        f"align-items:center;gap:5px 18px;line-height:1.4'>{inner}</div>",
        unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# 📊 기업가치 비교 — yfinance 원본 수치만으로 밸류에이션·재무비율·DCF (추측 없음)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False, ttl=3600)
def fetch_valuation_cached(symbol: str) -> dict:
    """원본 info + 계산된 비율 (1시간 캐시)."""
    from app.analytics.valuation import compute_ratios, fetch_info
    info = fetch_info(symbol)
    return {"info": info, "ratios": compute_ratios(info)}


def _fmt_mult(v):  return f"{v:,.2f}" if v is not None else "N/A"       # 배수
def _fmt_pct(v):   return f"{v * 100:,.2f}%" if v is not None else "N/A"  # 소수→%
def _fmt_pctpt(v): return f"{v:,.2f}" if v is not None else "N/A"       # 이미 %p(D/E)
def _fmt_usd(v):   return f"${v:,.0f}" if v is not None else "N/A"
def _fmt_usd2(v):  return f"${v:,.2f}" if v is not None else "N/A"


# (key, 라벨, 포맷터) — 비교표 행 정의
_VAL_ROWS = [
    ("price",            "현재가",           _fmt_usd2),
    ("market_cap",       "시가총액",         _fmt_usd),
    ("enterprise_value", "기업가치 EV",      _fmt_usd),
    ("per_trailing",     "PER (후행)",       _fmt_mult),
    ("per_forward",      "PER (선행)",       _fmt_mult),
    ("pbr",              "PBR",             _fmt_mult),
    ("psr",              "PSR",             _fmt_mult),
    ("ev_ebitda",        "EV/EBITDA",       _fmt_mult),
    ("ev_revenue",       "EV/매출",         _fmt_mult),
    ("roe",              "ROE",             _fmt_pct),
    ("roa",              "ROA",             _fmt_pct),
    ("gross_margin",     "매출총이익률",     _fmt_pct),
    ("operating_margin", "영업이익률",       _fmt_pct),
    ("net_margin",       "순이익률",         _fmt_pct),
    ("debt_to_equity",   "부채비율 D/E(%)",  _fmt_pctpt),
    ("current_ratio",    "유동비율",         _fmt_mult),
    ("quick_ratio",      "당좌비율",         _fmt_mult),
    ("fcf_yield",        "FCF 수익률",       _fmt_pct),
    ("earnings_growth",  "이익성장률",       _fmt_pct),
    ("revenue_growth",   "매출성장률",       _fmt_pct),
]


# 통상(rule-of-thumb) 기준 — 시장 일반 참고치. * = 업종별 편차가 큼.
_VAL_BENCHMARK = {
    "PER (후행)":      "통상 15~25",
    "PER (선행)":      "통상 15~25",
    "PBR":            "통상 1~3",
    "PSR":            "통상 1~3 *",
    "EV/EBITDA":      "통상 8~15",
    "EV/매출":         "통상 1~5 *",
    "ROE":            "15%↑ 우량",
    "ROA":            "5%↑ 양호",
    "매출총이익률":     "40%↑ 우량 *",
    "영업이익률":       "15%↑ 우량",
    "순이익률":         "10%↑ 양호",
    "부채비율 D/E(%)":  "100 이하 안전",
    "유동비율":         "1.5↑ 건전",
    "당좌비율":         "1.0↑ 건전",
    "FCF 수익률":      "5%↑ 우량",
    "이익성장률":       "양수=성장",
    "매출성장률":       "양수=성장",
}


def _chart_val_multiples(data: dict, symbols: list[str]) -> go.Figure:
    """밸류에이션 배수(PER·PBR·PSR·EV/EBITDA) 종목 비교 — 낮을수록 저평가."""
    metrics = [("per_trailing", "PER"), ("pbr", "PBR"),
               ("psr", "PSR"), ("ev_ebitda", "EV/EBITDA")]
    fig = go.Figure()
    for i, s in enumerate(symbols):
        rr = data[s]["ratios"]
        ys = [rr.get(k) for k, _ in metrics]
        fig.add_trace(go.Bar(name=name_of(s), x=[lbl for _, lbl in metrics], y=ys,
            marker_color=PALETTE[i % len(PALETTE)],
            text=[f"{v:.1f}" if v is not None else "N/A" for v in ys],
            textposition="outside"))
    fig.update_layout(**BASE_LAYOUT, height=420, barmode="group",
        title=dict(text="밸류에이션 배수 비교 (낮을수록 저평가)",
                   font=dict(size=16, color=COLORS["accent"])),
        legend=dict(bgcolor="rgba(0,0,0,0)"))
    fig.update_yaxes(gridcolor=COLORS["grid"])
    return fig


def render_valuation_page(symbols: list[str]) -> None:
    st.caption("yfinance **원본 재무수치만으로** 밸류에이션·재무비율·DCF 를 계산합니다. "
               "원본에 없으면 **추정하지 않고 N/A**(임의 추측 금지), 절사 없이 표시합니다.")
    help_box(
        "**밸류에이션 배수 (낮을수록 저평가)**\n"
        "- **PER**: 주가÷주당순이익. **PBR**: 주가÷주당순자산. **PSR**: 시총÷매출.\n"
        "- **EV/EBITDA**: 기업가치÷영업현금성이익 — 부채까지 반영해 PER 보완.\n"
        "- **기업가치 EV** = 시가총액 + 순부채. 인수 관점의 회사 전체 값.\n\n"
        "**수익성·재무 건전성**\n"
        "- **ROE/ROA**: 자기자본·자산 대비 이익률(높을수록 효율적). 자사주 매입으로 ROE>100%도 가능.\n"
        "- **영업/순이익률**: 매출 1달러당 남는 이익. **D/E**: 부채÷자기자본(낮을수록 안전).\n"
        "- **유동/당좌비율**: 단기 지급능력(1↑ 양호). **FCF 수익률**: 잉여현금흐름÷시총.\n\n"
        "**💰 DCF·NPV (내재가치)**\n"
        "- 미래 잉여현금흐름(FCF)을 현재가치로 할인해 **적정 주가**를 추정합니다.\n"
        "- FCF·주식수·성장률은 **원본**, 할인율·영구성장률·기간은 **사용자가 정하는 명시적 가정**입니다.\n"
        "- 상승여력 = (내재가치 − 현재가)÷현재가. **가정에 매우 민감하니 참고용**입니다.\n\n"
        "⚠️ 원본 결측 종목은 일부 칸이 N/A 입니다. 교육용 참고 지표이며 투자 조언이 아닙니다.")

    with st.spinner("원본 재무지표 불러오는 중... (종목당 1시간 캐시)"):
        data = {s: fetch_valuation_cached(s) for s in symbols}

    # 업종 컨텍스트 — 통상 기준은 시장 일반치라 같은 업종끼리 비교해야 정확
    secs = []
    for s in symbols:
        info = data[s]["info"]
        sec, ind = info.get("sector"), info.get("industry")
        if sec:
            secs.append(f"**{name_of(s)}**: {sec}" + (f" / {ind}" if ind else ""))
    if secs:
        st.caption("🏷️ 업종 — " + "  ·  ".join(secs))

    # ── 핵심 지표 비교표 (지표=행, 종목=열) + 중앙값 + 통상 기준 ──────────────
    st.subheader("📊 핵심 지표 비교 (원본 + 통상 기준)")
    tbl = {}
    for s in symbols:
        rr = data[s]["ratios"]
        tbl[f"{name_of(s)} ({s})"] = {label: fmt(rr.get(key))
                                      for key, label, fmt in _VAL_ROWS}
    # 선택 종목 중앙값 (실제 원본값들로 계산 — 빠른 동종 비교용)
    if len(symbols) >= 2:
        med = {}
        for key, label, fmt in _VAL_ROWS:
            vals = [data[s]["ratios"].get(key) for s in symbols]
            vals = [v for v in vals if v is not None]
            med[label] = fmt(float(np.median(vals))) if vals else "N/A"
        tbl["중앙값(선택)"] = med
    # 통상(rule-of-thumb) 기준
    tbl["📏 통상 기준"] = {label: _VAL_BENCHMARK.get(label, "–")
                       for key, label, fmt in _VAL_ROWS}

    st.dataframe(pd.DataFrame(tbl), width="stretch")
    st.caption("✔ PER/PBR/PSR 은 Yahoo 원본값과 '원본 구성요소로 직접 계산한 값'이 소수점까지 "
               "일치함을 검증(결측은 N/A·추정 안 함). **📏 통상 기준**은 시장 일반 참고치이며 "
               "**업종별 편차가 큽니다(*)** — 같은 업종끼리 비교하세요.")

    # ── 밸류에이션 배수 시각 비교 ────────────────────────────────────────────
    if len(symbols) >= 2:
        st.plotly_chart(_chart_val_multiples(data, symbols), width="stretch")

    # ── DCF·NPV ──────────────────────────────────────────────────────────────
    st.subheader("💰 DCF·NPV 내재가치")
    st.caption("FCF·주식수·성장률 = **원본** / 할인율·영구성장률·기간 = **명시적 가정**(아래 조절). "
               "원본 FCF·주식수·성장률이 없으면 계산하지 않습니다.")
    a = st.columns(4)
    disc = a[0].slider("할인율 r (%)", 6.0, 15.0, 10.0, 0.5,
                       help="요구수익률(WACC). 높을수록 내재가치↓") / 100
    tg = a[1].slider("영구성장률 gt (%)", 0.0, 4.0, 2.5, 0.25,
                     help="예측기간 이후 영구 성장 가정. r 보다 작아야 함") / 100
    yrs = a[2].slider("예측기간 (년)", 5, 15, 10)
    use_src_g = a[3].toggle("성장률: 원본 사용", value=True,
                            help="끄면 성장률을 직접 지정 (원본 성장률이 과도할 때)")
    g_override = None
    if not use_src_g:
        g_override = st.slider("성장률 g 직접지정 (%)", -10.0, 40.0, 10.0, 1.0) / 100

    from app.analytics.valuation import dcf_npv
    rows = []
    for s in symbols:
        d = dcf_npv(data[s]["info"], discount_rate=disc, terminal_growth=tg,
                    years=yrs, growth_override=g_override)
        if d["ok"]:
            rows.append({
                "종목": f"{name_of(s)} ({s})",
                "현재가": f"${d['price']:,.2f}" if d["price"] else "N/A",
                "DCF 내재가치/주": f"${d['intrinsic_ps']:,.2f}",
                "상승여력": (f"{d['upside'] * 100:+.1f}%"
                          if d["upside"] is not None else "N/A"),
                "NPV(기업가치)": f"${d['npv_total']:,.0f}",
                "성장률 사용": f"{d['g_used'] * 100:.1f}%",
            })
        else:
            rows.append({"종목": f"{name_of(s)} ({s})", "현재가": "-",
                         "DCF 내재가치/주": "계산 불가", "상승여력": d["reason"],
                         "NPV(기업가치)": "-", "성장률 사용": "-"})
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    st.caption("⚠️ 내재가치는 가정(특히 성장률·할인율)에 매우 민감합니다. FCF(잉여현금흐름)를 "
               "주주현금흐름으로 단순화했습니다. 교육용 참고치이며 투자 조언이 아닙니다.")


# ─────────────────────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────────────────────

_hdr_open, _ = us_market_status()
render_header(
    "ALPHA", "DESK", subtitle="QUANT RESEARCH",
    clocks=[f"KST {datetime.now(KST):%H:%M}", f"ET {datetime.now(ET):%H:%M}"],
    status_color=("#22c55e" if _hdr_open else COLORS["muted"]),
    status_text=("US OPEN" if _hdr_open else "US CLOSED"),
)

with st.sidebar:
    st.header("⚙️ 설정")

    # 🎨 테마 — 다크모드 토글 + 메인 색상 피커 (차트 생성 '전'에 적용)
    dark_mode = st.toggle("🌙 다크 모드", value=True)
    accent_name = st.selectbox("강조색", list(ACCENT_PRESETS), index=0,
                               help="상승=빨강·하락=파랑과 충돌하지 않는 색만 제공합니다.")
    accent = ACCENT_PRESETS[accent_name]
    set_theme(dark=dark_mode, accent=accent)
    inject_terminal_css()

    # 멀티페이지에서 위젯 상태가 GC되지 않도록 입력값 유지 — 페이지 이동 시 티커 보존
    for _k in ("nav_group", "ms_stocks", "txt_stocks", "ms_crypto", "txt_crypto",
               "opt_pick", "opt_custom"):
        if _k in st.session_state:
            st.session_state[_k] = st.session_state[_k]

    st.divider()
    # 페이지를 3개 카테고리로 그룹핑 (코이핀형 좌측 네비)
    PAGE_GROUPS = {
        "📈 종목 분석": ["📈 실시간", "⭐ AI 추천", "📊 기업가치 비교", "🧪 모델 검증",
                       "💰 가치·ML 분석", "🎯 눌림목 스코어", "📊 수익률·낙폭",
                       "🔗 상관관계", "🔎 스크리너"],
        "🧮 전략·포트폴리오": ["🤖 전략 백테스트", "🎯 포트폴리오 3D", "🔬 전략 최적화 3D",
                          "🌊 변동성 서피스 3D", "💼 모의 포트폴리오"],
        "🌐 마켓·기타": ["💎 크립토", "📉 옵션", "🌐 거시·경제", "📰 뉴스·AI 심리"],
    }
    nav_group = st.radio("카테고리", list(PAGE_GROUPS), key="nav_group")
    page = st.radio("페이지", PAGE_GROUPS[nav_group], label_visibility="collapsed")
    st.divider()

    # 페이지별 티커 입력 — 크립토/옵션/스크리너는 전용, 나머지는 공통 주식 선택
    symbols: list[str] = []
    crypto_syms: list[str] = []
    option_sym = ""
    screener_universe = SCREENER_UNIVERSE
    rsi_lo, rsi_hi, min_ret3m, only_above_ma50 = 0.0, 100.0, -100.0, False
    sort_key = "기술점수"
    show_bb = show_ichimoku = show_macd = True
    mode, auto, interval = "일봉 (6개월)", False, 10

    if page == "🔎 스크리너":
        st.subheader("스크리너 조건")
        uni = st.radio("유니버스", ["대형주 49", "프리셋 10", "직접 입력"], index=0)
        if uni == "프리셋 10":
            screener_universe = list(PRESET_TICKERS)
        elif uni == "직접 입력":
            txt = st.text_area("티커 (쉼표/줄바꿈 구분)", value="AAPL, MSFT, NVDA, TSLA")
            screener_universe = [t.strip().upper()
                                 for t in txt.replace("\n", ",").split(",") if t.strip()]
        rsi_lo, rsi_hi = st.slider("RSI 범위", 0, 100, (0, 100))
        min_ret3m = st.slider("최소 3개월 수익률(%)", -50, 50, -50)
        only_above_ma50 = st.toggle("MA50 위 종목만", value=False)
        sort_key = st.selectbox("정렬 기준", ["기술점수", "RSI", "3개월 수익률",
                                "1개월 수익률", "52주 고점대비", "변동성(낮은순)"])

    elif page == "💎 크립토":
        st.subheader("암호화폐 선택")
        st.session_state.setdefault("ms_crypto", ["BTC-USD", "ETH-USD", "SOL-USD"])
        cpick = st.multiselect("프리셋에서 고르기", options=list(CRYPTO_PRESETS),
            key="ms_crypto", format_func=lambda s: f"{CRYPTO_PRESETS[s]} ({s})")
        ccustom = st.text_input("직접 입력 (쉼표로 구분)", key="txt_crypto",
            placeholder="예: DOT-USD, LTC-USD", help="yfinance 형식: 코인심볼-USD")
        cextra = [t.strip().upper() for t in ccustom.split(",") if t.strip()]
        crypto_syms = list(dict.fromkeys(cpick + cextra))
        st.divider()
        st.caption("📐 차트 지표")
        show_bb = st.toggle("볼린저밴드", value=True)
        show_ichimoku = st.toggle("일목 구름대", value=True)
        show_macd = st.toggle("MACD", value=True)

    elif page == "📉 옵션":
        st.subheader("옵션 기초자산")
        st.session_state.setdefault("opt_pick", OPTION_PRESETS[0])
        opt_pick = st.selectbox("프리셋", OPTION_PRESETS, key="opt_pick")
        opt_custom = st.text_input("직접 입력 (1개)", key="opt_custom",
            placeholder="예: GOOGL", help="옵션이 상장된 미국 주식·ETF 티커")
        option_sym = (opt_custom.strip().upper() or opt_pick)

    elif page in ("💼 모의 포트폴리오", "🌐 거시·경제"):
        # 종목 선택 불필요 — 페이지 자체에서 입력/조회
        st.caption("이 페이지는 종목 선택이 필요 없습니다.")

    else:
        st.subheader("종목 선택")
        st.session_state.setdefault("ms_stocks", [])   # 강제 디폴트 없음 — 빈 상태로 시작
        picked = st.multiselect("프리셋에서 고르기", options=list(PRESET_TICKERS.keys()),
            key="ms_stocks", format_func=lambda s: f"{s} · {PRESET_TICKERS[s]}",
            placeholder="종목을 선택하세요")
        custom = st.text_input("직접 입력 (쉼표로 구분)", key="txt_stocks",
            placeholder="예: COST, ORCL, BRK-B",
            help="선택은 페이지를 이동해도 유지됩니다. 비우면 빈 상태로 유지돼요.")
        extra = [t.strip().upper() for t in custom.split(",") if t.strip()]
        symbols = list(dict.fromkeys(picked + extra))

        if page == "📈 실시간":
            st.divider()
            mode = st.radio("차트 종류", ["인트라데이 (1분봉)", "일봉 (6개월)"], index=0)
            auto = st.toggle("자동 갱신 (실시간)", value=True)
            interval = st.slider("갱신 주기 (초)", 5, 60, 10, disabled=not auto)
            st.divider()
            st.caption("📐 차트 지표")
            show_bb = st.toggle("볼린저밴드", value=True)
            show_ichimoku = st.toggle("일목 구름대", value=True)
            show_macd = st.toggle("MACD", value=True)

    st.divider()
    st.caption(f"마지막 갱신: {datetime.now(KST):%Y-%m-%d %H:%M:%S} (한국)")

# 시장 지표 띠 — 사이드바(테마·CSS 주입) 이후에 그려 헤더 스타일이 적용된 상태로 노출.
# 비차단(자체 캐시·짧은 타임아웃)이라 FRED 가 느려도 페이지를 멈추지 않는다.
render_market_ribbon()

# ── 전용 페이지 — 자체 입력 사용 (공통 symbols 가드 우회) ─────────────────────
if page == "💎 크립토":
    render_crypto_page(crypto_syms, show_bb, show_ichimoku, show_macd)
    st.stop()
elif page == "📉 옵션":
    render_options_page(option_sym)
    st.stop()
elif page == "🔎 스크리너":
    render_screener_page(screener_universe, rsi_lo, rsi_hi, min_ret3m,
                         only_above_ma50, sort_key)
    st.stop()
elif page == "💼 모의 포트폴리오":
    render_paper_page()
    st.stop()
elif page == "🌐 거시·경제":
    render_macro_page()
    st.stop()
elif page == "⭐ AI 추천":
    render_ai_reco_page(symbols)   # 자동 추천은 종목 선택 불필요 (가드 우회)
    st.stop()

if not symbols:
    st.markdown(
        f"<div style='text-align:center; padding:5rem 1rem; color:{COLORS['muted']}'>"
        f"<div style='font-size:2.2rem; margin-bottom:0.6rem'>📊</div>"
        f"<div style='font-size:1.15rem; font-weight:700; color:{COLORS['text']}'>"
        f"분석할 종목을 선택하세요</div>"
        f"<div style='font-size:0.9rem; margin-top:0.4rem'>"
        f"왼쪽 사이드바 <b>종목 선택</b>에서 프리셋을 고르거나 티커를 직접 입력하면 "
        f"여기에 분석이 표시됩니다.</div></div>",
        unsafe_allow_html=True,
    )
    st.stop()


# ── 페이지: 실시간 ────────────────────────────────────────────────────────────
if page == "📈 실시간":
    is_open, status_label = us_market_status()
    (st.success if is_open else st.warning)(
        f"{status_label}  ·  데이터는 약 15분 지연된 무료 Yahoo 데이터입니다."
        + ("" if is_open else "  장이 닫혀 있어 가격은 마지막 거래일 기준으로 고정 표시됩니다."))
    help_box(
        "- **캔들**: 빨강=상승 봉, 파랑=하락 봉 (한국 관례, 토스증권과 동일). 위/아래 꼬리는 장중 고가·저가.\n"
        "- **MA20 / MA50**: 20일·50일 이동평균(추세선). MA20이 MA50 위에 있으면 단기 상승 우위.\n"
        "- **볼린저밴드(보라 음영, 20일·±2σ)**: 가격이 상단에 닿으면 단기 과열, 하단에 닿으면 과매도 신호. "
        "밴드 폭이 좁아지면(스퀴즈) 곧 큰 변동이 올 수 있음.\n"
        "- **일목 구름대(일목균형표)**: 빨간 구름(양운)=지지 구간, 파란 구름(음운)=저항 구간. "
        "가격이 구름 **위**면 상승 추세, **아래**면 하락 추세, 구름 **안**이면 방향 탐색 중. "
        "구름이 두꺼울수록 지지/저항이 강하고, 구름은 26봉 앞(미래)까지 그려집니다. "
        "점선은 전환선(9)·기준선(26) — 전환선이 기준선을 위로 돌파하면 단기 매수 신호.\n"
        "- **거래량**: 봉 색과 동일. 급등/급락 시 거래량이 함께 터지면 신뢰도↑.\n"
        "- **RSI(14)**: 70↑ 과매수(조정 가능), 30↓ 과매도(반등 가능), 50이 중립.\n"
        "- **MACD(12,26,9)**: 파란 선(MACD)이 주황 선(시그널)을 위로 돌파하면 매수, 아래로 돌파하면 매도 신호. "
        "막대(히스토그램)는 두 선의 간격 — 빨강이 커지면 상승 탄력 강화, 파랑이 커지면 하락 탄력 강화.\n"
        "- 사이드바 **📐 차트 지표** 토글로 볼린저밴드·구름대·MACD를 켜고 끌 수 있습니다.\n"
        "- 상단 숫자(등락률)는 **전일 종가 대비** 변화입니다.")

    bust = int(time.time() // (interval if auto else 10))
    cols = st.columns(len(symbols))
    data_cache: dict[str, pd.DataFrame] = {}
    for col, sym in zip(cols, symbols):
        try:
            df, last, prev = fetch_intraday(sym, mode, bust)
            if df is None:
                col.markdown(ticker_tile(name_of(sym), sym, None, None, None),
                             unsafe_allow_html=True)
                continue
            data_cache[sym] = df
            change = last - prev
            pct = (change / prev * 100) if prev else 0.0
            col.markdown(ticker_tile(name_of(sym), sym, last, change, pct),
                         unsafe_allow_html=True)
        except Exception:
            col.markdown(ticker_tile(name_of(sym), sym, None, None, None),
                         unsafe_allow_html=True)

    st.divider()
    if data_cache:
        tabs = st.tabs([f"{name_of(s)} ({s})" for s in data_cache])
        for tab, (sym, df) in zip(tabs, data_cache.items()):
            tab.plotly_chart(
                chart_candlestick(df, sym, name_of(sym), show_bb=show_bb,
                                  show_ichimoku=show_ichimoku, show_macd=show_macd),
                width="stretch")

    if auto:
        time.sleep(interval)
        st.rerun()


# ── 페이지: 뉴스 · AI 심리 (가격 데이터 불필요 — 일봉 로드 생략) ──────────────
elif page == "📰 뉴스·AI 심리":
    render_news_page(symbols)


# ── 페이지: 기업가치 비교 (원본 재무 — 일봉 가격 로드 불필요) ──────────────────
elif page == "📊 기업가치 비교":
    render_valuation_page(symbols)


# ── 분석 페이지 공통: 일봉 로드 ───────────────────────────────────────────────
else:
    with st.spinner("일봉 데이터 불러오는 중... (최초 1회, 이후 캐시)"):
        frames = load_frames(symbols)
    if not frames:
        st.error("데이터를 불러오지 못했습니다. 티커를 확인하세요.")
        st.stop()
    if len(frames) < len(symbols):
        missing = [s for s in symbols if s not in frames]
        st.warning(f"데이터가 부족해 제외된 종목: {', '.join(missing)}")

    need2 = page in ("🔗 상관관계", "🎯 포트폴리오 3D", "🌊 변동성 서피스 3D")
    if need2 and len(frames) < 2:
        st.info("이 분석은 종목이 **2개 이상** 필요합니다. 사이드바에서 더 추가하세요.")
        st.stop()

    if page == "🧪 모델 검증":
        render_validation_page(symbols)

    elif page == "📊 수익률·낙폭":
        help_box(
            "- **위 그래프(누적 수익률)**: 2년 전 100% 기준, 지금까지 몇 % 올랐는지. 선이 높을수록 많이 상승.\n"
            "- **아래 그래프(낙폭)**: 직전 고점 대비 얼마나 빠졌는지. 0에 붙어 있을수록 안정적,\n"
            "  깊게 파일수록 그 시기에 큰 손실을 견뎌야 했다는 뜻(최대낙폭=MDD).\n"
            "- 수익률이 비슷하면 **낙폭이 얕은 종목**이 심리적으로 보유하기 쉽습니다.\n\n"
            "**📐 리스크 지표 표 읽는 법** (헤지펀드·운용사가 보는 기준)\n"
            "- **CAGR**: 연평균 복리 수익률. | **연변동성**: 연환산 표준편차(출렁임).\n"
            "- **샤프**: 위험 1단위당 초과수익. 1↑ 양호, 2↑ 우수. | **소르티노**: 하락 변동성만 위험으로 본 샤프.\n"
            "- **칼마**: CAGR ÷ |MDD| — 최악의 낙폭 대비 수익 효율. | **MDD**: 최대 낙폭.\n"
            "- **VaR95**: 정상적인 날 95%는 하루 손실이 이 값 이내. **CVaR95**: 나머지 최악 5% 날의 평균 손실.\n"
            "- **베타**: S&P500이 1% 움직일 때 이 종목이 평균 몇 % 움직이나. 1↑ 시장보다 민감.\n"
            "- **승률(일)**: 상승 마감한 거래일 비율.")
        st.plotly_chart(chart_equity(frames), width="stretch")

        st.subheader("전문 리스크·성과 지표 (최근 2년, 샤프순 정렬)")
        metrics = risk_metrics_table(frames)
        if len(metrics):
            st.dataframe(metrics, width="stretch", hide_index=True)
            st.caption("무위험수익률 2% 가정 · 일봉 기준 연환산 · VaR/CVaR는 1일 95% 신뢰수준 · 베타는 SPY 대비")

    elif page == "🔗 상관관계":
        help_box(
            "- **왼쪽 히트맵**: 두 종목이 같이 움직이는 정도. +1(빨강)=거의 똑같이, 0=무관, -1(파랑)=반대.\n"
            "- 분산투자는 **상관이 낮은(0에 가까운)** 종목을 섞을수록 효과가 큽니다. 전부 진한 빨강이면 사실상 한 종목.\n"
            "- **오른쪽 분포**: 일별 수익률의 퍼짐. 폭이 넓을수록 변동성이 큰(=위험한) 종목.")
        st.plotly_chart(chart_correlation(frames), width="stretch")

    elif page == "🎯 포트폴리오 3D":
        help_box(
            "- 점 하나 = 종목 비중을 무작위로 섞은 **포트폴리오 하나**. 5,000개를 시뮬레이션했습니다.\n"
            "- 축: **X=위험(변동성)**, **Y=기대수익**, **Z·색=샤프(위험 1단위당 수익)**. 색이 밝을수록 효율적.\n"
            "- **★ 빨간 다이아 = 샤프 최고 지점**(가장 효율적). 아래에 그 추천 비중이 표시됩니다.\n"
            "- 마우스로 드래그하면 3D 회전. 왼쪽 위로 갈수록 '적은 위험에 높은 수익'.")
        fig, alloc = chart_frontier(frames)
        st.plotly_chart(fig, width="stretch")
        st.success(f"★ 최적 샤프 포트폴리오 비중 →  {alloc}")

    elif page == "🔬 전략 최적화 3D":
        target = st.selectbox("분석할 종목", list(frames.keys()),
                              format_func=lambda s: f"{name_of(s)} ({s})")
        help_box(
            "- **MA 크로스 전략**(단기 이평이 장기 이평을 넘으면 매수)을, 기간 조합을 바꿔가며 백테스트한 결과.\n"
            "- 축: **X=단기 MA, Y=장기 MA, Z·색=샤프 비율**. 빨강(높은 곳)일수록 과거 성과가 좋았던 조합.\n"
            "- **샤프 비율이란?** 위험(변동성) 1단위당 얻은 수익 = **수익률 ÷ 출렁임**. "
            "같은 수익이라도 **덜 흔들리며** 번 전략일수록 높습니다. "
            "대략 **1↑ 양호 · 2↑ 우수 · 0 이하면 위험 대비 보상 없음**. "
            "단순 수익률과 달리 '꾸준함'까지 함께 평가하는 지표입니다.\n"
            "- **★ = 과거 데이터상 최적 조합**. 단, 과거 최적이 미래를 보장하진 않습니다(과최적화 주의).")

        fig_ps, best_ps = chart_param_surface(frames[target], target)

        # 최적 파라미터를 3D 범례와 별개로 차트 위에 바로 표시
        st.markdown("##### ⭐ 최적 파라미터 (과거 2년 샤프 최대)")
        mc = st.columns(3)
        mc[0].metric("단기 MA", f"{best_ps['fast']}일")
        mc[1].metric("장기 MA", f"{best_ps['slow']}일")
        mc[2].metric("샤프 비율", f"{best_ps['sharpe']:.2f}",
                     help="위험(변동성) 1단위당 수익 = 수익률 ÷ 출렁임. "
                          "1↑ 양호 · 2↑ 우수 · 0 이하면 위험 대비 보상 없음.")
        st.caption(f"{name_of(target)} ({target}) · 단기 5–55일 × 장기 20–190일 "
                   f"**{best_ps['n_combos']}개 조합**을 백테스트해 샤프가 가장 높은 조합입니다.")

        st.plotly_chart(fig_ps, width="stretch")

    elif page == "🌊 변동성 서피스 3D":
        help_box(
            "- **종목 × 시간**별 30일 실현 변동성(연환산)을 지형도로 표현.\n"
            "- 색: **파랑=잔잔함(저위험), 빨강=출렁임(고위험)**. 높이도 변동성 크기.\n"
            "- 솟아오른 **봉우리 = 그 종목이 그 시기에 크게 흔들렸다**는 뜻(급락/급등 구간).\n"
            "- 드래그로 회전. 특정 종목 띠가 전반적으로 붉으면 평소 변동성이 큰 종목.")
        st.plotly_chart(chart_vol_surface(frames), width="stretch")

    elif page == "🤖 전략 백테스트":
        st.caption("규칙 기반 **알고리즘 트레이딩 전략**을 과거 2년 데이터로 검증합니다. "
                   "신호 다음 날 체결(룩어헤드 방지) · 거래비용 반영 · 롱/현금 전략.")
        help_box(
            "- **전략 선택 후 파라미터를 조절**하면 즉시 재계산됩니다.\n"
            "- **파란 실선** = 전략 누적 수익률, **회색 점선** = 그냥 사서 보유(바이앤홀드).\n"
            "  전략이 바이앤홀드를 **수수료까지 내고도** 이기는지가 핵심입니다.\n"
            "- **아래 빨간 음영** = 포지션 보유 구간. 비어 있으면 현금 보유 중.\n"
            "- **노출비율** = 전체 기간 중 주식을 들고 있던 비율. 노출이 낮은데 수익이 비슷하면 효율적.\n"
            "- ⚠️ 과거 성과가 미래를 보장하지 않으며, 파라미터를 과하게 맞추면 **과최적화** 위험이 있습니다.")

        c1, c2, c3 = st.columns([0.35, 0.4, 0.25])
        target = c1.selectbox("종목", list(frames.keys()),
                              format_func=lambda s: f"{name_of(s)} ({s})")
        strategy = c2.selectbox("전략", ["MA 크로스", "RSI 역추세", "MACD",
                                        "볼린저 평균회귀", "모멘텀 (12-1)"])
        cost_bps = c3.number_input("편도 비용(bp)", 0.0, 100.0, 10.0, 5.0,
                                   help="수수료+슬리피지. 10bp = 0.1%")

        p: dict = {}
        if strategy == "MA 크로스":
            pc1, pc2 = st.columns(2)
            p["fast"] = pc1.slider("단기 MA(일)", 5, 60, 20, 5)
            p["slow"] = pc2.slider("장기 MA(일)", 30, 200, 50, 10)
            if p["fast"] >= p["slow"]:
                st.warning("단기 MA는 장기 MA보다 짧아야 합니다.")
                st.stop()
        elif strategy == "RSI 역추세":
            pc1, pc2, pc3 = st.columns(3)
            p["rsi_period"] = pc1.slider("RSI 기간", 7, 28, 14)
            p["buy_th"] = pc2.slider("매수 기준 (RSI <)", 15, 40, 30)
            p["sell_th"] = pc3.slider("청산 기준 (RSI >)", 45, 80, 55)
        elif strategy == "모멘텀 (12-1)":
            pc1, pc2 = st.columns(2)
            p["look"] = pc1.slider("룩백(일)", 63, 252, 252, 21)
            p["skip"] = pc2.slider("최근 제외(일)", 0, 42, 21, 7)

        bt = run_backtest(frames[target], strategy, p, cost_bps=cost_bps)
        st.plotly_chart(chart_backtest(bt, target, strategy), width="stretch")

        s, b = bt["strat"], bt["bench"]
        m = st.columns(6)
        m[0].metric("전략 누적수익", f"{s['total']:+.1%}",
                    f"{(s['total'] - b['total']) * 100:+.1f}%p vs B&H")
        m[1].metric("전략 CAGR", f"{s['cagr']:+.1%}", f"B&H {b['cagr']:+.1%}",
                    delta_color="off")
        m[2].metric("샤프", f"{s['sharpe']:.2f}", f"B&H {b['sharpe']:.2f}",
                    delta_color="off")
        m[3].metric("MDD", f"{s['mdd']:.1%}", f"B&H {b['mdd']:.1%}",
                    delta_color="off")
        m[4].metric("거래 횟수", f"{bt['n_trades']}회",
                    f"승률 {bt['win_rate']:.0%}" if bt["win_rate"] is not None else "청산 트레이드 없음",
                    delta_color="off")
        m[5].metric("노출 비율", f"{bt['exposure']:.0%}")

    elif page == "💰 가치·ML 분석":
        st.caption("재무 기반 **가치투자 점수** + **머신러닝 예측**을 결합합니다. 종목당 약 6초(가치 1초 + ML 5초).")
        help_box(
            "**종합 점수 = 기술 20% + 가치 40% + ML 40%** 가중 평균(0~100).\n\n"
            "- 🎯 **레이더**: 5개 가치 축 — 피오트로스키(재무건전성 0~9), 버핏체크(경쟁우위 0~10),\n"
            "  DCF(내재가치 상승여력), 그레이엄(안전마진), 알트만Z(부도위험 낮을수록 높음). **넓을수록 우량**.\n"
            "- 📊 **랭킹**: 종합 점수 막대(65↑ 매수 / 50~65 보유 / 그 이하 관망·회피). "
            "오른쪽 산점도는 가치(가로) vs ML 상승확률(세로) — **오른쪽 위**가 둘 다 좋은 종목.\n"
            "- 🔬 **신뢰도/레짐**: ML이 얼마나 확신하는지, 그리고 현재가 상승장/횡보장/하락장 중 어디인지 확률.\n"
            "- 💡 **추천 & 목표가**: 월가 애널리스트 컨센서스(목표가·투자의견·인원수, Yahoo 제공) + DCF 적정가 +\n"
            "  통계적 1개월 예상범위(±1σ)를 함께 표시. **목표가 상승여력**과 **신뢰도**로 판단을 보조하세요.\n\n"
            "⚠️ **교육용 참고 지표이며 투자 조언이 아닙니다.** 무료 재무데이터라 일부 종목은 값이 비어 보정될 수 있습니다.")

        if "adv_results" not in st.session_state:
            st.session_state.adv_results = {}

        if st.button("🚀 가치·ML 분석 실행 / 갱신", type="primary"):
            st.session_state.adv_results = run_advanced(symbols, frames)

        res = {s: st.session_state.adv_results[s]
               for s in symbols if s in st.session_state.adv_results}

        if not res:
            st.info("위 **🚀 분석 실행** 버튼을 눌러주세요. (종목당 ~6초)")
        else:
            order = sorted(res, key=lambda x: res[x]["final"], reverse=True)

            # 💡 종목별 추천 & 목표가 카드
            st.subheader("종목별 추천 & 목표가")
            for s in order:
                r = res[s]; a = r.get("analyst") or {}; p = r["pred"]; vs = r["vs"]
                cur = r["cur"]; tgt = a.get("targetMeanPrice")
                with st.container(border=True):
                    st.markdown(f"#### {name_of(s)} ({s}) — {verdict(r['final'])}  ·  종합 {r['final']:.0f}/100")
                    m = st.columns(4)
                    m[0].metric("현재가", f"${cur:,.2f}")
                    m[1].metric("애널리스트 목표가",
                                f"${tgt:,.2f}" if tgt else "-",
                                f"{(tgt - cur) / cur * 100:+.1f}%" if tgt else None)
                    m[2].metric("애널리스트 의견", reco_kor(a.get("recommendationKey")),
                                f"{a.get('numberOfAnalystOpinions') or 0}명")
                    m[3].metric("ML 신호 · 신뢰도", p.signal if p else "-",
                                f"신뢰도 {p.confidence:.0%}" if p else None)
                    parts = []
                    if a.get("targetLowPrice") and a.get("targetHighPrice"):
                        parts.append(f"애널 목표범위 ${a['targetLowPrice']:,.2f}~${a['targetHighPrice']:,.2f}")
                    if vs and vs.dcf_upside:
                        parts.append(f"DCF 적정가 ${cur * (1 + vs.dcf_upside / 100):,.2f} ({vs.dcf_upside:+.1f}%)")
                    lo, hi = r["band"]
                    parts.append(f"1개월 통계적 예상범위(±1σ, 약 68%) ${lo:,.2f}~${hi:,.2f}")
                    st.caption("  ·  ".join(parts))

            # 📋 요약 표
            st.subheader("요약 표")
            rows = []
            for s in order:
                r = res[s]; vs = r["vs"]; p = r["pred"]; a = r.get("analyst") or {}
                cur = r["cur"]; tgt = a.get("targetMeanPrice")
                rows.append({
                    "종목": f"{name_of(s)} ({s})",
                    "종합점수": round(r["final"], 1),
                    "판단": verdict(r["final"]),
                    "애널의견": reco_kor(a.get("recommendationKey")),
                    "목표가": f"${tgt:,.2f}" if tgt else "-",
                    "목표상승": f"{(tgt - cur) / cur * 100:+.1f}%" if tgt else "-",
                    "DCF상승": f"{vs.dcf_upside:+.1f}%" if vs else "-",
                    "ML신호": p.signal if p else "-",
                    "상승확률": f"{p.up_probability:.0%}" if p else "-",
                    "신뢰도": f"{p.confidence:.0%}" if p else "-",
                    "레짐": p.regime if p else "-",
                })
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

            # 🔬 펀더멘털 디테일 — ML 확률 틸트의 근거가 되는 재무 체력
            with st.expander("🔬 펀더멘털 디테일 (PEG·지속ROE·FCF — ML 확률 틸트의 근거)"):
                st.markdown(
                    "가치투자 핵심 지표. **PEG < 1** = 성장 대비 저평가, "
                    "**지속ROE(4년) ≥ 15%** = 버핏식 우량기업, **FCF수익률 ≥ 5%** = 현금창출력 대비 저렴. "
                    "종합점수(0~100, 50=중립)가 ML 상승확률을 최대 ±6%p 보정합니다.")
                frows = []
                for s in order:
                    f = res[s].get("fund") or {}
                    pred = res[s]["pred"]
                    frows.append({
                        "종목": f"{name_of(s)} ({s})",
                        "PEG": round(f["peg_ratio"], 2) if "peg_ratio" in f else "-",
                        "ROE": f"{f['roe']:.0%}" if "roe" in f else "-",
                        "지속ROE(4y)": f"{f['roe_mean_4y']:.0%}" if "roe_mean_4y" in f else "-",
                        "ROE일관성": f"{f['roe_consistency']:.0%}" if "roe_consistency" in f else "-",
                        "FCF수익률": f"{f['fcf_yield']:.1%}" if "fcf_yield" in f else "-",
                        "순이익률": f"{f['profit_margin']:.0%}" if "profit_margin" in f else "-",
                        "이익성장": f"{f['earnings_growth']:+.0%}" if "earnings_growth" in f else "-",
                        "D/E": round(f["debt_to_equity"], 2) if "debt_to_equity" in f else "-",
                        "P/B": round(f["price_to_book"], 1) if "price_to_book" in f else "-",
                        "DCF갭": f"{f['dcf_upside']:+.1f}%" if "dcf_upside" in f else "-",
                        "종합점수": round(res[s].get("fund_score", 50.0), 0),
                        "확률틸트": (f"{pred.fundamental_tilt:+.1%}"
                                   if pred is not None and hasattr(pred, "fundamental_tilt") else "-"),
                    })
                st.dataframe(pd.DataFrame(frows), width="stretch", hide_index=True)

            value_results = {s: res[s]["vs"] for s in res}
            c1, c2 = st.columns(2)
            c1.plotly_chart(chart_value_radar(value_results), width="stretch")
            c2.plotly_chart(chart_ranking(res), width="stretch")
            if any(res[s]["pred"] for s in res):
                st.plotly_chart(chart_confidence(res), width="stretch")

            warned = [(s, res[s]["vs"].warnings) for s in res
                      if res[s]["vs"] and res[s]["vs"].warnings]
            if warned:
                st.subheader("경고")
                for s, ws in warned:
                    for w in ws:
                        st.write(f"- **{name_of(s)}**: {w}")

    elif page == "🎯 눌림목 스코어":
        st.caption("하나증권 이경수 위원 **'과열주 눌림목'** 전략 응용 — 3개월 강세 종목이 1개월 눌렸을 때, "
                   "실적·목표주가 상향으로 확인된 종목을 매수 후보로.")
        help_box(
            "**핵심 아이디어(하나증권 실전 퀀트 2026.05.27):** 강하게 오른 종목(3개월 과열)이 단기(1개월) 눌릴 때가 "
            "추격매수보다 유리하다(백테스트상 단순 과열주 매수 대비 +20.8%p, '10년~ 연 16.6%).\n\n"
            "- **눌림목 맵**: 가로=3개월 수익률(오른쪽일수록 강세), 세로=1개월 수익률(아래일수록 눌림). "
            "**파란 음영(눌림목 존)** = 3개월 강세 + 1개월 적정 눌림 → 가장 매력적인 구간.\n"
            "- **눌림목 점수(0~100)** = 3개월 강세 30% + 1개월 눌림 25% + 목표가 상승여력 18% + 실적 추정치 상향 15% + 거래대금 증가 12%.\n"
            "- **판정**: 🟢 강세 눌림목(매수 후보) / 🟡 관찰 / 🟠 눌림 없음(과열 추격 주의) / 🔴 추세 이탈(낙폭과대) / ⚪ 추세 미형성.\n"
            "- '1개월 눌림' 점수는 **−8% 부근에서 최고** — 너무 안 빠졌으면(추격 위험) 낮고, 너무 빠졌으면(추세 이탈) 낮음.\n\n"
            "⚠️ 원 리포트의 **수급(개인/기관/외인) 팩터는 미국 무료데이터로 불가**해 의도적으로 제외했습니다. 교육용 참고 지표.")

        with st.spinner("팩터 계산 중 (목표주가·실적추정치 포함, 종목당 ~1초)..."):
            rows = [compute_pullback(frames[s], s) for s in frames]
        rows.sort(key=lambda r: r["total"], reverse=True)

        st.plotly_chart(chart_pullback(rows), width="stretch")

        tbl = pd.DataFrame([{
            "종목": f"{name_of(r['sym'])} ({r['sym']})",
            "눌림목점수": round(r["total"], 1),
            "판정": r["verdict"],
            "3개월": f"{r['ret_3m']:+.1f}%",
            "1개월": f"{r['ret_1m']:+.1f}%",
            "거래대금증가(1M/3M)": f"{r['turnover_chg']:+.1f}%",
            "목표가상승여력": f"{r['tp_up']:+.1f}%" if r["tp_up"] is not None else "-",
            "실적추정치(30일)": f"{r['eps_rev']:+.1f}%" if r["eps_rev"] is not None else "-",
        } for r in rows])
        st.dataframe(tbl, width="stretch", hide_index=True)

        st.caption("📚 출처: 하나증권 이경수, 실전 퀀트(Quant MP) 2026.05.27 「모멘텀 발산기, 과열주 눌림목 전략」. "
                   "원 전략은 KRX300 롱-숏·월간 리밸런싱 기준이며, 본 페이지는 가격·목표주가·실적추정치 팩터만 재현한 단순화 버전입니다.")
