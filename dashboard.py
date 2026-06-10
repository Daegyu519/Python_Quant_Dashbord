"""
=============================================================================
실시간 미국 주식 대시보드 (Streamlit)
=============================================================================
Docker / frontend 불필요. 파이썬 프로세스 하나로 localhost 에서 실시간 갱신.

실행:
    pip install streamlit                # 최초 1회
    streamlit run dashboard.py           # → http://localhost:8501 자동 열림

페이지:
    📈 실시간          현재가 · 등락률 · 캔들차트(MA/거래량/RSI), N초 자동 갱신
    📊 수익률·낙폭     종목별 누적 수익률 + 드로다운 비교
    🔗 상관관계        수익률 상관 히트맵 + 분포
    🎯 포트폴리오 3D   Markowitz 효율적 프론티어 (몬테카를로)
    🔬 전략 최적화 3D  MA 크로스 파라미터별 샤프 지형도
    🌊 변동성 서피스 3D 종목·시간별 실현 변동성
    💰 가치·ML 분석    가치투자 점수 + ML 예측 (버튼 실행, 종목당 ~6초)

각 페이지의 "📖 이 차트 읽는 법"을 펼치면 해석 가이드가 나옵니다.

⚠️ 무료 Yahoo 데이터는 약 15분 지연이며, 미국 장 시간(한국 밤~새벽)에만 가격이 움직입니다.
⚠️ 가치·ML 분석은 교육용 참고 지표이며 투자 조언이 아닙니다.
=============================================================================
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from datetime import datetime, time as dtime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # app 패키지 임포트용

# ─────────────────────────────────────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="실시간 주식 대시보드", page_icon="📈", layout="wide")

# 모바일 반응형 — 좁은 화면(≤640px)에서 가로 컬럼을 세로로 쌓아 찌부됨 방지
st.markdown(
    """
    <style>
    @media (max-width: 640px) {
        /* st.columns 가로 배치를 세로로 전환 */
        [data-testid="stHorizontalBlock"] {
            flex-direction: column !important;
            gap: 0.4rem !important;
        }
        [data-testid="stColumn"], [data-testid="column"] {
            width: 100% !important;
            flex: 1 1 100% !important;
            min-width: 100% !important;
        }
        /* 좌우 여백 축소로 가로 공간 확보 */
        .block-container {
            padding-left: 0.7rem !important;
            padding-right: 0.7rem !important;
            padding-top: 1rem !important;
        }
        /* 표·차트가 화면 밖으로 안 넘치게 */
        [data-testid="stDataFrame"], [data-testid="stPlotlyChart"] {
            overflow-x: auto !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

COLORS = {
    "bg": "#0d1117", "panel": "#161b22",
    "green": "#00d4aa", "red": "#ff4757",
    "gold": "#ffd700", "blue": "#4ecdc4",
    "purple": "#a78bfa", "text": "#e6edf3", "grid": "#21262d",
}
PALETTE = [COLORS["green"], COLORS["blue"], COLORS["gold"],
           COLORS["purple"], COLORS["red"], "#ff8c00"]

PRESET_TICKERS = {
    "AAPL": "애플", "MSFT": "마이크로소프트", "NVDA": "엔비디아",
    "GOOGL": "구글", "AMZN": "아마존", "TSLA": "테슬라",
    "META": "메타", "AMD": "AMD", "NFLX": "넷플릭스", "SPY": "S&P500 ETF",
}

ET = ZoneInfo("America/New_York")
KST = ZoneInfo("Asia/Seoul")

BASE_LAYOUT = dict(
    paper_bgcolor=COLORS["bg"], plot_bgcolor=COLORS["panel"],
    font=dict(color=COLORS["text"], family="monospace"),
)


def hex_to_rgba(hex_color: str, alpha: float = 0.15) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def name_of(sym: str) -> str:
    return PRESET_TICKERS.get(sym, sym)


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
    df = (tk.history(period="1d", interval="1m") if mode == "인트라데이 (1분봉)"
          else tk.history(period="6mo", interval="1d"))
    if df is None or df.empty:
        return None, None, None
    df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
    last_price = float(df["close"].iloc[-1])
    try:
        prev_close = float(tk.fast_info["previous_close"])
    except Exception:
        prev_close = None
    if not prev_close:
        prev_close = float(df["open"].iloc[0]) if len(df) else last_price
    return df, last_price, prev_close


@st.cache_data(show_spinner=False, ttl=300)
def fetch_daily(symbol: str) -> pd.DataFrame | None:
    df = yf.Ticker(symbol).history(period="2y", interval="1d")
    if df is None or df.empty:
        return None
    df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
    df.index = pd.to_datetime(df.index)
    return df


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


def tech_score(df: pd.DataFrame) -> tuple[float, float]:
    """기술적 점수(0~100)와 현재 RSI. advanced_analysis.py 로직과 동일."""
    c = df["close"].values
    rsi14 = 50.0
    if len(c) >= 15:
        d = np.diff(c)
        g = np.where(d > 0, d, 0.0)
        l = np.where(d < 0, -d, 0.0)
        ag = pd.Series(g).ewm(alpha=1 / 14).mean().values[-1]
        al = pd.Series(l).ewm(alpha=1 / 14).mean().values[-1]
        rsi14 = 100 - 100 / (1 + ag / (al + 1e-9))
    ma20 = c[-20:].mean() if len(c) >= 20 else c.mean()
    ma50 = c[-50:].mean() if len(c) >= 50 else c.mean()
    ts = 50.0
    if rsi14 < 30:   ts = 75
    elif rsi14 > 70: ts = 25
    elif rsi14 < 45: ts = 60
    elif rsi14 > 55: ts = 40
    if c[-1] > ma20 > ma50:   ts += 10
    elif c[-1] < ma20 < ma50: ts -= 10
    return float(ts), float(rsi14)


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

def chart_candlestick(df: pd.DataFrame, symbol: str, name: str) -> go.Figure:
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

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        row_heights=[0.6, 0.2, 0.2], vertical_spacing=0.04,
                        subplot_titles=["가격", "거래량", "RSI(14)"])
    # 볼린저 밴드 (캔들 뒤에 깔리도록 먼저 추가)
    fig.add_trace(go.Scatter(x=df.index, y=df["bb_up"],
        line=dict(color="rgba(167,139,250,0.4)", width=1),
        name="BB 상단", showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["bb_dn"],
        line=dict(color="rgba(167,139,250,0.4)", width=1),
        fill="tonexty", fillcolor="rgba(167,139,250,0.07)",
        name="볼린저밴드(20,±2σ)"), row=1, col=1)
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["open"], high=df["high"], low=df["low"], close=df["close"],
        increasing_line_color=COLORS["green"], decreasing_line_color=COLORS["red"], name="가격",
    ), row=1, col=1)
    for ma, color, label in [("ma20", COLORS["blue"], "MA20"), ("ma50", COLORS["gold"], "MA50")]:
        fig.add_trace(go.Scatter(x=df.index, y=df[ma], line=dict(color=color, width=1.2),
                                 name=label), row=1, col=1)
    vc = [COLORS["green"] if c >= o else COLORS["red"] for c, o in zip(df["close"], df["open"])]
    fig.add_trace(go.Bar(x=df.index, y=df["volume"], marker_color=vc,
                         name="거래량", showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["rsi"], line=dict(color=COLORS["purple"], width=1.5),
                             name="RSI", showlegend=False), row=3, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color=COLORS["red"], opacity=0.4, row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color=COLORS["green"], opacity=0.4, row=3, col=1)
    fig.update_layout(**BASE_LAYOUT,
        title=dict(text=f"{name} ({symbol})", font=dict(size=16, color=COLORS["gold"])),
        height=600, xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=60, b=10),
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
    fig.add_hline(y=0, line_dash="dot", line_color="white", opacity=0.3, row=1, col=1)
    fig.update_layout(**BASE_LAYOUT,
        title=dict(text="📊 종목별 누적 수익률 & 낙폭 (최근 2년)", font=dict(size=17, color=COLORS["gold"])),
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
        colorscale=[[0.0, COLORS["red"]], [0.5, COLORS["bg"]], [1.0, COLORS["green"]]],
        zmin=-1, zmax=1, text=np.round(corr.values, 2), texttemplate="%{text}",
        textfont=dict(size=12), colorbar=dict(title="상관계수", x=0.45, thickness=12)), row=1, col=1)
    for i, col in enumerate(ret_df.columns):
        fig.add_trace(go.Histogram(x=ret_df[col] * 100, name=col, opacity=0.65, nbinsx=60,
                                   marker_color=PALETTE[i % len(PALETTE)],
                                   histnorm="probability density"), row=1, col=2)
    fig.update_layout(**BASE_LAYOUT,
        title=dict(text="🔗 종목 간 상관관계 & 수익률 분포", font=dict(size=17, color=COLORS["gold"])),
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
        marker=dict(size=10, color=COLORS["red"], symbol="diamond"),
        text=["★ 최적"], textfont=dict(color=COLORS["gold"], size=13),
        name=f"최적 샤프 ({sharpes[bi]:.2f})"))
    fig.update_layout(**BASE_LAYOUT,
        title=dict(text="🎯 3D 효율적 프론티어 — Markowitz 포트폴리오 최적화",
                   font=dict(size=17, color=COLORS["gold"])),
        scene=dict(
            xaxis=dict(title="변동성(연,%)", backgroundcolor=COLORS["panel"], gridcolor=COLORS["grid"]),
            yaxis=dict(title="기대수익(연,%)", backgroundcolor=COLORS["panel"], gridcolor=COLORS["grid"]),
            zaxis=dict(title="샤프 비율", backgroundcolor=COLORS["panel"], gridcolor=COLORS["grid"]),
            bgcolor=COLORS["bg"], camera=dict(eye=dict(x=1.8, y=-1.6, z=0.9))),
        height=720)
    alloc = "  ·  ".join(f"{labels[j]} {weights[bi][j]:.0%}" for j in range(n))
    return fig, alloc


def chart_param_surface(df: pd.DataFrame, sym: str) -> go.Figure:
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
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=fv, y=sv, z=zv, mode="markers",
        marker=dict(size=5, color=zv,
                    colorscale=[[0.0, COLORS["red"]], [0.5, "#888"], [1.0, COLORS["green"]]],
                    colorbar=dict(title="샤프", thickness=15, x=1.02), opacity=0.85),
        hovertemplate="Fast:%{x}<br>Slow:%{y}<br>샤프:%{z:.3f}<extra></extra>", name="파라미터"))
    fig.add_trace(go.Scatter3d(x=[fv[bi]], y=[sv[bi]], z=[zv[bi]], mode="markers+text",
        marker=dict(size=14, color=COLORS["gold"], symbol="diamond", line=dict(color="white", width=2)),
        text=[f"★{fv[bi]}/{sv[bi]}"], textfont=dict(color=COLORS["gold"], size=11),
        name=f"최적 Fast={fv[bi]} Slow={sv[bi]}"))
    fig.update_layout(**BASE_LAYOUT,
        title=dict(text=f"🔬 3D MA 크로스 파라미터 최적화 — {name_of(sym)} ({sym})<br>"
                        f"<sub>최적: Fast={fv[bi]}일, Slow={sv[bi]}일, Sharpe={zv[bi]:.2f}</sub>",
                   font=dict(size=16, color=COLORS["gold"])),
        scene=dict(
            xaxis=dict(title="단기 MA(일)", backgroundcolor=COLORS["panel"], gridcolor=COLORS["grid"]),
            yaxis=dict(title="장기 MA(일)", backgroundcolor=COLORS["panel"], gridcolor=COLORS["grid"]),
            zaxis=dict(title="샤프 비율", backgroundcolor=COLORS["panel"], gridcolor=COLORS["grid"]),
            bgcolor=COLORS["bg"], camera=dict(eye=dict(x=2.0, y=-1.8, z=1.0))),
        height=720)
    return fig


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
    Z = np.full((n_sym, n_time), np.nan)
    for i, sym in enumerate(labels):
        by_date = {d.date(): v for d, v in vol_data[sym].items()}
        for j, d in enumerate(date_list):
            Z[i, j] = by_date.get(d, np.nan)
    for i in range(n_sym):
        Z[i] = pd.Series(Z[i]).interpolate().ffill().bfill().values
    step = max(1, n_time // 8)
    tick_vals = list(range(0, n_time, step))
    tick_text = [date_strs[i][:7] for i in tick_vals]
    fig = go.Figure(go.Surface(
        x=list(range(n_time)), y=list(range(n_sym)), z=Z,
        colorscale=[[0.0, "#00d4aa"], [0.4, "#ffd700"], [0.7, "#ff8c00"], [1.0, "#ff4757"]],
        colorbar=dict(title=dict(text="변동성(%)", side="right"), thickness=16, x=1.02),
        opacity=0.92,
        contours=dict(z=dict(show=True, usecolormap=True, highlightcolor="white", project_z=True)),
        hovertemplate="시간:%{x}<br>종목:%{y}<br>변동성:%{z:.1f}%<extra></extra>"))
    fig.update_layout(**BASE_LAYOUT,
        title=dict(text="🌊 3D 실현 변동성 서피스 — 종목별 리스크 지형도",
                   font=dict(size=17, color=COLORS["gold"])),
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
        title=dict(text="🎯 가치투자 레이더 (5개 축, 100=최우량)", font=dict(size=16, color=COLORS["gold"])),
        legend=dict(bgcolor="rgba(0,0,0,0.5)"), height=520)
    return fig


def chart_ranking(res: dict) -> go.Figure:
    syms = sorted(res.keys(), key=lambda s: res[s]["final"], reverse=True)
    finals = [res[s]["final"] for s in syms]
    colors = [COLORS["green"] if f >= 65 else COLORS["gold"] if f >= 50 else COLORS["red"] for f in finals]
    fig = make_subplots(rows=1, cols=2, column_widths=[0.55, 0.45],
                        subplot_titles=["📊 종합 투자 점수 랭킹", "💰 가치점수 vs ML 상승확률"])
    fig.add_trace(go.Bar(x=[name_of(s) for s in syms], y=finals, marker_color=colors,
        text=[f"{f:.0f}" for f in finals], textposition="outside", showlegend=False), row=1, col=1)
    fig.add_hline(y=65, line_dash="dash", line_color=COLORS["green"], opacity=0.5, row=1, col=1)
    fig.add_hline(y=50, line_dash="dash", line_color=COLORS["gold"], opacity=0.5, row=1, col=1)
    val_sc = [res[s]["vs"].total_score if res[s]["vs"] else 50 for s in syms]
    ml_sc = [res[s]["pred"].up_probability * 100 if res[s]["pred"] else 50 for s in syms]
    fig.add_trace(go.Scatter(x=val_sc, y=ml_sc, mode="markers+text",
        marker=dict(size=16, color=finals, colorscale="RdYlGn", cmin=0, cmax=100,
                    line=dict(color="white", width=1)),
        text=syms, textposition="top center", showlegend=False), row=1, col=2)
    fig.add_hline(y=50, line_dash="dot", line_color="white", opacity=0.3, row=1, col=2)
    fig.add_vline(x=50, line_dash="dot", line_color="white", opacity=0.3, row=1, col=2)
    fig.update_layout(**BASE_LAYOUT, height=520,
        title=dict(text="📈 종합 투자 랭킹", font=dict(size=16, color=COLORS["gold"])))
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
    for reg, c in {"상승장": COLORS["green"], "횡보장": COLORS["gold"], "하락장": COLORS["red"]}.items():
        fig.add_trace(go.Bar(x=[name_of(s) for s in syms],
            y=[res[s]["pred"].regime_proba.get(reg, 0) * 100 for s in syms],
            name=reg, marker_color=c), row=1, col=2)
    fig.update_layout(**BASE_LAYOUT, height=460, barmode="stack",
        title=dict(text="🔬 ML 신뢰도 & 시장 레짐", font=dict(size=16, color=COLORS["gold"])),
        legend=dict(bgcolor="rgba(0,0,0,0.5)"))
    fig.update_yaxes(ticksuffix="%", gridcolor=COLORS["grid"])
    return fig


def run_advanced(symbols: list[str], frames: dict) -> dict:
    """종목별 가치투자 + ML 분석 실행. 무거우므로 버튼으로만 호출."""
    from app.value_investing.screener import ValueInvestingScreener
    from app.ml.models.improved_ensemble import ImprovedEnsembleModel

    screener = ValueInvestingScreener()
    res = {}
    prog = st.progress(0.0, text="분석 준비 중...")
    for i, sym in enumerate(symbols):
        if sym not in frames:
            continue
        prog.progress(i / len(symbols), text=f"{name_of(sym)} 가치분석 중...")
        try:
            vs = run_async(screener.analyze(sym))
        except Exception:
            vs = None
        prog.progress((i + 0.5) / len(symbols), text=f"{name_of(sym)} ML 학습 중...")
        try:
            model = ImprovedEnsembleModel()
            model.fit(_Frame(frames[sym], sym))
            pred = model.predict(_Frame(frames[sym], sym),
                                 value_score=(vs.total_score if vs else 50.0))
        except Exception:
            pred = None
        ts, _ = tech_score(frames[sym])
        val = vs.total_score if vs else 50.0
        mlp = (pred.up_probability * 100) if pred else 50.0
        final = float(np.clip(ts * 0.20 + val * 0.40 + mlp * 0.40, 0, 100))

        analyst = fetch_analyst(sym)
        cur = float(frames[sym]["close"].iloc[-1])
        ann_vol = float(frames[sym]["close"].pct_change().dropna().std() * np.sqrt(252))
        move = cur * ann_vol * np.sqrt(21 / 252)   # 1개월(21거래일) ±1σ 예상 변동폭
        res[sym] = dict(vs=vs, pred=pred, tech=ts, final=final,
                        analyst=analyst, cur=cur, band=(cur - move, cur + move))
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
                  y0=-18, y1=-3, fillcolor="rgba(0,212,170,0.10)",
                  line=dict(color="rgba(0,212,170,0.5)", width=1))
    fig.add_annotation(x=10, y=-3, text="  ◀ 눌림목 존(강세+눌림)", showarrow=False,
                       font=dict(color=COLORS["green"], size=11), xanchor="left", yanchor="bottom")
    fig.add_hline(y=0, line_dash="dot", line_color="white", opacity=0.3)
    fig.add_vline(x=0, line_dash="dot", line_color="white", opacity=0.3)
    fig.add_trace(go.Scatter(
        x=[r["ret_3m"] for r in rows], y=[r["ret_1m"] for r in rows],
        mode="markers+text", text=[r["sym"] for r in rows], textposition="top center",
        marker=dict(size=18, color=[r["total"] for r in rows], colorscale="RdYlGn",
                    cmin=0, cmax=100, line=dict(color="white", width=1),
                    colorbar=dict(title="눌림목<br>점수", thickness=14)),
        hovertemplate="%{text}<br>3개월:%{x:.1f}%<br>1개월:%{y:.1f}%<extra></extra>"))
    fig.update_layout(**BASE_LAYOUT, height=560,
        title=dict(text="🎯 눌림목 맵 — 3개월 강세(가로) vs 1개월 눌림(세로)",
                   font=dict(size=16, color=COLORS["gold"])))
    fig.update_xaxes(title_text="3개월 수익률 (%) — 오른쪽=강세", ticksuffix="%", gridcolor=COLORS["grid"])
    fig.update_yaxes(title_text="1개월 수익률 (%) — 아래=눌림", ticksuffix="%", gridcolor=COLORS["grid"])
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────────────────────

st.title("📈 실시간 미국 주식 대시보드")

with st.sidebar:
    st.header("⚙️ 설정")
    page = st.radio("📑 페이지", [
        "📈 실시간", "📊 수익률·낙폭", "🔗 상관관계",
        "🎯 포트폴리오 3D", "🔬 전략 최적화 3D", "🌊 변동성 서피스 3D",
        "💰 가치·ML 분석", "🎯 눌림목 스코어",
    ])
    st.divider()
    st.subheader("종목 선택")
    picked = st.multiselect("프리셋에서 고르기", options=list(PRESET_TICKERS.keys()),
        default=["AAPL", "NVDA", "TSLA"], format_func=lambda s: f"{s} · {PRESET_TICKERS[s]}")
    custom = st.text_input("직접 입력 (쉼표로 구분)", placeholder="예: COST, ORCL, BRK-B",
        help="미국 티커 형식. 입력한 종목도 함께 표시됩니다.")
    extra = [t.strip().upper() for t in custom.split(",") if t.strip()]
    symbols = list(dict.fromkeys(picked + extra))

    if page == "📈 실시간":
        st.divider()
        mode = st.radio("차트 종류", ["인트라데이 (1분봉)", "일봉 (6개월)"], index=0)
        auto = st.toggle("자동 갱신 (실시간)", value=True)
        interval = st.slider("갱신 주기 (초)", 5, 60, 10, disabled=not auto)
    else:
        mode, auto, interval = "일봉 (6개월)", False, 10

    st.divider()
    st.caption(f"마지막 갱신: {datetime.now(KST):%Y-%m-%d %H:%M:%S} (한국)")

if not symbols:
    st.info("👈 사이드바에서 종목을 하나 이상 선택하거나 입력하세요.")
    st.stop()


# ── 페이지: 실시간 ────────────────────────────────────────────────────────────
if page == "📈 실시간":
    is_open, status_label = us_market_status()
    (st.success if is_open else st.warning)(
        f"{status_label}  ·  데이터는 약 15분 지연된 무료 Yahoo 데이터입니다."
        + ("" if is_open else "  장이 닫혀 있어 가격은 마지막 거래일 기준으로 고정 표시됩니다."))
    help_box(
        "- **캔들**: 초록=상승 봉, 빨강=하락 봉. 위/아래 꼬리는 장중 고가·저가.\n"
        "- **MA20 / MA50**: 20일·50일 이동평균(추세선). MA20이 MA50 위에 있으면 단기 상승 우위.\n"
        "- **볼린저밴드(보라 음영, 20일·±2σ)**: 가격이 상단에 닿으면 단기 과열, 하단에 닿으면 과매도 신호. "
        "밴드 폭이 좁아지면(스퀴즈) 곧 큰 변동이 올 수 있음.\n"
        "- **거래량**: 봉 색과 동일. 급등/급락 시 거래량이 함께 터지면 신뢰도↑.\n"
        "- **RSI(14)**: 70↑ 과매수(조정 가능), 30↓ 과매도(반등 가능), 50이 중립.\n"
        "- 상단 숫자(등락률)는 **전일 종가 대비** 변화입니다.")

    bust = int(time.time() // (interval if auto else 10))
    cols = st.columns(len(symbols))
    data_cache: dict[str, pd.DataFrame] = {}
    for col, sym in zip(cols, symbols):
        try:
            df, last, prev = fetch_intraday(sym, mode, bust)
            if df is None:
                col.metric(f"{name_of(sym)} ({sym})", "데이터 없음"); continue
            data_cache[sym] = df
            change = last - prev
            pct = (change / prev * 100) if prev else 0
            col.metric(f"{name_of(sym)} ({sym})", f"${last:,.2f}", f"{change:+.2f} ({pct:+.2f}%)")
        except Exception as e:
            col.metric(f"{name_of(sym)} ({sym})", "오류"); col.caption(str(e)[:60])

    st.divider()
    if data_cache:
        tabs = st.tabs([f"{name_of(s)} ({s})" for s in data_cache])
        for tab, (sym, df) in zip(tabs, data_cache.items()):
            tab.plotly_chart(chart_candlestick(df, sym, name_of(sym)), use_container_width=True)

    if auto:
        time.sleep(interval)
        st.rerun()


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

    if page == "📊 수익률·낙폭":
        help_box(
            "- **위 그래프(누적 수익률)**: 2년 전 100% 기준, 지금까지 몇 % 올랐는지. 선이 높을수록 많이 상승.\n"
            "- **아래 그래프(낙폭)**: 직전 고점 대비 얼마나 빠졌는지. 0에 붙어 있을수록 안정적,\n"
            "  깊게 파일수록 그 시기에 큰 손실을 견뎌야 했다는 뜻(최대낙폭=MDD).\n"
            "- 수익률이 비슷하면 **낙폭이 얕은 종목**이 심리적으로 보유하기 쉽습니다.")
        st.plotly_chart(chart_equity(frames), use_container_width=True)

    elif page == "🔗 상관관계":
        help_box(
            "- **왼쪽 히트맵**: 두 종목이 같이 움직이는 정도. +1(초록)=거의 똑같이, 0=무관, -1(빨강)=반대.\n"
            "- 분산투자는 **상관이 낮은(0에 가까운)** 종목을 섞을수록 효과가 큽니다. 전부 초록이면 사실상 한 종목.\n"
            "- **오른쪽 분포**: 일별 수익률의 퍼짐. 폭이 넓을수록 변동성이 큰(=위험한) 종목.")
        st.plotly_chart(chart_correlation(frames), use_container_width=True)

    elif page == "🎯 포트폴리오 3D":
        help_box(
            "- 점 하나 = 종목 비중을 무작위로 섞은 **포트폴리오 하나**. 5,000개를 시뮬레이션했습니다.\n"
            "- 축: **X=위험(변동성)**, **Y=기대수익**, **Z·색=샤프(위험 1단위당 수익)**. 색이 밝을수록 효율적.\n"
            "- **★ 빨간 다이아 = 샤프 최고 지점**(가장 효율적). 아래에 그 추천 비중이 표시됩니다.\n"
            "- 마우스로 드래그하면 3D 회전. 왼쪽 위로 갈수록 '적은 위험에 높은 수익'.")
        fig, alloc = chart_frontier(frames)
        st.plotly_chart(fig, use_container_width=True)
        st.success(f"★ 최적 샤프 포트폴리오 비중 →  {alloc}")

    elif page == "🔬 전략 최적화 3D":
        target = st.selectbox("분석할 종목", list(frames.keys()),
                              format_func=lambda s: f"{name_of(s)} ({s})")
        help_box(
            "- **MA 크로스 전략**(단기 이평이 장기 이평을 넘으면 매수)을, 기간 조합을 바꿔가며 백테스트한 결과.\n"
            "- 축: **X=단기 MA, Y=장기 MA, Z·색=샤프 비율**. 초록(높은 곳)일수록 과거 성과가 좋았던 조합.\n"
            "- **★ = 과거 데이터상 최적 조합**. 단, 과거 최적이 미래를 보장하진 않습니다(과최적화 주의).")
        st.plotly_chart(chart_param_surface(frames[target], target), use_container_width=True)

    elif page == "🌊 변동성 서피스 3D":
        help_box(
            "- **종목 × 시간**별 30일 실현 변동성(연환산)을 지형도로 표현.\n"
            "- 색: **초록=잔잔함(저위험), 빨강=출렁임(고위험)**. 높이도 변동성 크기.\n"
            "- 솟아오른 **봉우리 = 그 종목이 그 시기에 크게 흔들렸다**는 뜻(급락/급등 구간).\n"
            "- 드래그로 회전. 특정 종목 띠가 전반적으로 붉으면 평소 변동성이 큰 종목.")
        st.plotly_chart(chart_vol_surface(frames), use_container_width=True)

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
            st.subheader("💡 종목별 추천 & 목표가")
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
                        parts.append(f"애널 목표범위 ${a['targetLowPrice']:,.0f}~${a['targetHighPrice']:,.0f}")
                    if vs and vs.dcf_upside:
                        parts.append(f"DCF 적정가 ${cur * (1 + vs.dcf_upside / 100):,.0f} ({vs.dcf_upside:+.0f}%)")
                    lo, hi = r["band"]
                    parts.append(f"1개월 통계적 예상범위(±1σ, 약 68%) ${lo:,.0f}~${hi:,.0f}")
                    st.caption("  ·  ".join(parts))

            # 📋 요약 표
            st.subheader("📋 요약 표")
            rows = []
            for s in order:
                r = res[s]; vs = r["vs"]; p = r["pred"]; a = r.get("analyst") or {}
                cur = r["cur"]; tgt = a.get("targetMeanPrice")
                rows.append({
                    "종목": f"{name_of(s)} ({s})",
                    "종합점수": round(r["final"], 1),
                    "판단": verdict(r["final"]),
                    "애널의견": reco_kor(a.get("recommendationKey")),
                    "목표가": f"${tgt:,.0f}" if tgt else "-",
                    "목표상승": f"{(tgt - cur) / cur * 100:+.0f}%" if tgt else "-",
                    "DCF상승": f"{vs.dcf_upside:+.0f}%" if vs else "-",
                    "ML신호": p.signal if p else "-",
                    "상승확률": f"{p.up_probability:.0%}" if p else "-",
                    "신뢰도": f"{p.confidence:.0%}" if p else "-",
                    "레짐": p.regime if p else "-",
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            value_results = {s: res[s]["vs"] for s in res}
            c1, c2 = st.columns(2)
            c1.plotly_chart(chart_value_radar(value_results), use_container_width=True)
            c2.plotly_chart(chart_ranking(res), use_container_width=True)
            if any(res[s]["pred"] for s in res):
                st.plotly_chart(chart_confidence(res), use_container_width=True)

            warned = [(s, res[s]["vs"].warnings) for s in res
                      if res[s]["vs"] and res[s]["vs"].warnings]
            if warned:
                st.subheader("⚠️ 경고")
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
            "**초록 음영(눌림목 존)** = 3개월 강세 + 1개월 적정 눌림 → 가장 매력적인 구간.\n"
            "- **눌림목 점수(0~100)** = 3개월 강세 30% + 1개월 눌림 25% + 목표가 상승여력 18% + 실적 추정치 상향 15% + 거래대금 증가 12%.\n"
            "- **판정**: 🟢 강세 눌림목(매수 후보) / 🟡 관찰 / 🟠 눌림 없음(과열 추격 주의) / 🔴 추세 이탈(낙폭과대) / ⚪ 추세 미형성.\n"
            "- '1개월 눌림' 점수는 **−8% 부근에서 최고** — 너무 안 빠졌으면(추격 위험) 낮고, 너무 빠졌으면(추세 이탈) 낮음.\n\n"
            "⚠️ 원 리포트의 **수급(개인/기관/외인) 팩터는 미국 무료데이터로 불가**해 의도적으로 제외했습니다. 교육용 참고 지표.")

        with st.spinner("팩터 계산 중 (목표주가·실적추정치 포함, 종목당 ~1초)..."):
            rows = [compute_pullback(frames[s], s) for s in frames]
        rows.sort(key=lambda r: r["total"], reverse=True)

        st.plotly_chart(chart_pullback(rows), use_container_width=True)

        tbl = pd.DataFrame([{
            "종목": f"{name_of(r['sym'])} ({r['sym']})",
            "눌림목점수": round(r["total"], 1),
            "판정": r["verdict"],
            "3개월": f"{r['ret_3m']:+.1f}%",
            "1개월": f"{r['ret_1m']:+.1f}%",
            "거래대금증가(1M/3M)": f"{r['turnover_chg']:+.0f}%",
            "목표가상승여력": f"{r['tp_up']:+.0f}%" if r["tp_up"] is not None else "-",
            "실적추정치(30일)": f"{r['eps_rev']:+.1f}%" if r["eps_rev"] is not None else "-",
        } for r in rows])
        st.dataframe(tbl, use_container_width=True, hide_index=True)

        st.caption("📚 출처: 하나증권 이경수, 실전 퀀트(Quant MP) 2026.05.27 「모멘텀 발산기, 과열주 눌림목 전략」. "
                   "원 전략은 KRX300 롱-숏·월간 리밸런싱 기준이며, 본 페이지는 가격·목표주가·실적추정치 팩터만 재현한 단순화 버전입니다.")
