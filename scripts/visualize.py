"""
=============================================================================
AI Quant Trading Platform — 통합 시각화 대시보드
=============================================================================
실행:
    python3 scripts/visualize.py

결과물 (브라우저에서 자동으로 열림):
    charts/01_candlestick.html      ← 캔들 차트 + 매매 신호
    charts/02_equity_curve.html     ← 백테스팅 수익 곡선
    charts/03_3d_frontier.html      ← 3D 효율적 프론티어
    charts/04_3d_param_surface.html ← 3D 전략 파라미터 최적화
    charts/05_3d_vol_surface.html   ← 3D 변동성 서피스
    charts/06_correlation.html      ← 상관관계 히트맵
    charts/07_dashboard.html        ← 전체 통합 대시보드
=============================================================================
"""

from __future__ import annotations

import asyncio
import os
import sys
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio

# 결과 저장 폴더
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "charts")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 퀀트 전용 색상 팔레트
COLORS = {
    "bg":        "#0d1117",
    "panel":     "#161b22",
    "green":     "#00d4aa",
    "red":       "#ff4757",
    "blue":      "#4ecdc4",
    "gold":      "#ffd700",
    "purple":    "#a78bfa",
    "text":      "#e6edf3",
    "grid":      "#21262d",
}

def hex_to_rgba(hex_color: str, alpha: float = 0.15) -> str:
    """hex 색상을 rgba() 문자열로 변환"""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


BASE_LAYOUT = dict(
    paper_bgcolor=COLORS["bg"],
    plot_bgcolor=COLORS["panel"],
    font=dict(color=COLORS["text"], family="monospace"),
    xaxis=dict(gridcolor=COLORS["grid"], zerolinecolor=COLORS["grid"]),
    yaxis=dict(gridcolor=COLORS["grid"], zerolinecolor=COLORS["grid"]),
)


# ─────────────────────────────────────────────────────────────────────────────
# 데이터 수집
# ─────────────────────────────────────────────────────────────────────────────

async def fetch_data():
    """Yahoo Finance에서 여러 종목 데이터 수집"""
    from app.data.collectors.yahoo_collector import YahooFinanceCollector
    from app.core.types import Timeframe

    collector = YahooFinanceCollector()
    start = datetime(2022, 1, 1)

    print("  📡 데이터 수집 중...")
    symbols = {
        "GOOGL":  "구글",
        "AAPL":   "애플",
        "MSFT":   "마이크로소프트",
        "NVDA":   "엔비디아",
        "005930.KS": "삼성전자",
    }

    frames = {}
    for sym, name in symbols.items():
        try:
            frame = await collector.fetch_ohlcv(sym, Timeframe.ONE_DAY, start)
            frames[sym] = frame
            print(f"     ✅ {name} ({sym}): {frame.n_bars}개 봉")
        except Exception as e:
            print(f"     ⚠️  {name} 실패: {e}")

    return frames


# ─────────────────────────────────────────────────────────────────────────────
# 1. 캔들스틱 + 매매 신호
# ─────────────────────────────────────────────────────────────────────────────

def chart_candlestick(frame, symbol: str = "AAPL") -> go.Figure:
    """캔들차트 + 이동평균 + RSI + 매수/매도 신호"""

    df = frame.data.copy()
    df.index = pd.to_datetime(df.index)

    # 이동평균
    df["ma20"]  = df["close"].rolling(20).mean()
    df["ma50"]  = df["close"].rolling(50).mean()
    df["ma200"] = df["close"].rolling(200).mean()

    # 볼린저 밴드
    df["bb_mid"]   = df["close"].rolling(20).mean()
    df["bb_std"]   = df["close"].rolling(20).std()
    df["bb_upper"] = df["bb_mid"] + 2 * df["bb_std"]
    df["bb_lower"] = df["bb_mid"] - 2 * df["bb_std"]

    # RSI
    delta  = df["close"].diff()
    gain   = delta.clip(lower=0).rolling(14).mean()
    loss   = (-delta.clip(upper=0)).rolling(14).mean()
    rs     = gain / loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))

    # 매수/매도 신호 (MA 크로스)
    df["signal"] = 0
    df.loc[(df["ma20"] > df["ma50"]) & (df["ma20"].shift(1) <= df["ma50"].shift(1)), "signal"] = 1
    df.loc[(df["ma20"] < df["ma50"]) & (df["ma20"].shift(1) >= df["ma50"].shift(1)), "signal"] = -1

    buy_sig  = df[df["signal"] == 1]
    sell_sig = df[df["signal"] == -1]

    # 최근 300일만 표시
    df = df.tail(300)
    buy_sig  = buy_sig[buy_sig.index >= df.index[0]]
    sell_sig = sell_sig[sell_sig.index >= df.index[0]]

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=[0.6, 0.2, 0.2],
        vertical_spacing=0.03,
        subplot_titles=["가격 차트", "거래량", "RSI (14)"],
    )

    # 캔들
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["open"], high=df["high"],
        low=df["low"], close=df["close"],
        increasing_line_color=COLORS["green"],
        decreasing_line_color=COLORS["red"],
        name="가격",
    ), row=1, col=1)

    # 볼린저 밴드
    fig.add_trace(go.Scatter(
        x=df.index, y=df["bb_upper"], line=dict(color="rgba(255,215,0,0.3)", width=1),
        name="BB 상단", showlegend=False,
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=df["bb_lower"], line=dict(color="rgba(255,215,0,0.3)", width=1),
        fill="tonexty", fillcolor="rgba(255,215,0,0.05)",
        name="BB 밴드", showlegend=False,
    ), row=1, col=1)

    # 이동평균
    for ma, color, label in [
        ("ma20", COLORS["blue"], "MA 20"),
        ("ma50", COLORS["gold"], "MA 50"),
        ("ma200", COLORS["purple"], "MA 200"),
    ]:
        fig.add_trace(go.Scatter(
            x=df.index, y=df[ma],
            line=dict(color=color, width=1.2),
            name=label,
        ), row=1, col=1)

    # 매수 신호
    fig.add_trace(go.Scatter(
        x=buy_sig.index, y=buy_sig["low"] * 0.99,
        mode="markers", marker=dict(symbol="triangle-up", size=12, color=COLORS["green"]),
        name="🟢 매수 신호",
    ), row=1, col=1)

    # 매도 신호
    fig.add_trace(go.Scatter(
        x=sell_sig.index, y=sell_sig["high"] * 1.01,
        mode="markers", marker=dict(symbol="triangle-down", size=12, color=COLORS["red"]),
        name="🔴 매도 신호",
    ), row=1, col=1)

    # 거래량
    colors = [COLORS["green"] if c >= o else COLORS["red"]
              for c, o in zip(df["close"], df["open"])]
    fig.add_trace(go.Bar(
        x=df.index, y=df["volume"], marker_color=colors,
        name="거래량", showlegend=False,
    ), row=2, col=1)

    # RSI
    fig.add_trace(go.Scatter(
        x=df.index, y=df["rsi"],
        line=dict(color=COLORS["purple"], width=1.5),
        name="RSI", showlegend=False,
    ), row=3, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color=COLORS["red"],   opacity=0.5, row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color=COLORS["green"], opacity=0.5, row=3, col=1)

    fig.update_layout(
        **BASE_LAYOUT,
        title=dict(text=f"📈 {symbol} — 캔들차트 + 매매 신호", font=dict(size=18, color=COLORS["gold"])),
        height=800,
        xaxis_rangeslider_visible=False,
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=COLORS["grid"]),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 2. 백테스팅 수익 곡선
# ─────────────────────────────────────────────────────────────────────────────

def chart_equity_curve(frames: dict) -> go.Figure:
    """여러 전략/종목의 수익 곡선 비교"""

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.7, 0.3],
        vertical_spacing=0.05,
        subplot_titles=["누적 수익률 (%)", "낙폭 (Drawdown %)"],
    )

    palette = [COLORS["green"], COLORS["blue"], COLORS["gold"], COLORS["purple"], COLORS["red"]]

    for i, (sym, frame) in enumerate(frames.items()):
        df   = frame.data.copy()
        ret  = df["close"].pct_change().fillna(0)
        cum  = (1 + ret).cumprod()
        peak = cum.cummax()
        dd   = (cum - peak) / peak * 100
        pct  = (cum - 1) * 100

        color = palette[i % len(palette)]
        label = sym.replace(".KS", " (KRX)")

        fig.add_trace(go.Scatter(
            x=df.index, y=pct,
            line=dict(color=color, width=2),
            name=label,
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=df.index, y=dd,
            line=dict(color=color, width=1),
            fill="tozeroy", fillcolor=hex_to_rgba(color, 0.08),
            name=label, showlegend=False,
        ), row=2, col=1)

    # Buy & Hold 기준선
    fig.add_hline(y=0, line_dash="dot", line_color="white", opacity=0.3, row=1, col=1)

    fig.update_layout(
        **BASE_LAYOUT,
        title=dict(text="📊 종목별 누적 수익률 비교 (Buy & Hold)", font=dict(size=18, color=COLORS["gold"])),
        height=650,
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_yaxes(ticksuffix="%", row=1, col=1)
    fig.update_yaxes(ticksuffix="%", row=2, col=1)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 3. 3D 효율적 프론티어 (Markowitz)
# ─────────────────────────────────────────────────────────────────────────────

def chart_3d_efficient_frontier(frames: dict) -> go.Figure:
    """
    3D 효율적 프론티어:
      X = 포트폴리오 변동성 (위험)
      Y = 포트폴리오 기대 수익률
      Z = 샤프 비율
      색상 = 샤프 비율
    """
    # 수익률 행렬 구성 (타임존 제거 → 날짜 기준으로 정렬)
    returns_dict = {}
    for sym, frame in frames.items():
        ret = frame.data["close"].pct_change().dropna()
        ret.index = pd.to_datetime(ret.index).tz_localize(None).normalize()
        ret = ret.groupby(ret.index).last()   # 중복 날짜 제거
        returns_dict[sym] = ret

    ret_df = pd.DataFrame(returns_dict).dropna()
    mu  = ret_df.mean().values * 252          # 연간 기대수익률
    cov = ret_df.cov().values * 252           # 연간 공분산
    n   = len(mu)

    # 몬테카를로 시뮬레이션으로 랜덤 포트폴리오 생성
    np.random.seed(42)
    N_SIM  = 8000
    vols, rets, sharpes, weights_list = [], [], [], []

    for _ in range(N_SIM):
        w = np.random.dirichlet(np.ones(n))         # 합이 1인 랜덤 비중
        r = float(w @ mu)
        v = float(np.sqrt(w @ cov @ w))
        s = (r - 0.02) / v if v > 0 else 0          # 무위험 이자율 2%
        vols.append(v * 100)
        rets.append(r * 100)
        sharpes.append(s)
        weights_list.append(w)

    vols    = np.array(vols)
    rets    = np.array(rets)
    sharpes = np.array(sharpes)

    # 효율적 프론티어 근사 (구간별 최대 수익)
    vol_bins  = np.linspace(vols.min(), vols.max(), 60)
    frontier_v, frontier_r = [], []
    for lo, hi in zip(vol_bins[:-1], vol_bins[1:]):
        mask = (vols >= lo) & (vols < hi)
        if mask.sum() > 0:
            best = rets[mask].max()
            frontier_v.append((lo + hi) / 2)
            frontier_r.append(best)

    # 최고 샤프 포트폴리오
    best_idx = np.argmax(sharpes)
    best_sym_labels = list(frames.keys())

    hover_texts = [
        "<br>".join([f"{best_sym_labels[j]}: {weights_list[i][j]:.1%}"
                     for j in range(n)])
        for i in range(N_SIM)
    ]

    fig = go.Figure()

    # 랜덤 포트폴리오 구름
    fig.add_trace(go.Scatter3d(
        x=vols, y=rets, z=sharpes,
        mode="markers",
        marker=dict(
            size=2.5,
            color=sharpes,
            colorscale="Viridis",
            colorbar=dict(title="샤프 비율", thickness=15, x=1.02),
            opacity=0.7,
        ),
        text=hover_texts,
        hovertemplate="변동성: %{x:.1f}%<br>수익률: %{y:.1f}%<br>샤프: %{z:.2f}<br>%{text}<extra></extra>",
        name="포트폴리오 구름",
    ))

    # 효율적 프론티어 라인
    frontier_sharpe = [(r - 2) / v if v > 0 else 0
                       for v, r in zip(frontier_v, frontier_r)]
    fig.add_trace(go.Scatter3d(
        x=frontier_v, y=frontier_r, z=frontier_sharpe,
        mode="lines",
        line=dict(color=COLORS["gold"], width=5),
        name="효율적 프론티어",
    ))

    # 최고 샤프 포인트
    fig.add_trace(go.Scatter3d(
        x=[vols[best_idx]], y=[rets[best_idx]], z=[sharpes[best_idx]],
        mode="markers+text",
        marker=dict(size=10, color=COLORS["red"], symbol="diamond"),
        text=["★ 최적"],
        textfont=dict(color=COLORS["gold"], size=13),
        name=f"최적 샤프 ({sharpes[best_idx]:.2f})",
    ))

    fig.update_layout(
        **BASE_LAYOUT,
        title=dict(text="🎯 3D 효율적 프론티어 — Markowitz 포트폴리오 최적화", font=dict(size=17, color=COLORS["gold"])),
        scene=dict(
            xaxis=dict(title="변동성 (연간, %)", backgroundcolor=COLORS["panel"], gridcolor=COLORS["grid"]),
            yaxis=dict(title="기대 수익률 (연간, %)", backgroundcolor=COLORS["panel"], gridcolor=COLORS["grid"]),
            zaxis=dict(title="샤프 비율", backgroundcolor=COLORS["panel"], gridcolor=COLORS["grid"]),
            bgcolor=COLORS["bg"],
            camera=dict(eye=dict(x=1.8, y=-1.6, z=0.9)),
        ),
        height=750,
        legend=dict(bgcolor="rgba(0,0,0,0.5)"),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 4. 3D 전략 파라미터 최적화 서피스
# ─────────────────────────────────────────────────────────────────────────────

def chart_3d_param_surface(frame) -> go.Figure:
    """
    3D 전략 파라미터 최적화:
      X = 단기 이동평균 기간 (fast MA)
      Y = 장기 이동평균 기간 (slow MA)
      Z = 샤프 비율
    → 어떤 파라미터 조합이 가장 좋은지 3D로 확인
    """
    df     = frame.data.copy()
    closes = df["close"].values

    fast_range = range(5, 60, 5)     # 5, 10, 15, ..., 55
    slow_range = range(20, 200, 10)  # 20, 30, 40, ..., 190

    fast_vals, slow_vals, sharpe_vals = [], [], []

    for fast in fast_range:
        for slow in slow_range:
            if fast >= slow:
                continue

            # 이동평균 계산
            ma_fast = pd.Series(closes).rolling(fast).mean().values
            ma_slow = pd.Series(closes).rolling(slow).mean().values

            # 포지션 (MA 크로스)
            pos = np.where(ma_fast > ma_slow, 1.0, -1.0)
            pos = np.roll(pos, 1)   # 다음 날 진입 (미래 안 봄)
            pos[0] = 0

            # 수익률
            ret    = np.diff(closes) / closes[:-1]
            strat  = pos[:-1] * ret

            # 샤프 비율
            if strat.std() == 0:
                sharpe = 0.0
            else:
                sharpe = float((strat.mean() / strat.std()) * np.sqrt(252))

            fast_vals.append(fast)
            slow_vals.append(slow)
            sharpe_vals.append(np.clip(sharpe, -3, 4))

    fast_vals  = np.array(fast_vals)
    slow_vals  = np.array(slow_vals)
    sharpe_vals = np.array(sharpe_vals)

    # 최적 파라미터
    best_idx   = np.argmax(sharpe_vals)
    best_fast  = fast_vals[best_idx]
    best_slow  = slow_vals[best_idx]
    best_sharpe = sharpe_vals[best_idx]

    fig = go.Figure()

    # 3D 산점도 (서피스 근사)
    fig.add_trace(go.Scatter3d(
        x=fast_vals, y=slow_vals, z=sharpe_vals,
        mode="markers",
        marker=dict(
            size=5,
            color=sharpe_vals,
            colorscale=[
                [0.0, COLORS["red"]],
                [0.5, "#888"],
                [1.0, COLORS["green"]],
            ],
            colorbar=dict(title="샤프 비율", thickness=15, x=1.02),
            opacity=0.85,
        ),
        hovertemplate="Fast MA: %{x}<br>Slow MA: %{y}<br>샤프: %{z:.3f}<extra></extra>",
        name="파라미터 공간",
    ))

    # 최적 포인트 강조
    fig.add_trace(go.Scatter3d(
        x=[best_fast], y=[best_slow], z=[best_sharpe],
        mode="markers+text",
        marker=dict(size=14, color=COLORS["gold"], symbol="diamond", line=dict(color="white", width=2)),
        text=[f"★ Fast={best_fast} / Slow={best_slow}\nSharpe={best_sharpe:.2f}"],
        textfont=dict(color=COLORS["gold"], size=11),
        name=f"최적: Fast={best_fast}, Slow={best_slow}",
    ))

    fig.update_layout(
        **BASE_LAYOUT,
        title=dict(
            text=f"🔬 3D 전략 파라미터 최적화 — MA 크로스 전략<br>"
                 f"<sub>최적: Fast MA={best_fast}일, Slow MA={best_slow}일, Sharpe={best_sharpe:.2f}</sub>",
            font=dict(size=16, color=COLORS["gold"]),
        ),
        scene=dict(
            xaxis=dict(title="단기 MA 기간 (일)", backgroundcolor=COLORS["panel"], gridcolor=COLORS["grid"]),
            yaxis=dict(title="장기 MA 기간 (일)", backgroundcolor=COLORS["panel"], gridcolor=COLORS["grid"]),
            zaxis=dict(title="샤프 비율", backgroundcolor=COLORS["panel"], gridcolor=COLORS["grid"]),
            bgcolor=COLORS["bg"],
            camera=dict(eye=dict(x=2.0, y=-1.8, z=1.0)),
        ),
        height=750,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 5. 3D 변동성 서피스
# ─────────────────────────────────────────────────────────────────────────────

def chart_3d_vol_surface(frames: dict) -> go.Figure:
    """
    3D 실현 변동성 서피스 (go.Surface — 면으로 표현):
      X축 = 시간 인덱스
      Y축 = 종목 인덱스
      Z축 = 30일 롤링 연환산 변동성 (%)
      색상 = 변동성 크기 (낮음=초록, 높음=빨강)
    → 언제 어떤 종목이 얼마나 출렁였는지 지형도처럼 표현
    """
    # ── 1. 각 종목 변동성 계산 (날짜를 date 문자열로 정규화 → 시간대 문제 해결)
    vol_data = {}
    for sym, frame in frames.items():
        df  = frame.data.copy()
        ret = df["close"].pct_change()
        vol = ret.rolling(30).std() * np.sqrt(252) * 100
        vol = vol.dropna()
        # 타임존 제거 후 날짜 문자열로 통일
        vol.index = pd.to_datetime(vol.index).tz_localize(None).normalize()
        vol_data[sym] = vol

    # ── 2. 공통 날짜 범위 생성 (날짜 기준으로 교집합)
    common_dates = None
    for vol in vol_data.values():
        idx_set = set(vol.index.date)
        common_dates = idx_set if common_dates is None else common_dates & idx_set

    if not common_dates:
        return go.Figure()

    date_list = sorted(common_dates)
    # 너무 많으면 주 1개 샘플링 (5 거래일 간격)
    date_list = date_list[::5] if len(date_list) > 400 else date_list
    date_strs = [str(d) for d in date_list]

    labels = list(vol_data.keys())
    n_sym  = len(labels)
    n_time = len(date_list)

    # ── 3. Z 행렬 구성 (종목 × 시간)
    Z = np.full((n_sym, n_time), np.nan)
    for i, sym in enumerate(labels):
        vol = vol_data[sym]
        vol_by_date = {d.date(): v for d, v in vol.items()}
        for j, d in enumerate(date_list):
            Z[i, j] = vol_by_date.get(d, np.nan)

    # NaN 선형 보간
    for i in range(n_sym):
        row = pd.Series(Z[i])
        Z[i] = row.interpolate().ffill().bfill().values

    # ── 4. 축 레이블
    x_ticks = list(range(n_time))
    step     = max(1, n_time // 8)
    tick_vals = list(range(0, n_time, step))
    tick_text = [date_strs[i][:7] for i in tick_vals]   # "YYYY-MM"

    fig = go.Figure()

    # ── 5. Surface 트레이스
    fig.add_trace(go.Surface(
        x=x_ticks,
        y=list(range(n_sym)),
        z=Z,
        colorscale=[
            [0.0,  "#00d4aa"],   # 낮은 변동성 = 초록
            [0.4,  "#ffd700"],   # 중간 = 금색
            [0.7,  "#ff8c00"],   # 높음 = 주황
            [1.0,  "#ff4757"],   # 매우 높음 = 빨강
        ],
        colorbar=dict(
            title=dict(text="변동성 (%)", side="right"),
            thickness=16, x=1.02,
        ),
        opacity=0.92,
        contours=dict(
            z=dict(show=True, usecolormap=True, highlightcolor="white", project_z=True)
        ),
        hovertemplate="시간: %{x}<br>종목: %{y}<br>변동성: %{z:.1f}%<extra></extra>",
        name="변동성 서피스",
    ))

    # ── 6. 종목별 최고 변동성 피크 마커
    palette = [COLORS["green"], COLORS["blue"], COLORS["gold"], COLORS["purple"], COLORS["red"]]
    for i, sym in enumerate(labels):
        peak_j = int(np.argmax(Z[i]))
        fig.add_trace(go.Scatter3d(
            x=[peak_j], y=[i], z=[float(Z[i, peak_j])],
            mode="markers+text",
            marker=dict(size=8, color=palette[i % len(palette)],
                        symbol="diamond", line=dict(color="white", width=1)),
            text=[f"▲{Z[i, peak_j]:.0f}%"],
            textfont=dict(color="white", size=10),
            name=sym.replace(".KS", "(KRX)"),
        ))

    fig.update_layout(
        **BASE_LAYOUT,
        title=dict(
            text="🌊 3D 실현 변동성 서피스 — 종목별 리스크 지형도",
            font=dict(size=17, color=COLORS["gold"]),
        ),
        scene=dict(
            xaxis=dict(
                title="시간",
                tickvals=tick_vals, ticktext=tick_text,
                backgroundcolor=COLORS["panel"], gridcolor=COLORS["grid"],
            ),
            yaxis=dict(
                title="종목",
                tickvals=list(range(n_sym)),
                ticktext=[s.replace(".KS", "(KRX)") for s in labels],
                backgroundcolor=COLORS["panel"], gridcolor=COLORS["grid"],
            ),
            zaxis=dict(
                title="연환산 변동성 (%)",
                backgroundcolor=COLORS["panel"], gridcolor=COLORS["grid"],
            ),
            bgcolor=COLORS["bg"],
            camera=dict(eye=dict(x=2.0, y=-2.2, z=1.3)),
        ),
        height=750,
        legend=dict(bgcolor="rgba(0,0,0,0.5)"),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 6. 상관관계 히트맵
# ─────────────────────────────────────────────────────────────────────────────

def chart_correlation(frames: dict) -> go.Figure:
    """수익률 상관관계 히트맵 + 분포"""

    returns_dict = {}
    for sym, frame in frames.items():
        label = sym.replace(".KS", "(KRX)")
        ret   = frame.data["close"].pct_change().dropna()
        ret.index = pd.to_datetime(ret.index).tz_localize(None).normalize()
        ret = ret.groupby(ret.index).last()
        returns_dict[label] = ret

    ret_df = pd.DataFrame(returns_dict).dropna()
    corr   = ret_df.corr()

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["수익률 상관관계 행렬", "수익률 분포 비교"],
        column_widths=[0.5, 0.5],
    )

    # 상관관계 히트맵
    fig.add_trace(go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.index.tolist(),
        colorscale=[
            [0.0, COLORS["red"]],
            [0.5, COLORS["bg"]],
            [1.0, COLORS["green"]],
        ],
        zmin=-1, zmax=1,
        text=np.round(corr.values, 2),
        texttemplate="%{text}",
        textfont=dict(size=12),
        colorbar=dict(title="상관계수", x=0.45, thickness=12),
        name="상관관계",
    ), row=1, col=1)

    # 수익률 분포 (KDE 근사)
    palette = [COLORS["green"], COLORS["blue"], COLORS["gold"], COLORS["purple"], COLORS["red"]]
    for i, (label, ret) in enumerate(returns_dict.items()):
        fig.add_trace(go.Histogram(
            x=ret * 100,
            name=label,
            opacity=0.65,
            nbinsx=60,
            marker_color=palette[i % len(palette)],
            histnorm="probability density",
        ), row=1, col=2)

    fig.update_layout(
        **BASE_LAYOUT,
        title=dict(text="🔗 종목 간 상관관계 & 수익률 분포", font=dict(size=17, color=COLORS["gold"])),
        height=550,
        barmode="overlay",
    )
    fig.update_xaxes(title_text="일별 수익률 (%)", row=1, col=2, ticksuffix="%")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 7. 통합 대시보드
# ─────────────────────────────────────────────────────────────────────────────

def chart_dashboard(frames: dict) -> go.Figure:
    """한 페이지에 핵심 지표를 모두 표시하는 통합 대시보드"""

    fig = make_subplots(
        rows=3, cols=3,
        subplot_titles=[
            "누적 수익률", "30일 롤링 변동성", "30일 롤링 상관관계 (vs AAPL)",
            "월별 수익률 히트맵 (AAPL)", "RSI 비교", "연간 수익률 막대",
            "MDD (최대 낙폭)", "거래량 변화", "수익률 분포",
        ],
        vertical_spacing=0.1,
        horizontal_spacing=0.07,
    )

    palette = [COLORS["green"], COLORS["blue"], COLORS["gold"], COLORS["purple"], COLORS["red"]]
    labels  = list(frames.keys())

    # 수익률 계산 (타임존 제거)
    ret_dict = {}
    for sym, frame in frames.items():
        ret = frame.data["close"].pct_change().fillna(0)
        ret.index = pd.to_datetime(ret.index).tz_localize(None).normalize()
        ret = ret.groupby(ret.index).last()
        ret_dict[sym] = ret

    ret_df = pd.DataFrame(ret_dict).dropna()

    # ── Row 1, Col 1: 누적 수익률
    for i, (sym, ret) in enumerate(ret_dict.items()):
        cum = (1 + ret).cumprod() - 1
        fig.add_trace(go.Scatter(
            x=ret.index, y=cum * 100,
            line=dict(color=palette[i % len(palette)], width=1.5),
            name=sym.replace(".KS", "(KRX)"),
        ), row=1, col=1)

    # ── Row 1, Col 2: 롤링 변동성
    for i, (sym, ret) in enumerate(ret_dict.items()):
        vol = ret.rolling(30).std() * np.sqrt(252) * 100
        fig.add_trace(go.Scatter(
            x=vol.index, y=vol,
            line=dict(color=palette[i % len(palette)], width=1.2),
            name=sym, showlegend=False,
        ), row=1, col=2)

    # ── Row 1, Col 3: 상관관계 vs AAPL
    base_sym = labels[0]
    for i, sym in enumerate(labels[1:], 1):
        roll_corr = ret_df[base_sym].rolling(30).corr(ret_df[sym])
        fig.add_trace(go.Scatter(
            x=roll_corr.index, y=roll_corr,
            line=dict(color=palette[i % len(palette)], width=1.2),
            name=sym, showlegend=False,
        ), row=1, col=3)
    fig.add_hline(y=0, line_dash="dot", line_color="white", opacity=0.3, row=1, col=3)

    # ── Row 2, Col 1: 월별 수익률 히트맵 (AAPL)
    aapl_frame = list(frames.values())[0]
    df_m = aapl_frame.data.copy()
    df_m["year"]  = pd.to_datetime(df_m.index).year
    df_m["month"] = pd.to_datetime(df_m.index).month
    monthly = df_m.groupby(["year", "month"])["close"].last().pct_change() * 100
    pivot   = monthly.unstack(level="month").fillna(0)

    fig.add_trace(go.Heatmap(
        z=pivot.values,
        x=["1월","2월","3월","4월","5월","6월","7월","8월","9월","10월","11월","12월"],
        y=pivot.index.tolist(),
        colorscale=[[0,"#ff4757"],[0.5,"#21262d"],[1,"#00d4aa"]],
        zmin=-15, zmax=15,
        text=np.round(pivot.values, 1),
        texttemplate="%{text}%",
        textfont=dict(size=9),
        showscale=False,
        name="월별 수익률",
    ), row=2, col=1)

    # ── Row 2, Col 2: RSI 비교
    for i, (sym, frame) in enumerate(frames.items()):
        c = frame.data["close"]
        d = c.diff()
        g = d.clip(lower=0).rolling(14).mean()
        l = (-d.clip(upper=0)).rolling(14).mean()
        rsi = 100 - (100 / (1 + g / l.replace(0, np.nan)))
        fig.add_trace(go.Scatter(
            x=rsi.index[-120:], y=rsi.iloc[-120:],
            line=dict(color=palette[i % len(palette)], width=1.2),
            name=sym, showlegend=False,
        ), row=2, col=2)
    fig.add_hline(y=70, line_dash="dash", line_color=COLORS["red"],   opacity=0.4, row=2, col=2)
    fig.add_hline(y=30, line_dash="dash", line_color=COLORS["green"], opacity=0.4, row=2, col=2)

    # ── Row 2, Col 3: 연간 수익률 막대
    for i, (sym, ret) in enumerate(ret_dict.items()):
        ann = ret.resample("YE").apply(lambda r: (1 + r).prod() - 1) * 100
        fig.add_trace(go.Bar(
            x=ann.index.year, y=ann.values,
            name=sym, marker_color=palette[i % len(palette)],
            showlegend=False, opacity=0.8,
        ), row=2, col=3)

    # ── Row 3, Col 1: MDD
    for i, (sym, ret) in enumerate(ret_dict.items()):
        cum  = (1 + ret).cumprod()
        peak = cum.cummax()
        dd   = (cum - peak) / peak * 100
        fig.add_trace(go.Scatter(
            x=dd.index, y=dd,
            fill="tozeroy",
            fillcolor=hex_to_rgba(palette[i % len(palette)], 0.13),
            line=dict(color=palette[i % len(palette)], width=1),
            name=sym, showlegend=False,
        ), row=3, col=1)

    # ── Row 3, Col 2: 거래량 변화율 (최초 대비)
    for i, (sym, frame) in enumerate(frames.items()):
        vol_norm = frame.data["volume"] / frame.data["volume"].rolling(30).mean()
        fig.add_trace(go.Scatter(
            x=vol_norm.index[-180:], y=vol_norm.iloc[-180:],
            line=dict(color=palette[i % len(palette)], width=1.2),
            name=sym, showlegend=False,
        ), row=3, col=2)
    fig.add_hline(y=1, line_dash="dot", line_color="white", opacity=0.3, row=3, col=2)

    # ── Row 3, Col 3: 수익률 분포
    for i, (sym, ret) in enumerate(ret_dict.items()):
        fig.add_trace(go.Histogram(
            x=ret * 100,
            name=sym, marker_color=palette[i % len(palette)],
            opacity=0.6, nbinsx=80, histnorm="probability density",
            showlegend=False,
        ), row=3, col=3)

    fig.update_layout(
        **BASE_LAYOUT,
        title=dict(text="📊 퀀트 트레이딩 통합 대시보드", font=dict(size=20, color=COLORS["gold"])),
        height=1100,
        barmode="overlay",
        showlegend=True,
        legend=dict(bgcolor="rgba(0,0,0,0.5)", x=1.01),
    )

    # Y축 포맷
    for r, c, suf in [(1,1,"%"), (1,2,"%"), (3,1,"%"), (3,2,"x")]:
        fig.update_yaxes(ticksuffix=suf, row=r, col=c)

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────────────────────────

def save_and_open(fig: go.Figure, filename: str, title: str):
    path = os.path.join(OUTPUT_DIR, filename)
    fig.write_html(path, include_plotlyjs="cdn", full_html=True)
    print(f"  💾 저장: charts/{filename}")


async def main():
    print("\n" + "=" * 65)
    print("  📊 AI Quant Trading — 시각화 생성 시작")
    print("=" * 65)

    # 데이터 수집
    frames = await fetch_data()

    if not frames:
        print("❌ 데이터 수집 실패. 인터넷 연결을 확인하세요.")
        return

    main_sym   = list(frames.keys())[0]
    main_frame = frames[main_sym]

    print("\n  🎨 차트 생성 중...")

    charts = [
        (chart_candlestick(main_frame, main_sym), "01_candlestick.html",    "캔들 차트"),
        (chart_equity_curve(frames),              "02_equity_curve.html",    "수익 곡선"),
        (chart_3d_efficient_frontier(frames),     "03_3d_frontier.html",     "3D 효율적 프론티어"),
        (chart_3d_param_surface(main_frame),      "04_3d_param_surface.html","3D 파라미터 최적화"),
        (chart_3d_vol_surface(frames),            "05_3d_vol_surface.html",  "3D 변동성 서피스"),
        (chart_correlation(frames),               "06_correlation.html",     "상관관계"),
        (chart_dashboard(frames),                 "07_dashboard.html",       "통합 대시보드"),
    ]

    for fig, fname, title in charts:
        print(f"  ⚙️  {title} ...", end=" ", flush=True)
        save_and_open(fig, fname, title)
        print("✅")

    print("\n" + "=" * 65)
    print("  ✅ 모든 차트 생성 완료!")
    print(f"  📁 저장 위치: {OUTPUT_DIR}")
    print("=" * 65)
    print("""
  브라우저에서 열기:
    open charts/07_dashboard.html         ← 통합 대시보드 (추천)
    open charts/03_3d_frontier.html       ← 3D 효율적 프론티어
    open charts/04_3d_param_surface.html  ← 3D 전략 파라미터 맵
    open charts/05_3d_vol_surface.html    ← 3D 변동성 서피스
""")

    # 자동으로 통합 대시보드 열기
    import subprocess
    dashboard_path = os.path.join(OUTPUT_DIR, "07_dashboard.html")
    subprocess.Popen(["open", dashboard_path])
    print("  🌐 브라우저에서 통합 대시보드를 자동으로 열고 있습니다...")


if __name__ == "__main__":
    asyncio.run(main())
