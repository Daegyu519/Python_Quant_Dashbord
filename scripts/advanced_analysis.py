


"""
=============================================================================
고급 투자 분석 — 기술적 + 가치투자 + ML 통합
=============================================================================
실행:
    python3 scripts/advanced_analysis.py

포함 기능:
  ① 가치투자 분석  (피오트로스키, 그레이엄, DCF, 알트만, 버핏)
  ② HMM 레짐 감지 (상승장/횡보장/하락장)
  ③ 현대 퀀트 ML  (XGBoost + LightGBM + RF + 메타 모델)
  ④ 종합 투자 판단 (기술적 + 가치 + ML 결합, 신뢰도 향상)
  ⑤ 시각화 업데이트
=============================================================================
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "charts")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 분석 대상 (미국 주식 위주 — 가치투자 데이터 풍부)
SYMBOLS = {
    "GOOGL": "구글",
    "AAPL":  "애플",
    "MSFT":  "마이크로소프트",
    "BRK-B": "버크셔 해서웨이",   # 버핏 회사 자체
    "JNJ":   "존슨앤존슨",
    "KO":    "코카콜라",           # 버핏 애정 종목
}

COLORS = {
    "bg": "#0d1117", "panel": "#161b22",
    "green": "#00d4aa", "red": "#ff4757",
    "gold": "#ffd700", "blue": "#4ecdc4",
    "purple": "#a78bfa", "text": "#e6edf3",
    "grid": "#21262d",
}


# ─────────────────────────────────────────────────────────────────────────────
# 메인 파이프라인
# ─────────────────────────────────────────────────────────────────────────────

async def run_pipeline():
    print("\n" + "=" * 65)
    print("  🔬 고급 퀀트 투자 분석 시작")
    print("=" * 65)

    # ── 1. 데이터 수집 ───────────────────────────────────────────────────────
    from app.data.collectors.yahoo_collector import YahooFinanceCollector
    from app.core.types import Timeframe

    collector = YahooFinanceCollector()
    start     = datetime(2020, 1, 1)   # 4년치 (ML 학습에 충분)

    print("\n  📡 가격 데이터 수집 중...")
    frames = {}
    for sym, name in SYMBOLS.items():
        try:
            f = await collector.fetch_ohlcv(sym, Timeframe.ONE_DAY, start)
            frames[sym] = f
            print(f"     ✅ {name} ({sym}): {f.n_bars}개 봉")
        except Exception as e:
            print(f"     ⚠️  {name} 실패: {e}")

    if not frames:
        print("❌ 데이터 없음")
        return

    # ── 2. 가치투자 분석 ─────────────────────────────────────────────────────
    print("\n  💰 가치투자 분석 중... (yfinance 기초 데이터)")
    from app.value_investing.screener import ValueInvestingScreener

    screener      = ValueInvestingScreener()
    value_results = {}

    for sym in list(frames.keys()):
        try:
            vs = await screener.analyze(sym)
            value_results[sym] = vs
            icon = "🟢" if "매수" in vs.rating else "🟡" if "보유" in vs.rating else "🔴"
            print(f"     {icon} {sym}: 종합 {vs.total_score:.0f}/100  "
                  f"피오트로스키 {vs.piotroski_score}/9  "
                  f"버핏 {vs.buffett_score}/10  "
                  f"DCF {vs.dcf_upside:+.0f}%")
        except Exception as e:
            print(f"     ⚠️  {sym} 가치분석 실패: {e}")
            value_results[sym] = None

    # ── 3. ML 학습 + 예측 ────────────────────────────────────────────────────
    print("\n  🤖 ML 모델 학습 + 예측 중...")
    from app.ml.models.improved_ensemble import ImprovedEnsembleModel

    ml_results = {}
    for sym, frame in frames.items():
        vs    = value_results.get(sym)
        vscore = vs.total_score if vs else 50.0
        try:
            t0    = time.time()
            model = ImprovedEnsembleModel()
            model.fit(frame)
            pred  = model.predict(frame, value_score=vscore)
            elapsed = time.time() - t0

            ml_results[sym] = pred
            sig_icon = "🔼" if pred.signal == "매수" else "🔽" if pred.signal == "매도" else "⏸️"
            print(f"     {sig_icon} {sym}: {pred.signal}  "
                  f"상승확률 {pred.up_probability:.1%}  "
                  f"신뢰도 {pred.confidence:.1%}  "
                  f"레짐: {pred.regime}  "
                  f"({elapsed:.0f}s)")
        except Exception as e:
            print(f"     ⚠️  {sym} ML 실패: {e}")
            ml_results[sym] = None

    # ── 4. 종합 판단 출력 ────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  📋 종합 투자 판단")
    print("=" * 65)

    final_scores = {}
    for sym, name in SYMBOLS.items():
        if sym not in frames:
            continue

        frame = frames[sym]
        vs    = value_results.get(sym)
        ml    = ml_results.get(sym)

        # 종합 점수 계산
        tech_score  = 50.0  # 기본값
        val_score   = vs.total_score if vs else 50.0
        ml_score    = (ml.up_probability * 100) if ml else 50.0
        confidence  = ml.confidence if ml else 0.5

        # 기술적 점수: MA + RSI 기반
        c = frame.data["close"].values
        rsi14 = 50.0
        if len(c) >= 15:
            d = np.diff(c)
            g = np.where(d > 0, d, 0)
            l = np.where(d < 0, -d, 0)
            ag = pd.Series(g).ewm(alpha=1/14).mean().values[-1]
            al = pd.Series(l).ewm(alpha=1/14).mean().values[-1]
            rsi14 = 100 - 100 / (1 + ag / (al + 1e-9))
        ma20  = c[-20:].mean() if len(c) >= 20 else c.mean()
        ma50  = c[-50:].mean() if len(c) >= 50 else c.mean()

        if rsi14 < 30:   tech_score = 75
        elif rsi14 > 70: tech_score = 25
        elif rsi14 < 45: tech_score = 60
        elif rsi14 > 55: tech_score = 40

        if c[-1] > ma20 > ma50:     tech_score += 10
        elif c[-1] < ma20 < ma50:   tech_score -= 10

        # 가중 종합 점수 (기술 20% + 가치 40% + ML 40%)
        final = tech_score * 0.20 + val_score * 0.40 + ml_score * 0.40
        final = float(np.clip(final, 0, 100))
        final_scores[sym] = final

        # 최종 판단
        if final >= 65:   verdict = "🟢 매수 추천"
        elif final >= 50: verdict = "🟡 보유/관망"
        elif final >= 35: verdict = "🟠 관망"
        else:             verdict = "🔴 회피"

        regime_str = ml.regime if ml else "N/A"
        print(f"""
  ┌─ {name} ({sym}) {'─'*(40-len(name)-len(sym))}
  │  현재가:   ${c[-1]:>9.2f}
  │  기술 점수: {tech_score:>5.1f}  RSI={rsi14:.0f}  {'상승추세' if c[-1]>ma50 else '하락추세'}
  │  가치 점수: {val_score:>5.1f}  피오트로스키 {vs.piotroski_score if vs else 'N/A'}/9  버핏 {vs.buffett_score if vs else 'N/A'}/10
  │  ML 예측:  {ml_score:>5.1f}  상승확률 {f"{ml.up_probability:.1%}" if ml else "학습실패"}  신뢰도 {confidence:.1%}
  │  레짐:     {regime_str}
  │  ──────────────────────────────────────────
  │  종합 점수: {final:.1f}/100
  └─ {verdict}  (신뢰도: {confidence:.1%})""")

        if vs and vs.warnings:
            for w in vs.warnings:
                print(f"     {w}")

    # ── 5. 시각화 업데이트 ───────────────────────────────────────────────────
    print("\n\n  🎨 차트 생성 중...")

    # 가치투자 레이더 차트
    fig_radar = make_radar_chart(value_results)
    path = os.path.join(OUTPUT_DIR, "08_value_radar.html")
    fig_radar.write_html(path, include_plotlyjs="cdn")
    print(f"     💾 charts/08_value_radar.html")

    # 종합 랭킹 차트
    fig_rank = make_ranking_chart(
        list(SYMBOLS.keys()), list(SYMBOLS.values()),
        value_results, ml_results, final_scores
    )
    path = os.path.join(OUTPUT_DIR, "09_ranking.html")
    fig_rank.write_html(path, include_plotlyjs="cdn")
    print(f"     💾 charts/09_ranking.html")

    # ML 신뢰도 비교
    fig_conf = make_confidence_chart(ml_results, value_results)
    path = os.path.join(OUTPUT_DIR, "10_confidence.html")
    fig_conf.write_html(path, include_plotlyjs="cdn")
    print(f"     💾 charts/10_confidence.html")

    print("\n" + "=" * 65)
    print("  ✅ 분석 완료!")
    print("=" * 65)
    print("""
  브라우저에서 열기:
    open charts/08_value_radar.html  ← 가치투자 레이더 차트
    open charts/09_ranking.html      ← 종합 투자 랭킹
    open charts/10_confidence.html   ← ML 신뢰도 비교
""")

    # 자동 열기
    import subprocess
    for f in ["09_ranking.html", "08_value_radar.html", "10_confidence.html"]:
        subprocess.Popen(["open", os.path.join(OUTPUT_DIR, f)])


# ─────────────────────────────────────────────────────────────────────────────
# 시각화
# ─────────────────────────────────────────────────────────────────────────────

def make_radar_chart(value_results: dict) -> go.Figure:
    """종목별 가치투자 레이더 차트"""
    categories = [
        "피오트로스키\n(재무건전성)",
        "버핏 체크\n(경쟁우위)",
        "DCF\n(내재가치)",
        "그레이엄\n(안전마진)",
        "알트만 Z\n(부도위험↓)",
    ]
    palette = [COLORS["green"], COLORS["blue"], COLORS["gold"],
               COLORS["purple"], COLORS["red"], "#ff8c00"]

    fig = go.Figure()

    for i, (sym, vs) in enumerate(value_results.items()):
        if vs is None:
            continue

        # 각 축을 0~100으로 정규화
        vals = [
            vs.piotroski_score / 9 * 100,
            vs.buffett_score / 10 * 100,
            float(np.clip(vs.dcf_upside + 50, 0, 100)),       # -50~+100 → 0~100
            float(np.clip(vs.graham_upside + 50, 0, 100)),
            float(np.clip(vs.altman_z / 5 * 100, 0, 100)),
        ]
        vals.append(vals[0])   # 닫기

        hex_c = palette[i % len(palette)].lstrip("#")
        r, g, b = int(hex_c[0:2], 16), int(hex_c[2:4], 16), int(hex_c[4:6], 16)
        fig.add_trace(go.Scatterpolar(
            r=vals,
            theta=categories + [categories[0]],
            fill="toself",
            fillcolor=f"rgba({r},{g},{b},0.18)",
            line=dict(color=palette[i % len(palette)], width=2),
            name=f"{sym} (종합 {vs.total_score:.0f}점)",
        ))

    fig.update_layout(
        paper_bgcolor=COLORS["bg"],
        plot_bgcolor=COLORS["panel"],
        font=dict(color=COLORS["text"], family="monospace"),
        polar=dict(
            bgcolor=COLORS["panel"],
            radialaxis=dict(
                visible=True, range=[0, 100],
                gridcolor=COLORS["grid"],
                tickfont=dict(size=9),
            ),
            angularaxis=dict(gridcolor=COLORS["grid"]),
        ),
        title=dict(
            text="🎯 가치투자 레이더 차트 — 종목별 5개 축 비교",
            font=dict(size=17, color=COLORS["gold"]),
        ),
        legend=dict(bgcolor="rgba(0,0,0,0.5)"),
        height=650,
    )
    return fig


def make_ranking_chart(symbols, names, value_results, ml_results, final_scores) -> go.Figure:
    """종합 랭킹 막대 차트"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "📊 종합 투자 점수 랭킹",
            "💰 가치 점수 vs ML 예측 확률",
            "🏆 피오트로스키 + 버핏 점수",
            "🌊 레짐별 현황",
        ],
        vertical_spacing=0.18,
        horizontal_spacing=0.12,
    )

    valid_syms  = [s for s in symbols if s in final_scores]
    valid_names = [names[symbols.index(s)] for s in valid_syms]

    # 정렬
    sort_idx   = np.argsort([final_scores[s] for s in valid_syms])[::-1]
    sorted_syms = [valid_syms[i] for i in sort_idx]
    sorted_nms  = [valid_names[i] for i in sort_idx]
    sorted_scores = [final_scores[s] for s in sorted_syms]

    colors = [COLORS["green"] if s >= 65 else COLORS["gold"] if s >= 50 else COLORS["red"]
              for s in sorted_scores]

    # ── Row1, Col1: 종합 랭킹
    fig.add_trace(go.Bar(
        x=sorted_nms, y=sorted_scores,
        marker_color=colors,
        text=[f"{s:.1f}" for s in sorted_scores],
        textposition="outside",
        textfont=dict(size=11),
        name="종합 점수",
    ), row=1, col=1)
    fig.add_hline(y=65, line_dash="dash", line_color=COLORS["green"], opacity=0.6, row=1, col=1)
    fig.add_hline(y=50, line_dash="dash", line_color=COLORS["gold"], opacity=0.6, row=1, col=1)

    # ── Row1, Col2: 가치 vs ML
    val_sc = [value_results[s].total_score if value_results.get(s) else 50 for s in sorted_syms]
    ml_sc  = [ml_results[s].up_probability * 100 if ml_results.get(s) else 50 for s in sorted_syms]

    fig.add_trace(go.Scatter(
        x=val_sc, y=ml_sc,
        mode="markers+text",
        marker=dict(size=14, color=sorted_scores, colorscale="RdYlGn",
                    cmin=0, cmax=100, line=dict(color="white", width=1)),
        text=sorted_syms,
        textposition="top center",
        textfont=dict(size=10),
        name="가치 vs ML",
        showlegend=False,
    ), row=1, col=2)
    fig.add_hline(y=50, line_dash="dot", line_color="white", opacity=0.3, row=1, col=2)
    fig.add_vline(x=50, line_dash="dot", line_color="white", opacity=0.3, row=1, col=2)

    # ── Row2, Col1: 피오트로스키 + 버핏
    pf_sc  = [value_results[s].piotroski_score if value_results.get(s) else 0 for s in sorted_syms]
    buf_sc = [value_results[s].buffett_score if value_results.get(s) else 0 for s in sorted_syms]

    fig.add_trace(go.Bar(
        x=sorted_nms, y=pf_sc,
        name="피오트로스키 /9",
        marker_color=COLORS["blue"], opacity=0.8,
    ), row=2, col=1)
    fig.add_trace(go.Bar(
        x=sorted_nms, y=buf_sc,
        name="버핏 체크 /10",
        marker_color=COLORS["gold"], opacity=0.8,
    ), row=2, col=1)

    # ── Row2, Col2: 레짐 현황
    regimes = {"상승장": 0, "횡보장": 0, "하락장": 0}
    for s in valid_syms:
        ml = ml_results.get(s)
        if ml:
            regimes[ml.regime] = regimes.get(ml.regime, 0) + 1

    fig.add_trace(go.Bar(
        x=list(regimes.keys()),
        y=list(regimes.values()),
        marker_color=[COLORS["green"], COLORS["gold"], COLORS["red"]],
        text=list(regimes.values()),
        textposition="outside",
        name="레짐 분포",
        showlegend=False,
    ), row=2, col=2)

    fig.update_layout(
        paper_bgcolor=COLORS["bg"],
        plot_bgcolor=COLORS["panel"],
        font=dict(color=COLORS["text"], family="monospace"),
        height=800,
        title=dict(text="📈 종합 투자 분석 랭킹 대시보드", font=dict(size=18, color=COLORS["gold"])),
        barmode="group",
        legend=dict(bgcolor="rgba(0,0,0,0.5)"),
    )
    for row in [1, 2]:
        for col in [1, 2]:
            fig.update_xaxes(gridcolor=COLORS["grid"], row=row, col=col)
            fig.update_yaxes(gridcolor=COLORS["grid"], row=row, col=col)

    fig.update_xaxes(title_text="가치 점수", row=1, col=2)
    fig.update_yaxes(title_text="ML 상승 확률 (%)", row=1, col=2)
    fig.update_yaxes(range=[0, 100], row=1, col=1)

    return fig


def make_confidence_chart(ml_results: dict, value_results: dict) -> go.Figure:
    """ML 신뢰도 vs 기존 방식 비교 + 레짐 확률"""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["🤖 ML 신뢰도 비교 (구 55% → 개선)", "🌊 레짐 확률 분포"],
        horizontal_spacing=0.12,
    )

    palette = [COLORS["green"], COLORS["blue"], COLORS["gold"],
               COLORS["purple"], COLORS["red"], "#ff8c00"]
    syms = [s for s in ml_results if ml_results[s] is not None]

    # ── 신뢰도 비교 (구 vs 신)
    old_conf = [0.55] * len(syms)  # 기존 고정값
    new_conf = [ml_results[s].confidence for s in syms]
    labels   = syms

    x = np.arange(len(syms))
    fig.add_trace(go.Bar(
        x=labels, y=[c * 100 for c in old_conf],
        name="기존 신뢰도 (고정 55%)",
        marker_color="rgba(100,100,100,0.6)",
    ), row=1, col=1)
    fig.add_trace(go.Bar(
        x=labels, y=[c * 100 for c in new_conf],
        name="업그레이드 신뢰도 (ML)",
        marker_color=[palette[i % len(palette)] for i in range(len(syms))],
    ), row=1, col=1)
    fig.add_hline(y=55, line_dash="dot", line_color="gray", opacity=0.7,
                  annotation_text="기존 55%", row=1, col=1)

    # ── 레짐 확률 스택 바
    regime_colors = {"상승장": COLORS["green"], "횡보장": COLORS["gold"], "하락장": COLORS["red"]}
    for regime, color in regime_colors.items():
        vals = []
        for s in syms:
            ml = ml_results.get(s)
            vals.append(ml.regime_proba.get(regime, 0) * 100 if ml else 33.3)
        fig.add_trace(go.Bar(
            x=syms, y=vals,
            name=regime,
            marker_color=color,
        ), row=1, col=2)

    fig.update_layout(
        paper_bgcolor=COLORS["bg"],
        plot_bgcolor=COLORS["panel"],
        font=dict(color=COLORS["text"], family="monospace"),
        height=500,
        title=dict(text="🔬 ML 신뢰도 & 시장 레짐 분석", font=dict(size=17, color=COLORS["gold"])),
        barmode="stack",
        legend=dict(bgcolor="rgba(0,0,0,0.5)"),
    )
    for row in [1]:
        for col in [1, 2]:
            fig.update_xaxes(gridcolor=COLORS["grid"], row=row, col=col)
            fig.update_yaxes(gridcolor=COLORS["grid"], ticksuffix="%", row=row, col=col)

    return fig


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    asyncio.run(run_pipeline())
