"""
=============================================================================
순수 금융 계산 함수 — Streamlit·Plotly 비의존 (단위 테스트 대상)
=============================================================================
대시보드(dashboard.py)에서 쓰던 순수 계산 로직을 모아 import·테스트 가능하게 분리.
⚠️ UI/페이지 구조 분리와는 별개의 '순수 계산 추출'이다. 여기에는 st.*, go.* 가
   절대 들어가지 않는다.

  rsi_wilder()    Wilder 평활 RSI(14) — TradingView/Yahoo 동일
  tech_score()    기술적 점수(0~100) + 현재 RSI
  ichimoku()      일목균형표 (전환·기준·선행스팬, 26봉 선행 연장)
  max_pain()      옵션 최대 고통 행사가
  model_forecast() 상승확률 → 확률 가중 기대수익률
=============================================================================
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def rsi_wilder(prices: np.ndarray, period: int = 14) -> float:
    """Wilder 평활(RMA) RSI 의 최신값. 데이터 부족 시 50.0(중립)."""
    c = np.asarray(prices, dtype=float)
    if len(c) < period + 1:
        return 50.0
    d = np.diff(c)
    g = np.where(d > 0, d, 0.0)
    l = np.where(d < 0, -d, 0.0)
    ag = pd.Series(g).ewm(alpha=1 / period, adjust=False).mean().values[-1]
    al = pd.Series(l).ewm(alpha=1 / period, adjust=False).mean().values[-1]
    if al == 0:
        return 100.0
    return float(100 - 100 / (1 + ag / al))


def tech_score(df: pd.DataFrame) -> tuple[float, float]:
    """기술적 점수(0~100)와 현재 RSI. RSI 과매수/과매도 + MA 배열 기반."""
    c = df["close"].values
    rsi14 = rsi_wilder(c, 14)
    ma20 = c[-20:].mean() if len(c) >= 20 else c.mean()
    ma50 = c[-50:].mean() if len(c) >= 50 else c.mean()
    ts = 50.0
    if rsi14 < 30:   ts = 75
    elif rsi14 > 70: ts = 25
    elif rsi14 < 45: ts = 60
    elif rsi14 > 55: ts = 40
    if c[-1] > ma20 > ma50:   ts += 10
    elif c[-1] < ma20 < ma50: ts -= 10
    return float(np.clip(ts, 0, 100)), float(rsi14)


def ichimoku(df: pd.DataFrame, shift: int = 26) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    """
    일목균형표(Ichimoku) 계산.

    전환선(9)/기준선(26)/선행스팬A·B(26봉 선행 이동)를 계산하고,
    선행스팬이 미래로 뻗도록 shift 봉 연장한 인덱스를 함께 반환한다.
    """
    h, l = df["high"], df["low"]
    tenkan = (h.rolling(9).max() + l.rolling(9).min()) / 2
    kijun = (h.rolling(26).max() + l.rolling(26).min()) / 2
    span_a = (tenkan + kijun) / 2
    span_b = (h.rolling(52).max() + l.rolling(52).min()) / 2

    if len(df.index) >= 2:
        step = pd.Series(df.index).diff().median()
    else:
        step = pd.Timedelta(days=1)
    future = pd.DatetimeIndex([df.index[-1] + step * (i + 1) for i in range(shift)])
    ext_index = df.index.append(future)

    out = pd.DataFrame(index=ext_index)
    out["tenkan"] = tenkan.reindex(ext_index)
    out["kijun"] = kijun.reindex(ext_index)
    out["span_a"] = span_a.reindex(ext_index).shift(shift)
    out["span_b"] = span_b.reindex(ext_index).shift(shift)
    return out, ext_index


def max_pain(calls: pd.DataFrame, puts: pd.DataFrame) -> float | None:
    """최대 고통(max pain) 행사가 — 옵션 매도자 총 지급액이 최소가 되는 가격."""
    strikes = sorted(set(calls["strike"]) | set(puts["strike"]))
    if not strikes:
        return None
    co = calls.set_index("strike")["openInterest"].fillna(0)
    po = puts.set_index("strike")["openInterest"].fillna(0)
    best_k, best_pay = None, None
    for k in strikes:
        call_pay = float((co[co.index < k] * (k - co.index[co.index < k])).sum())
        put_pay = float((po[po.index > k] * (po.index[po.index > k] - k)).sum())
        total = call_pay + put_pay
        if best_pay is None or total < best_pay:
            best_pay, best_k = total, k
    return best_k


def model_forecast(df: pd.DataFrame, up_prob: float,
                   horizon: int = 5) -> dict:
    """
    모델 상승확률(horizon일)을 과거 수익률 분포와 결합한 확률 가중 기대수익.

      E[r] = p(상승) × E[r | 과거 상승시] + p(하락) × E[r | 과거 하락시]
    """
    r = df["close"].pct_change(horizon).dropna()
    up_m = float(r[r > 0].mean()) if (r > 0).any() else 0.0
    dn_m = float(r[r <= 0].mean()) if (r <= 0).any() else 0.0
    exp_h = up_prob * up_m + (1 - up_prob) * dn_m
    exp_21 = (1 + exp_h) ** (21 / horizon) - 1
    return dict(exp5=float(exp_h), exp21=float(exp_21), up_m=up_m, dn_m=dn_m)
