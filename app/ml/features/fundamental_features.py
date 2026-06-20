"""
=============================================================================
펀더멘털 피처 — ML 모델용 가치 지표 (PEG · 지속 ROE · FCF · 내재가치)
=============================================================================
yfinance 무료 데이터에서 다음을 추출해 ML 피처 사전(dict)으로 반환:

  peg_ratio          PEG (P/E ÷ 이익성장률). 1 미만이면 성장 대비 저평가.
  roe                최근 ROE (자기자본이익률, 소수).
  roe_mean_4y        최근 최대 4개 회계연도 평균 ROE — '지속' ROE.
  roe_consistency    ROE ≥ 15% 를 기록한 연도 비율 (0~1) — 버핏식 품질 체크.
  fcf_yield          잉여현금흐름 / 시가총액. 높을수록 현금창출력 대비 저렴.
  profit_margin      순이익률.
  earnings_growth    이익 성장률 (YoY).
  revenue_growth     매출 성장률 (YoY).
  debt_to_equity     부채비율 (D/E, %를 배수로 정규화).
  price_to_book      PBR.
  pe_compression     forward P/E ÷ trailing P/E. 1 미만 = 이익 증가 기대.

값이 없으면 키 자체를 생략(모델 쪽에서 0 처리). 모든 값은 이상치 클립.

사용:
    fund = fetch_fundamental_features("AAPL")
    model.fit(frame, fundamentals=fund)
    score = fundamental_score(fund)     # 0~100 종합 점수
=============================================================================
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import yfinance as yf

# (키, 클립 하한, 클립 상한) — RobustScaler 가 있어도 극단치는 미리 제거
_CLIPS: dict[str, tuple[float, float]] = {
    "peg_ratio":       (0.0, 5.0),
    "roe":             (-1.0, 1.5),
    "roe_mean_4y":     (-1.0, 1.5),
    "roe_consistency": (0.0, 1.0),
    "fcf_yield":       (-0.2, 0.3),
    "profit_margin":   (-1.0, 1.0),
    "earnings_growth": (-1.0, 3.0),
    "revenue_growth":  (-1.0, 3.0),
    "debt_to_equity":  (0.0, 10.0),
    "price_to_book":   (0.0, 30.0),
    "pe_compression":  (0.2, 3.0),
}


def _safe_float(x) -> Optional[float]:
    """숫자 변환 — None/NaN/inf 는 None."""
    try:
        v = float(x)
        if v != v or v in (float("inf"), float("-inf")):
            return None
        return v
    except (TypeError, ValueError):
        return None


def _clip(key: str, value: float) -> float:
    lo, hi = _CLIPS.get(key, (-1e9, 1e9))
    return float(np.clip(value, lo, hi))


def _sustained_roe(tk: "yf.Ticker") -> dict[str, float]:
    """재무제표에서 최근 최대 4개 연도의 ROE 시계열 → 평균·일관성."""
    out: dict[str, float] = {}
    try:
        inc = tk.income_stmt
        bs = tk.balance_sheet
        ni = inc.loc["Net Income"].dropna()
        eq = bs.loc["Stockholders Equity"].dropna()
        common = ni.index.intersection(eq.index)[:4]
        if len(common) == 0:
            return out
        roes = (ni[common] / eq[common]).replace([np.inf, -np.inf], np.nan).dropna()
        if len(roes) == 0:
            return out
        out["roe_mean_4y"] = _clip("roe_mean_4y", float(roes.mean()))
        out["roe_consistency"] = _clip("roe_consistency",
                                       float((roes >= 0.15).mean()))
    except Exception:
        pass
    return out


def fetch_fundamental_features(symbol: str) -> dict[str, float]:
    """
    종목 펀더멘털 피처 사전. 실패한 항목은 생략하고 가능한 것만 반환.
    네트워크 호출이 있으므로 호출부에서 캐싱(@st.cache_data 등) 권장.
    """
    out: dict[str, float] = {}
    try:
        tk = yf.Ticker(symbol)
        info = tk.info or {}
    except Exception:
        return out

    # ── info 기반 단일 지표 ──────────────────────────────────────────────
    peg = _safe_float(info.get("trailingPegRatio") or info.get("pegRatio"))
    if peg is not None and peg > 0:
        out["peg_ratio"] = _clip("peg_ratio", peg)

    roe = _safe_float(info.get("returnOnEquity"))
    if roe is not None:
        out["roe"] = _clip("roe", roe)

    fcf = _safe_float(info.get("freeCashflow"))
    mcap = _safe_float(info.get("marketCap"))
    if fcf is not None and mcap and mcap > 0:
        out["fcf_yield"] = _clip("fcf_yield", fcf / mcap)

    for key, alias in [("profit_margin", "profitMargins"),
                       ("earnings_growth", "earningsGrowth"),
                       ("revenue_growth", "revenueGrowth"),
                       ("price_to_book", "priceToBook")]:
        v = _safe_float(info.get(alias))
        if v is not None:
            out[key] = _clip(key, v)

    de = _safe_float(info.get("debtToEquity"))
    if de is not None:
        out["debt_to_equity"] = _clip("debt_to_equity", de / 100.0)  # % → 배수

    fpe = _safe_float(info.get("forwardPE"))
    tpe = _safe_float(info.get("trailingPE"))
    if fpe and tpe and tpe > 0 and fpe > 0:
        out["pe_compression"] = _clip("pe_compression", fpe / tpe)

    # ── 재무제표 기반: 지속 ROE ─────────────────────────────────────────
    out.update(_sustained_roe(tk))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 종합 펀더멘털 점수 (0~100) — 모델의 확률 틸트 및 화면 표시용
# ─────────────────────────────────────────────────────────────────────────────

def _lin(x: Optional[float], lo: float, hi: float) -> Optional[float]:
    """lo→0, hi→100 선형 매핑. 결측은 None 유지(평균에서 제외)."""
    if x is None:
        return None
    if hi == lo:
        return 50.0
    return float(np.clip((x - lo) / (hi - lo) * 100, 0, 100))


def fundamental_score(fund: dict[str, float]) -> float:
    """
    펀더멘털 종합 점수 (0~100, 50=중립).

    구성 (가중치):
      PEG 35%        — 낮을수록 좋음 (0.5 이하 만점, 3 이상 0점)
      지속 ROE 25%   — roe_mean_4y 와 roe_consistency 평균
      FCF 수익률 20% — 높을수록 좋음
      이익성장 10%   — earnings_growth
      내재가치 10%   — dcf_upside (가치 스크리너에서 주입, % 단위)

    값이 없는 항목은 가중치에서 제외하고 나머지로 재정규화.
    """
    peg = fund.get("peg_ratio")
    peg_s = _lin(-peg, -3.0, -0.5) if peg is not None else None  # 낮을수록 높은 점수

    roe_parts = [s for s in (
        _lin(fund.get("roe_mean_4y"), 0.0, 0.30),
        _lin(fund.get("roe_consistency"), 0.0, 1.0),
    ) if s is not None]
    roe_s = float(np.mean(roe_parts)) if roe_parts else None

    fcf_s = _lin(fund.get("fcf_yield"), -0.02, 0.08)
    grw_s = _lin(fund.get("earnings_growth"), -0.20, 0.40)
    dcf_s = _lin(fund.get("dcf_upside"), -30.0, 50.0)

    parts = [(peg_s, 0.35), (roe_s, 0.25), (fcf_s, 0.20), (grw_s, 0.10), (dcf_s, 0.10)]
    avail = [(s, w) for s, w in parts if s is not None]
    if not avail:
        return 50.0
    total_w = sum(w for _, w in avail)
    return float(np.clip(sum(s * w for s, w in avail) / total_w, 0, 100))
