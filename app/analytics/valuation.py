"""
=============================================================================
밸류에이션·재무비율 계산 — yfinance 원본 수치만 사용 (추측·날조 없음)
=============================================================================
원칙 (사용자 요구사항):
  - 모든 값은 Yahoo Finance 가 제공하는 '원본 숫자'에서만 계산한다.
  - 원본이 없으면 해당 지표는 None(=화면에서 N/A). 임의 추정/기본값 대입 금지.
  - 절사/반올림으로 정보를 버리지 않는다(표시는 충분한 유효숫자 유지).
  - DCF/NPV 는 모델 특성상 할인율·영구성장률 가정이 불가피하므로, 이를 '사용자가
    조정 가능한 명시적 입력'으로 노출한다(숨은 가정 없음). 그 외(FCF·주식수·성장률)
    는 전부 원본 재무수치이며, 하나라도 없으면 계산하지 않고 사유를 반환한다.
=============================================================================
"""

from __future__ import annotations

import math
from typing import Optional

import yfinance as yf


def _num(x) -> Optional[float]:
    """yfinance 원본값 → float 또는 None. NaN/Inf/문자열/0길이는 None 처리(추정 안 함)."""
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if math.isnan(v) or math.isinf(v):
        return None
    return v


def fetch_info(symbol: str) -> dict:
    """yfinance .info 원본 dict (실패 시 빈 dict). 네트워크 호출은 호출부에서 캐싱."""
    try:
        return yf.Ticker(symbol).info or {}
    except Exception:
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# 핵심 지표 (원본 그대로 또는 원본 구성요소로 계산)
# ─────────────────────────────────────────────────────────────────────────────

def compute_ratios(info: dict) -> dict[str, Optional[float]]:
    """
    원본 재무수치로 핵심 밸류에이션·재무비율 산출.

    값이 None 이면 원본에 해당 수치가 없다는 뜻(추정하지 않음).
    비율 단위 주석:
      - roe/roa/마진류·growth : 소수(0.25 = 25%)  → 표시 시 ×100
      - debt_to_equity        : Yahoo 원본은 퍼센트포인트(예: 150.3 = 150.3%)
      - per/pbr/psr/ev_*       : 배수(그대로)
    """
    g = info.get
    price = _num(g("currentPrice")) or _num(g("regularMarketPrice"))

    market_cap = _num(g("marketCap"))
    revenue = _num(g("totalRevenue"))
    fcf = _num(g("freeCashflow"))
    eps = _num(g("trailingEps"))
    bvps = _num(g("bookValue"))            # 주당순자산
    ebitda = _num(g("ebitda"))
    ev = _num(g("enterpriseValue"))

    # 원본 제공 비율 (Yahoo 가 원본 재무로 계산해 제공)
    per_t = _num(g("trailingPE"))
    per_f = _num(g("forwardPE"))
    pbr = _num(g("priceToBook"))
    psr = _num(g("priceToSalesTrailing12Months"))
    ev_ebitda = _num(g("enterpriseToEbitda"))
    ev_rev = _num(g("enterpriseToRevenue"))

    # 원본 구성요소로 직접 재계산 (계산 검증용 — 둘 다 있으면 일치해야 함)
    per_calc = (price / eps) if (price is not None and eps not in (None, 0)) else None
    pbr_calc = (price / bvps) if (price is not None and bvps not in (None, 0)) else None
    psr_calc = (market_cap / revenue) if (market_cap is not None and revenue not in (None, 0)) else None
    ev_ebitda_calc = (ev / ebitda) if (ev is not None and ebitda not in (None, 0)) else None
    fcf_yield = (fcf / market_cap) if (fcf is not None and market_cap not in (None, 0)) else None

    return {
        "price":            price,
        "market_cap":       market_cap,
        "enterprise_value": ev,
        "revenue":          revenue,
        "ebitda":           ebitda,
        "fcf":              fcf,
        "eps_trailing":     eps,
        "book_value_ps":    bvps,
        # 밸류에이션 배수 (원본 제공값 우선, 없으면 직접계산값)
        "per_trailing":     per_t if per_t is not None else per_calc,
        "per_forward":      per_f,
        "pbr":              pbr if pbr is not None else pbr_calc,
        "psr":              psr if psr is not None else psr_calc,
        "ev_ebitda":        ev_ebitda if ev_ebitda is not None else ev_ebitda_calc,
        "ev_revenue":       ev_rev,
        "fcf_yield":        fcf_yield,
        # 수익성·재무비율 (원본 소수)
        "roe":              _num(g("returnOnEquity")),
        "roa":              _num(g("returnOnAssets")),
        "gross_margin":     _num(g("grossMargins")),
        "operating_margin": _num(g("operatingMargins")),
        "net_margin":       _num(g("profitMargins")),
        "debt_to_equity":   _num(g("debtToEquity")),
        "current_ratio":    _num(g("currentRatio")),
        "quick_ratio":      _num(g("quickRatio")),
        # 성장·기타 (원본)
        "earnings_growth":  _num(g("earningsGrowth")),
        "revenue_growth":   _num(g("revenueGrowth")),
        "beta":             _num(g("beta")),
        "dividend_yield":   _num(g("dividendYield")),
        # 검산용 (원본 구성요소로 계산한 값)
        "_per_calc":        per_calc,
        "_pbr_calc":        pbr_calc,
        "_psr_calc":        psr_calc,
    }


# ─────────────────────────────────────────────────────────────────────────────
# DCF / NPV — 원본 FCF·주식수·성장률 + 명시적(사용자조정) 할인율·영구성장률
# ─────────────────────────────────────────────────────────────────────────────

def dcf_npv(
    info: dict,
    discount_rate: float = 0.10,
    terminal_growth: float = 0.025,
    years: int = 10,
    growth_override: Optional[float] = None,
) -> dict:
    """
    할인현금흐름(DCF)으로 내재가치/주 + NPV(기업가치) 계산.

    원본 사용:
      FCF        = info['freeCashflow']      (없으면 계산 불가 → ok=False)
      주식수      = info['sharesOutstanding']
      성장률 g    = growth_override 또는 info['earningsGrowth'] / ['revenueGrowth']
    명시적 가정(사용자 입력):
      discount_rate(r), terminal_growth(gt), years(N)

    내재가치/주 = Σ_{t=1..N} FCF_ps·(1+g)^t / (1+r)^t
                + [FCF_ps·(1+g)^N·(1+gt)/(r−gt)] / (1+r)^N   (영구가치 현가)
    NPV(총액)   = 내재가치/주 × 주식수

    반환: ok=True 시 intrinsic_ps·npv_total·upside·g_used·pv_explicit·pv_terminal·fcf_ps
          ok=False 시 reason (원본 결측 등) — 추정으로 메우지 않는다.
    """
    fcf = _num(info.get("freeCashflow"))
    shares = _num(info.get("sharesOutstanding"))
    price = _num(info.get("currentPrice")) or _num(info.get("regularMarketPrice"))

    if fcf is None or shares in (None, 0):
        return {"ok": False, "reason": "원본 FCF 또는 발행주식수가 없어 계산 불가"}

    g = growth_override
    if g is None:
        g = _num(info.get("earningsGrowth"))
    if g is None:
        g = _num(info.get("revenueGrowth"))
    if g is None:
        return {"ok": False, "reason": "원본 성장률(이익/매출)이 없어 추정하지 않음"}

    r, gt = float(discount_rate), float(terminal_growth)
    if r <= gt:
        return {"ok": False, "reason": "할인율이 영구성장률보다 커야 함(r > gt)"}

    fcf_ps = fcf / shares
    pv_explicit = sum(fcf_ps * (1 + g) ** t / (1 + r) ** t
                      for t in range(1, years + 1))
    terminal = fcf_ps * (1 + g) ** years * (1 + gt) / (r - gt)
    pv_terminal = terminal / (1 + r) ** years
    intrinsic_ps = pv_explicit + pv_terminal
    npv_total = intrinsic_ps * shares
    upside = ((intrinsic_ps - price) / price) if price not in (None, 0) else None

    return {
        "ok": True,
        "intrinsic_ps": intrinsic_ps,
        "npv_total": npv_total,
        "upside": upside,
        "g_used": g,
        "fcf_ps": fcf_ps,
        "pv_explicit": pv_explicit * shares,
        "pv_terminal": pv_terminal * shares,
        "shares": shares,
        "price": price,
    }
