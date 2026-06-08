"""
=============================================================================
가치투자 스크리너 (Value Investing Screener)
=============================================================================
워런 버핏 / 벤저민 그레이엄 스타일 가치투자 분석 모듈.

포함 지표:
  ① 피오트로스키 F-스코어  (재무 건전성 9개 기준, 0~9점)
  ② 그레이엄 넘버         (EPS × BPS 기반 적정가)
  ③ DCF 내재가치          (할인 현금흐름)
  ④ 알트만 Z-스코어       (부도 위험 지수)
  ⑤ 버핏 체크리스트       (경쟁우위·ROE·부채 등)
  ⑥ 종합 가치 점수        (0~100, 높을수록 저평가 우량주)
=============================================================================
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

from dataclasses import dataclass, field
from typing import Optional
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# 데이터 클래스
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FundamentalData:
    """yfinance에서 수집한 기업 기초 데이터"""
    symbol:              str
    name:                str    = "N/A"
    current_price:       float  = 0.0
    market_cap:          float  = 0.0

    # 가격 지표
    pe_trailing:         Optional[float] = None   # 후행 PER
    pe_forward:          Optional[float] = None   # 선행 PER
    pb_ratio:            Optional[float] = None   # PBR
    ps_ratio:            Optional[float] = None   # PSR
    ev_ebitda:           Optional[float] = None   # EV/EBITDA

    # 수익성
    eps:                 Optional[float] = None   # 주당순이익
    book_value:          Optional[float] = None   # 주당순자산
    roe:                 Optional[float] = None   # 자기자본이익률
    roa:                 Optional[float] = None   # 총자산이익률
    roic:                Optional[float] = None   # 투하자본이익률
    gross_margin:        Optional[float] = None   # 매출총이익률
    operating_margin:    Optional[float] = None   # 영업이익률
    net_margin:          Optional[float] = None   # 순이익률

    # 성장성
    revenue_growth:      Optional[float] = None   # 매출 성장률
    earnings_growth:     Optional[float] = None   # 이익 성장률

    # 재무 안정성
    debt_to_equity:      Optional[float] = None   # 부채비율
    current_ratio:       Optional[float] = None   # 유동비율
    quick_ratio:         Optional[float] = None   # 당좌비율
    interest_coverage:   Optional[float] = None   # 이자보상배율

    # 현금흐름
    free_cashflow:       Optional[float] = None   # 잉여현금흐름
    operating_cashflow:  Optional[float] = None   # 영업현금흐름

    # 배당
    dividend_yield:      Optional[float] = None   # 배당수익률
    payout_ratio:        Optional[float] = None   # 배당성향

    # 52주
    week52_high:         Optional[float] = None
    week52_low:          Optional[float] = None
    beta:                Optional[float] = None


@dataclass
class ValueScore:
    """종합 가치투자 분석 결과"""
    symbol:             str
    name:               str

    # 세부 점수
    piotroski_score:    int    = 0      # 0~9 (9 = 최우량)
    graham_number:      float  = 0.0   # 그레이엄 적정가
    graham_upside:      float  = 0.0   # 현재가 대비 상승여력 (%)
    dcf_value:          float  = 0.0   # DCF 내재가치
    dcf_upside:         float  = 0.0   # DCF 상승여력 (%)
    altman_z:           float  = 0.0   # Z < 1.81 위험, > 2.99 안전
    buffett_score:      int    = 0     # 0~10 버핏 체크리스트

    # 핵심 지표 요약
    pe:                 Optional[float] = None
    pb:                 Optional[float] = None
    roe:                Optional[float] = None
    debt_equity:        Optional[float] = None
    fcf_yield:          Optional[float] = None    # FCF / 시총 (%)

    # 종합
    total_score:        float  = 0.0   # 0~100
    rating:             str    = "N/A" # BUY / HOLD / AVOID
    summary:            str    = ""
    warnings:           list   = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# 메인 스크리너
# ─────────────────────────────────────────────────────────────────────────────

class ValueInvestingScreener:
    """
    현대 가치투자 스크리너.

    사용법:
        screener = ValueInvestingScreener()
        result   = await screener.analyze("AAPL")
        print(result.total_score, result.rating)
    """

    RISK_FREE_RATE   = 0.045   # 무위험 수익률 (미국채 4.5%)
    EQUITY_RISK_PREM = 0.055   # 주식 위험 프리미엄 (역사적 평균)
    TERMINAL_GROWTH  = 0.025   # 영구 성장률 (명목 GDP 수준)
    DCF_YEARS        = 10      # 예측 기간

    async def analyze(self, symbol: str) -> ValueScore:
        """단일 종목 종합 가치투자 분석"""
        fund = await self._fetch_fundamentals(symbol)
        return self._compute_score(fund)

    async def screen_multiple(self, symbols: list[str]) -> list[ValueScore]:
        """여러 종목 일괄 스크리닝"""
        import asyncio
        tasks = [self.analyze(sym) for sym in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return [r for r in results if isinstance(r, ValueScore)]

    # ── 데이터 수집 ──────────────────────────────────────────────────────────

    async def _fetch_fundamentals(self, symbol: str) -> FundamentalData:
        """yfinance에서 기업 기초 데이터 수집"""
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._fetch_sync, symbol)

    def _fetch_sync(self, symbol: str) -> FundamentalData:
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        info   = ticker.info or {}

        def g(key, default=None):
            v = info.get(key, default)
            return float(v) if v is not None and not (isinstance(v, float) and np.isnan(v)) else default

        return FundamentalData(
            symbol          = symbol,
            name            = info.get("longName", symbol),
            current_price   = g("currentPrice") or g("regularMarketPrice") or 0.0,
            market_cap      = g("marketCap") or 0.0,

            pe_trailing     = g("trailingPE"),
            pe_forward      = g("forwardPE"),
            pb_ratio        = g("priceToBook"),
            ps_ratio        = g("priceToSalesTrailing12Months"),
            ev_ebitda       = g("enterpriseToEbitda"),

            eps             = g("trailingEps"),
            book_value      = g("bookValue"),
            roe             = g("returnOnEquity"),
            roa             = g("returnOnAssets"),
            gross_margin    = g("grossMargins"),
            operating_margin= g("operatingMargins"),
            net_margin      = g("profitMargins"),

            revenue_growth  = g("revenueGrowth"),
            earnings_growth = g("earningsGrowth"),

            debt_to_equity  = g("debtToEquity"),
            current_ratio   = g("currentRatio"),
            quick_ratio     = g("quickRatio"),

            free_cashflow   = g("freeCashflow"),
            operating_cashflow = g("operatingCashflow"),

            dividend_yield  = g("dividendYield"),
            payout_ratio    = g("payoutRatio"),

            week52_high     = g("fiftyTwoWeekHigh"),
            week52_low      = g("fiftyTwoWeekLow"),
            beta            = g("beta"),
        )

    # ── 점수 계산 ─────────────────────────────────────────────────────────────

    def _compute_score(self, f: FundamentalData) -> ValueScore:
        score = ValueScore(symbol=f.symbol, name=f.name)
        warnings_list = []
        component_scores = []

        # ① 피오트로스키 F-스코어
        pf, pf_detail = self._piotroski(f)
        score.piotroski_score = pf

        # ② 그레이엄 넘버
        gn = self._graham_number(f)
        score.graham_number = gn
        if gn > 0 and f.current_price > 0:
            score.graham_upside = (gn - f.current_price) / f.current_price * 100

        # ③ DCF 내재가치
        dcf = self._dcf_value(f)
        score.dcf_value = dcf
        if dcf > 0 and f.current_price > 0:
            score.dcf_upside = (dcf - f.current_price) / f.current_price * 100

        # ④ 알트만 Z-스코어
        score.altman_z = self._altman_z(f)

        # ⑤ 버핏 체크리스트
        score.buffett_score, buf_detail = self._buffett_checklist(f)

        # 요약 지표
        score.pe          = f.pe_trailing
        score.pb          = f.pb_ratio
        score.roe         = f.roe
        score.debt_equity = f.debt_to_equity
        if f.free_cashflow and f.market_cap and f.market_cap > 0:
            score.fcf_yield = f.free_cashflow / f.market_cap * 100

        # ⑥ 종합 점수 (0~100)
        score.total_score, component_scores = self._total_score(f, score)

        # 경고 생성
        if f.pe_trailing and f.pe_trailing > 40:
            warnings_list.append(f"⚠️ PER {f.pe_trailing:.1f} — 고평가 구간")
        if f.debt_to_equity and f.debt_to_equity > 200:
            warnings_list.append(f"⚠️ 부채비율 {f.debt_to_equity:.0f}% — 높은 레버리지")
        if score.altman_z > 0 and score.altman_z < 1.81:
            warnings_list.append(f"🚨 알트만 Z {score.altman_z:.2f} — 부도 위험 구간")
        if score.graham_upside < -30:
            warnings_list.append(f"⚠️ 그레이엄 대비 {score.graham_upside:.0f}% 고평가")

        score.warnings = warnings_list
        score.rating   = self._rating(score.total_score)
        score.summary  = self._make_summary(score, component_scores)
        return score

    # ── 피오트로스키 F-스코어 ─────────────────────────────────────────────────

    def _piotroski(self, f: FundamentalData) -> tuple[int, dict]:
        """
        피오트로스키 F-스코어 (0~9점).
        조셉 피오트로스키 교수가 1980~1996년 데이터로 검증.
        7점 이상: 강한 매수 신호 / 3점 이하: 회피
        """
        detail = {}

        # [수익성] Profitability (4점)
        detail["ROA > 0"]       = 1 if (f.roa or 0) > 0 else 0
        detail["영업CF > 0"]     = 1 if (f.operating_cashflow or 0) > 0 else 0
        detail["ROA 개선"]       = 1 if (f.earnings_growth or 0) > 0 else 0
        # CF > ROA (발생주의 vs 현금주의): 현금이 이익보다 많으면 양호
        cf_ratio = (f.operating_cashflow / (f.market_cap or 1)) if f.operating_cashflow else 0
        detail["CF > ROA"]      = 1 if cf_ratio > (f.roa or 0) else 0

        # [레버리지/유동성] Leverage (3점)
        detail["부채 감소"]      = 1 if (f.debt_to_equity or 999) < 100 else 0
        detail["유동비율 양호"]  = 1 if (f.current_ratio or 0) > 1.5 else 0
        # 주식 희석 없음 (근사: 성장률이 있다는 가정)
        detail["희석 없음"]      = 1 if (f.earnings_growth or -1) >= 0 else 0

        # [운영 효율성] Efficiency (2점)
        detail["매출총이익률 개선"] = 1 if (f.gross_margin or 0) > 0.3 else 0
        detail["자산회전율"]      = 1 if (f.revenue_growth or 0) > 0 else 0

        total = sum(detail.values())
        return total, detail

    # ── 그레이엄 넘버 ─────────────────────────────────────────────────────────

    def _graham_number(self, f: FundamentalData) -> float:
        """
        벤저민 그레이엄의 적정주가 공식.
        Graham Number = √(22.5 × EPS × BPS)

        22.5 = PER 15 × PBR 1.5 (그레이엄이 허용하는 최대치)
        그레이엄 넘버 이하에서 매수하면 안전마진 확보.
        """
        eps = f.eps
        bps = f.book_value

        if eps is None or bps is None:
            return 0.0
        if eps <= 0 or bps <= 0:
            return 0.0

        return float(np.sqrt(22.5 * eps * bps))

    # ── DCF 내재가치 ──────────────────────────────────────────────────────────

    def _dcf_value(self, f: FundamentalData) -> float:
        """
        간략화된 DCF (Discounted Cash Flow) 모델.

        단계:
          1. 기준 FCF = 주당 잉여현금흐름
          2. 성장률 = min(earnings_growth, 25%) — 지나친 낙관 방지
          3. 10년간 현금흐름 할인 (WACC = 무위험률 + β × ERP)
          4. 영구 성장 모형으로 Terminal Value 계산
          5. 주당 내재가치 반환

        WACC:
          beta = 베타 계수 (시장 민감도)
          WACC = Rf + beta × ERP
               = 4.5% + beta × 5.5%
        """
        if not f.free_cashflow or not f.market_cap or f.market_cap <= 0:
            return 0.0

        # 주당 FCF 추정 (시총 / 주가 = 주식 수)
        if f.current_price <= 0:
            return 0.0

        shares = f.market_cap / f.current_price
        fcf_ps = f.free_cashflow / shares

        if fcf_ps <= 0:
            return 0.0

        # 성장률 (보수적으로 캡)
        g = min(f.earnings_growth or 0.08, 0.25)
        if g < 0:
            g = 0.03

        # WACC
        beta = f.beta or 1.0
        beta = max(0.5, min(beta, 2.5))   # 극단값 클리핑
        wacc = self.RISK_FREE_RATE + beta * self.EQUITY_RISK_PREM
        wacc = max(wacc, 0.06)             # 최소 6%

        # 현금흐름 할인
        pv = 0.0
        for t in range(1, self.DCF_YEARS + 1):
            cf = fcf_ps * ((1 + g) ** t)
            pv += cf / ((1 + wacc) ** t)

        # Terminal Value (Gordon Growth Model)
        terminal_cf  = fcf_ps * ((1 + g) ** self.DCF_YEARS) * (1 + self.TERMINAL_GROWTH)
        terminal_val = terminal_cf / (wacc - self.TERMINAL_GROWTH)
        tv_pv        = terminal_val / ((1 + wacc) ** self.DCF_YEARS)

        intrinsic = pv + tv_pv
        return max(intrinsic, 0.0)

    # ── 알트만 Z-스코어 ───────────────────────────────────────────────────────

    def _altman_z(self, f: FundamentalData) -> float:
        """
        알트만 Z-스코어 (1968, Edward Altman).
        2년 내 부도 예측 정확도 ~72%.

        Z = 1.2×X1 + 1.4×X2 + 3.3×X3 + 0.6×X4 + 1.0×X5

        X1 = 운전자본 / 총자산     (유동성)
        X2 = 이익잉여금 / 총자산   (축적 수익성)
        X3 = EBIT / 총자산         (운영 수익성)
        X4 = 시총 / 부채 총계      (레버리지)
        X5 = 매출 / 총자산         (자산 효율성)

        해석:
          Z > 2.99 → 안전 지대
          1.81~2.99 → 그레이 존 (주의)
          Z < 1.81 → 위험 지대
        """
        if not f.market_cap or f.market_cap <= 0:
            return 0.0

        # 가용 지표로 근사 계산
        x1 = (f.current_ratio or 1.5) / 5.0        # 유동비율 근사
        x2 = max((f.roe or 0) * 0.3, 0)            # 이익잉여금 근사
        x3 = f.roa or 0                             # ROA ≈ EBIT/총자산
        x4 = 1.0 / max(f.debt_to_equity or 100, 1) * 100  # 역부채비율
        x5 = max(f.revenue_growth or 0, 0) + 0.5   # 매출 효율 근사

        z = 1.2*x1 + 1.4*x2 + 3.3*x3 + 0.6*x4 + 1.0*x5
        return round(float(np.clip(z, 0, 10)), 2)

    # ── 버핏 체크리스트 ───────────────────────────────────────────────────────

    def _buffett_checklist(self, f: FundamentalData) -> tuple[int, dict]:
        """
        워런 버핏의 투자 기준 10가지 체크리스트.
        (버핏의 주주 서한 및 인터뷰 기반 재구성)
        """
        detail = {}

        # 1. ROE > 15% (자기자본 효율)
        detail["ROE > 15%"]         = 1 if (f.roe or 0) > 0.15 else 0

        # 2. 순이익률 > 10%
        detail["순이익률 > 10%"]    = 1 if (f.net_margin or 0) > 0.10 else 0

        # 3. 부채비율 < 80% (D/E < 80%)
        detail["저부채 (D/E<80%)"]  = 1 if (f.debt_to_equity or 999) < 80 else 0

        # 4. 유동비율 > 1.0 (단기 지급 능력)
        detail["유동비율 > 1.0"]    = 1 if (f.current_ratio or 0) > 1.0 else 0

        # 5. 꾸준한 이익 성장 (양의 이익 성장)
        detail["이익 성장 (+)"]     = 1 if (f.earnings_growth or -1) > 0 else 0

        # 6. PER < 25 (적정 가격)
        detail["PER < 25"]          = 1 if f.pe_trailing and f.pe_trailing < 25 else 0

        # 7. PBR < 3.0 (자산 대비 가격)
        detail["PBR < 3.0"]         = 1 if f.pb_ratio and f.pb_ratio < 3.0 else 0

        # 8. 배당 지급 (주주 환원)
        detail["배당 지급"]          = 1 if (f.dividend_yield or 0) > 0 else 0

        # 9. 영업이익률 > 15% (경쟁우위 proxy)
        detail["영업이익률 > 15%"]  = 1 if (f.operating_margin or 0) > 0.15 else 0

        # 10. FCF 양수 (실제 현금 창출)
        detail["FCF > 0"]           = 1 if (f.free_cashflow or -1) > 0 else 0

        total = sum(detail.values())
        return total, detail

    # ── 종합 점수 ─────────────────────────────────────────────────────────────

    def _total_score(self, f: FundamentalData, s: ValueScore) -> tuple[float, dict]:
        """
        0~100 종합 가치 점수.
        각 요소에 가중치 부여:
          피오트로스키  25%
          버핏 체크     25%
          DCF 상승여력  20%
          그레이엄      15%
          알트만 Z      15%
        """
        comp = {}

        # 피오트로스키 (0~9 → 0~25)
        comp["piotroski"] = (s.piotroski_score / 9) * 25

        # 버핏 (0~10 → 0~25)
        comp["buffett"]   = (s.buffett_score / 10) * 25

        # DCF 상승여력 (-50%~+100% → 0~20)
        dcf_norm = np.clip(s.dcf_upside, -50, 100)
        comp["dcf"]       = ((dcf_norm + 50) / 150) * 20

        # 그레이엄 상승여력 (-50%~+100% → 0~15)
        gn_norm = np.clip(s.graham_upside, -50, 100)
        comp["graham"]    = ((gn_norm + 50) / 150) * 15

        # 알트만 Z (0~5+ → 0~15)
        z_norm = np.clip(s.altman_z, 0, 5)
        comp["altman"]    = (z_norm / 5) * 15

        total = float(np.clip(sum(comp.values()), 0, 100))
        return round(total, 1), comp

    def _rating(self, score: float) -> str:
        if score >= 70:   return "🟢 매수 (BUY)"
        if score >= 50:   return "🟡 보유 (HOLD)"
        if score >= 30:   return "🟠 관망 (WATCH)"
        return                   "🔴 회피 (AVOID)"

    def _make_summary(self, s: ValueScore, comp: dict) -> str:
        lines = []
        if s.piotroski_score >= 7:
            lines.append("재무 건전성 우수")
        elif s.piotroski_score <= 3:
            lines.append("재무 건전성 취약")
        if s.dcf_upside > 20:
            lines.append(f"DCF 대비 {s.dcf_upside:.0f}% 저평가")
        elif s.dcf_upside < -20:
            lines.append(f"DCF 대비 {abs(s.dcf_upside):.0f}% 고평가")
        if s.graham_upside > 0:
            lines.append(f"그레이엄 적정가 이하")
        if s.altman_z > 0 and s.altman_z < 1.81:
            lines.append("부도 위험 경고")
        return " | ".join(lines) if lines else "기본 분석 완료"

    # ── 리포트 출력 ───────────────────────────────────────────────────────────

    def print_report(self, s: ValueScore) -> None:
        print(f"""
╔══════════════════════════════════════════════════════════╗
║  💰 가치투자 분석: {s.symbol:<10} {s.name[:25]:<25} ║
╠══════════════════════════════════════════════════════════╣

  📊 핵심 지표
  ├─ PER:        {self._fmt(s.pe, '.1f')}배
  ├─ PBR:        {self._fmt(s.pb, '.2f')}배
  ├─ ROE:        {self._fmt(s.roe, '.1%') if s.roe else 'N/A'}
  ├─ 부채비율:   {self._fmt(s.debt_equity, '.0f')}%
  └─ FCF 수익률: {self._fmt(s.fcf_yield, '.1f')}%

  🏆 피오트로스키 F-스코어:  {s.piotroski_score}/9
     {"█" * s.piotroski_score}{"░" * (9 - s.piotroski_score)}
     {">= 7: 강한 매수 신호" if s.piotroski_score >= 7 else "<= 3: 회피 신호" if s.piotroski_score <= 3 else "4~6: 중립"}

  📐 그레이엄 넘버:   ${s.graham_number:>8.2f}
     상승여력:        {s.graham_upside:>+7.1f}%  {"✅ 저평가" if s.graham_upside > 0 else "❌ 고평가"}

  💵 DCF 내재가치:    ${s.dcf_value:>8.2f}
     상승여력:        {s.dcf_upside:>+7.1f}%  {"✅ 저평가" if s.dcf_upside > 0 else "❌ 고평가"}

  🛡️  알트만 Z-스코어: {s.altman_z:.2f}
     {"✅ 안전 지대 (>2.99)" if s.altman_z >= 2.99 else "⚠️ 그레이 존 (1.81~2.99)" if s.altman_z >= 1.81 else "🚨 위험 지대 (<1.81)"}

  🏅 버핏 체크리스트: {s.buffett_score}/10
     {"█" * s.buffett_score}{"░" * (10 - s.buffett_score)}

  ╔═══════════════════════════════════╗
  ║  종합 가치 점수: {s.total_score:>5.1f}/100        ║
  ║  최종 판단:  {s.rating:<25} ║
  ╚═══════════════════════════════════╝
  {s.summary}
""")
        for w in s.warnings:
            print(f"  {w}")
        if s.warnings:
            print()

    def _fmt(self, v, fmt) -> str:
        if v is None:
            return "N/A"
        try:
            return format(v, fmt)
        except Exception:
            return str(v)
