"""
금융 로직 단위 테스트 — 대시보드 핵심 계산의 정확성·경계조건 검증.

대상 (모두 Streamlit 비의존, import 가능):
  app.analytics.indicators   rsi_wilder, tech_score, ichimoku, max_pain, model_forecast
  app.ml.features.fundamental_features  fundamental_score, _clip, _safe_float
  app.ml.models.improved_ensemble       _rsi, build_features, RegimeDetector, validation_metrics
  app.news.aggregator        _iso_from_any, _normalize_yf_item
  app.storage.store          load_json / save_json 라운드트립

실행:  pytest app/tests/unit/test_financial_logic.py -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ─────────────────────────────────────────────────────────────────────────────
# 공통 픽스처
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def ohlcv() -> pd.DataFrame:
    """상승 추세 합성 OHLCV (재현 가능)."""
    rng = np.random.default_rng(42)
    n = 400
    close = 100 * np.exp(np.cumsum(rng.normal(0.0008, 0.015, n)))
    idx = pd.date_range("2023-01-02", periods=n, freq="B")
    return pd.DataFrame({
        "open": close * (1 + rng.normal(0, 0.002, n)),
        "high": close * 1.012, "low": close * 0.988,
        "close": close, "volume": rng.integers(1e6, 5e6, n).astype(float),
    }, index=idx)


# ─────────────────────────────────────────────────────────────────────────────
# RSI — 알려진 성질로 검증
# ─────────────────────────────────────────────────────────────────────────────

def test_rsi_monotone_up_is_high():
    from app.analytics.indicators import rsi_wilder
    prices = np.arange(1, 60, dtype=float)        # 계속 상승만
    assert rsi_wilder(prices, 14) > 99.0          # 하락 없음 → RSI ~100

def test_rsi_monotone_down_is_low():
    from app.analytics.indicators import rsi_wilder
    prices = np.arange(60, 1, -1, dtype=float)    # 계속 하락만
    assert rsi_wilder(prices, 14) < 1.0

def test_rsi_bounds_and_short_input():
    from app.analytics.indicators import rsi_wilder
    rng = np.random.default_rng(0)
    prices = 100 + np.cumsum(rng.normal(0, 1, 100))
    r = rsi_wilder(prices, 14)
    assert 0.0 <= r <= 100.0
    assert rsi_wilder(np.array([1.0, 2.0]), 14) == 50.0   # 데이터 부족 → 중립

def test_rsi_matches_improved_ensemble_impl(ohlcv):
    """analytics 의 RSI 와 ML 모듈의 _rsi 최신값이 일치(중복 구현 정합성)."""
    from app.analytics.indicators import rsi_wilder
    from app.ml.models.improved_ensemble import _rsi
    c = ohlcv["close"].values
    assert rsi_wilder(c, 14) == pytest.approx(float(_rsi(c, 14)[-1]), abs=1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# tech_score
# ─────────────────────────────────────────────────────────────────────────────

def test_tech_score_bounds(ohlcv):
    from app.analytics.indicators import tech_score
    ts, rsi = tech_score(ohlcv)
    assert 0.0 <= ts <= 100.0 and 0.0 <= rsi <= 100.0

def test_tech_score_oversold_boosts():
    """과매도(저RSI) 구간은 점수가 50 이상으로 가산된다."""
    from app.analytics.indicators import tech_score
    close = np.linspace(200, 100, 80)             # 꾸준한 하락 → 낮은 RSI
    df = pd.DataFrame({"open": close, "high": close * 1.01,
                       "low": close * 0.99, "close": close,
                       "volume": np.ones(80)})
    ts, rsi = tech_score(df)
    assert rsi < 30 and ts >= 65                  # 과매도 → 반등 기대 가산


# ─────────────────────────────────────────────────────────────────────────────
# ichimoku — 미래 26봉 선행, 컬럼 구성
# ─────────────────────────────────────────────────────────────────────────────

def test_ichimoku_extends_future(ohlcv):
    from app.analytics.indicators import ichimoku
    ichi, ext = ichimoku(ohlcv, shift=26)
    assert len(ext) == len(ohlcv) + 26
    assert set(["tenkan", "kijun", "span_a", "span_b"]) <= set(ichi.columns)
    # 선행스팬은 마지막 26봉(미래)까지 값이 있어야 함
    assert ichi["span_a"].reindex(ext[-26:]).notna().all()

def test_ichimoku_span_a_is_midpoint(ohlcv):
    """선행스팬A = (전환선+기준선)/2 를 26봉 시프트한 값 (정의 일치)."""
    from app.analytics.indicators import ichimoku
    ichi, ext = ichimoku(ohlcv, shift=26)
    raw_mid = (ichi["tenkan"] + ichi["kijun"]) / 2
    # span_a[t] == raw_mid[t-26]
    aligned = ichi["span_a"].dropna()
    for ts in aligned.index[-5:]:
        pos = ext.get_loc(ts)
        if pos >= 26:
            assert aligned[ts] == pytest.approx(raw_mid.iloc[pos - 26], rel=1e-6, nan_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# max_pain — 대칭/비대칭 OI
# ─────────────────────────────────────────────────────────────────────────────

def test_max_pain_symmetric_is_center():
    from app.analytics.indicators import max_pain
    calls = pd.DataFrame({"strike": [90, 100, 110], "openInterest": [100, 50, 10]})
    puts = pd.DataFrame({"strike": [90, 100, 110], "openInterest": [10, 50, 100]})
    assert max_pain(calls, puts) == 100

def test_max_pain_empty_returns_none():
    from app.analytics.indicators import max_pain
    empty = pd.DataFrame({"strike": [], "openInterest": []})
    assert max_pain(empty, empty) is None

def test_max_pain_handles_nan_oi():
    from app.analytics.indicators import max_pain
    calls = pd.DataFrame({"strike": [100, 110], "openInterest": [np.nan, 5]})
    puts = pd.DataFrame({"strike": [100, 110], "openInterest": [5, np.nan]})
    assert max_pain(calls, puts) in (100, 110)    # NaN→0 처리되어 예외 없음


# ─────────────────────────────────────────────────────────────────────────────
# model_forecast — 확률 단조성·경계
# ─────────────────────────────────────────────────────────────────────────────

def test_model_forecast_monotone_in_probability(ohlcv):
    from app.analytics.indicators import model_forecast
    lo = model_forecast(ohlcv, 0.2)["exp5"]
    mid = model_forecast(ohlcv, 0.5)["exp5"]
    hi = model_forecast(ohlcv, 0.8)["exp5"]
    assert lo < mid < hi                          # 상승확률↑ → 기대수익↑

def test_model_forecast_keys_and_compounding(ohlcv):
    from app.analytics.indicators import model_forecast
    fc = model_forecast(ohlcv, 0.6)
    assert {"exp5", "exp21", "up_m", "dn_m"} <= set(fc)
    assert fc["dn_m"] <= 0 <= fc["up_m"]          # 하락평균≤0≤상승평균


# ─────────────────────────────────────────────────────────────────────────────
# fundamental_score
# ─────────────────────────────────────────────────────────────────────────────

def test_fundamental_score_empty_is_neutral():
    from app.ml.features.fundamental_features import fundamental_score
    assert fundamental_score({}) == 50.0

def test_fundamental_score_good_vs_bad():
    from app.ml.features.fundamental_features import fundamental_score
    good = fundamental_score({"peg_ratio": 0.8, "roe_mean_4y": 0.25,
                              "roe_consistency": 1.0, "fcf_yield": 0.06,
                              "earnings_growth": 0.3, "dcf_upside": 25.0})
    bad = fundamental_score({"peg_ratio": 4.0, "roe_mean_4y": -0.1,
                             "fcf_yield": -0.05, "earnings_growth": -0.3})
    assert good > 65 and bad < 35 and good > bad

def test_fundamental_clip_and_safe_float():
    from app.ml.features.fundamental_features import _clip, _safe_float
    assert _clip("peg_ratio", 999) == 5.0         # 상한 클립
    assert _clip("peg_ratio", -5) == 0.0          # 하한 클립
    assert _safe_float("nan") is None and _safe_float(float("inf")) is None
    assert _safe_float("3.5") == 3.5


# ─────────────────────────────────────────────────────────────────────────────
# build_features / fundamental tilt
# ─────────────────────────────────────────────────────────────────────────────

def test_build_features_no_nan_and_fundamentals(ohlcv):
    from app.ml.models.improved_ensemble import build_features
    X = build_features(ohlcv, fundamentals={"peg_ratio": 1.2, "roe": 0.2})
    assert len(X) == len(ohlcv)
    assert not X.isna().any().any()               # NaN/inf 모두 0으로 정리됨
    assert any(c.startswith("fund_") for c in X.columns)

def test_fundamental_tilt_bounds():
    from app.ml.models.improved_ensemble import ImprovedEnsembleModel
    assert ImprovedEnsembleModel._fundamental_tilt(None) == 0.0
    t_good = ImprovedEnsembleModel._fundamental_tilt(
        {"peg_ratio": 0.7, "roe_mean_4y": 0.3, "roe_consistency": 1.0,
         "fcf_yield": 0.07, "dcf_upside": 30.0})
    t_bad = ImprovedEnsembleModel._fundamental_tilt(
        {"peg_ratio": 4.5, "roe_mean_4y": -0.2, "fcf_yield": -0.1})
    assert 0 < t_good <= 0.06 and -0.06 <= t_bad < 0


# ─────────────────────────────────────────────────────────────────────────────
# RegimeDetector — 폴백 경로 (hmmlearn 유무와 무관하게 동작)
# ─────────────────────────────────────────────────────────────────────────────

def test_regime_detector_outputs_valid_probabilities():
    from app.ml.models.improved_ensemble import RegimeDetector
    rng = np.random.default_rng(1)
    returns = rng.normal(0, 0.02, 300)
    det = RegimeDetector().fit(returns)
    name, proba = det.current_regime(returns)
    assert name in ("상승장", "횡보장", "하락장")
    assert abs(sum(proba.values()) - 1.0) < 1e-6  # 확률 합 = 1
    assert all(0 <= p <= 1 for p in proba.values())


# ─────────────────────────────────────────────────────────────────────────────
# 모델 검증 지표 (validation_metrics) — 학습 후 성적표 구조·범위
# ─────────────────────────────────────────────────────────────────────────────

def test_validation_metrics_structure(ohlcv):
    from app.ml.models.improved_ensemble import ImprovedEnsembleModel

    class _F:
        def __init__(self, d, s): self.data, self.symbol = d, s

    model = ImprovedEnsembleModel().fit(_F(ohlcv, "TEST"))
    m = model.validation_metrics()
    if m is None:                                  # 데이터 부족 시 None 허용
        pytest.skip("검증 표본 부족")
    assert m["n"] >= 20
    assert 0.0 <= m["accuracy"] <= 1.0
    assert 0.0 <= m["base_rate"] <= 1.0
    assert m["grade"] in ("A", "B", "C", "D", "F", "N/A")
    if m["auc"] == m["auc"]:                        # NaN 아니면
        assert 0.0 <= m["auc"] <= 1.0
    assert isinstance(m["calibration"], list)


# ─────────────────────────────────────────────────────────────────────────────
# 뉴스 수집기 정규화 (신/구 yfinance 형식)
# ─────────────────────────────────────────────────────────────────────────────

def test_iso_from_epoch_and_string():
    from app.news.aggregator import _iso_from_any
    assert _iso_from_any(1760000000).startswith("20")     # epoch → ISO
    assert _iso_from_any("2026-06-11T13:00:00Z").startswith("2026-06-11")
    assert _iso_from_any(None) == "" and _iso_from_any("garbage") == ""

def test_normalize_new_and_old_shapes():
    from app.news.aggregator import _normalize_yf_item
    new = {"content": {"title": "Apple beats", "summary": "EPS up",
            "pubDate": "2026-06-11T13:00:00Z",
            "provider": {"displayName": "Reuters"},
            "canonicalUrl": {"url": "https://x.com/a"}}}
    old = {"title": "Old style", "publisher": "AP", "link": "https://y.com",
           "providerPublishTime": 1760000000}
    a, b = _normalize_yf_item(new), _normalize_yf_item(old)
    assert a["source"] == "Reuters" and a["url"] == "https://x.com/a"
    assert b["source"] == "AP" and b["published"]
    assert _normalize_yf_item({"content": {"title": ""}}) is None   # 제목 없으면 제외


# ─────────────────────────────────────────────────────────────────────────────
# 영속 저장 라운드트립
# ─────────────────────────────────────────────────────────────────────────────

def test_storage_roundtrip_and_default(tmp_path, monkeypatch):
    import app.storage.store as store
    monkeypatch.setattr(store, "_DATA_DIR", tmp_path)
    assert store.load_json("missing.json", default=[]) == []      # 없으면 default
    payload = [{"ticker": "AAPL", "shares": 10, "price": 150.0}]
    assert store.save_json("p.json", payload) is True
    assert store.load_json("p.json") == payload                   # 라운드트립
