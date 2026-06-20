"""
=============================================================================
현대 퀀트 앙상블 모델 — 업그레이드 버전
=============================================================================
현재 문제점:
  - 신뢰도가 고정값 (0.55~0.80) — 데이터 기반이 아님
  - ML 모델이 실제 학습 없이 사용됨
  - 시장 레짐(국면) 무시

개선 사항:
  ① HMM 기반 레짐 감지 (상승장 / 하락장 / 횡보장)
  ② 레짐별 전략 가중치 동적 조절
  ③ Walk-forward 교차검증 + 확률 보정 (Platt Scaling)
  ④ 기술적 신호 + 가치 점수 + ML 예측 통합
  ⑤ 신뢰도 = 보정된 사후 확률 (0~100%)
=============================================================================
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# 출력 데이터 클래스
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ModelPrediction:
    """업그레이드된 앙상블 모델 예측 결과"""
    symbol:           str
    regime:           str          # "상승장" / "하락장" / "횡보장"
    regime_proba:     dict         # 각 레짐 확률
    signal:           str          # "매수" / "매도" / "중립"
    up_probability:   float        # 상승 확률 (0~1)
    confidence:       float        # 신뢰도 (0~1, 보정된 확률)
    feature_importance: dict       # 주요 기여 피처
    component_signals: dict        # 세부 신호 구성
    model_agreement:  float        # 모델 간 일치도
    fundamental_tilt: float = 0.0  # 펀더멘털 기반 확률 보정량 (±)


# ─────────────────────────────────────────────────────────────────────────────
# ① HMM 레짐 감지
# ─────────────────────────────────────────────────────────────────────────────

class RegimeDetector:
    """
    Hidden Markov Model (HMM) 기반 시장 레짐 감지.

    3가지 레짐:
      State 0 = 하락장 (Bear): 고변동성 + 음의 수익
      State 1 = 횡보장 (Sideways): 중간 변동성
      State 2 = 상승장 (Bull): 저변동성 + 양의 수익

    퀀트 활용:
      - 상승장: MA, 모멘텀 전략 과중
      - 하락장: 리버설, 방어 전략 과중
      - 횡보장: 레인지 전략 과중
    """

    N_STATES = 3
    REGIME_NAMES = {0: "하락장", 1: "횡보장", 2: "상승장"}

    def __init__(self):
        self._model = None
        self._fitted = False

    def fit(self, returns: np.ndarray) -> "RegimeDetector":
        """수익률 배열로 HMM 학습"""
        try:
            from hmmlearn import hmm
            obs = self._make_obs(returns)

            # diag 공분산 + 표준화된 입력 → full 대비 수렴이 훨씬 안정적
            self._model = hmm.GaussianHMM(
                n_components=self.N_STATES,
                covariance_type="diag",
                n_iter=500,
                tol=1e-4,
                random_state=42,
            )
            self._model.fit(obs)

            # 레짐 정렬: 평균 수익률 기준으로 정렬 (낮은→높은)
            means = self._model.means_[:, 0]
            self._order = np.argsort(means)   # 0=낮음(하락), 2=높음(상승)
            self._fitted = True

        except Exception:
            # HMM 실패 시 단순 변동성 기반 레짐으로 대체
            self._fitted = False
            self._returns = returns

        return self

    def predict(self, returns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns:
            states:  레짐 인덱스 배열 (0=하락, 1=횡보, 2=상승)
            probas:  각 상태 확률 행렬 (n_samples × 3)
        """
        if self._fitted and self._model is not None:
            obs    = self._make_obs(returns)
            raw    = self._model.predict(obs)
            probas = self._model.predict_proba(obs)

            # 정렬 적용
            mapped = np.vectorize(lambda s: np.where(self._order == s)[0][0])(raw)
            probas_reordered = probas[:, self._order]
            return mapped, probas_reordered
        else:
            # 폴백: 변동성 기반 단순 레짐
            return self._volatility_regime(returns)

    def current_regime(self, returns: np.ndarray) -> tuple[str, dict]:
        """
        현재 레짐과 확률 딕셔너리 반환.

        버그 수정: 이전에는 이름=Viterbi 경로, 확률=사후확률을 섞어 써서
        "이름은 횡보장인데 상승장 확률이 가장 높은" 모순이 생겼다.
        이제 최근 5일 사후확률 평균의 argmax 로 이름을 정해 항상 일치하고,
        하루짜리 노이즈에 레짐이 깜빡이는 것도 줄였다.
        """
        _, probas = self.predict(returns)
        recent = probas[-5:] if len(probas) >= 5 else probas
        cur_proba = recent.mean(axis=0)
        cur_proba = cur_proba / cur_proba.sum()       # 평균 후 재정규화
        cur_state = int(np.argmax(cur_proba))
        return (
            self.REGIME_NAMES[cur_state],
            {self.REGIME_NAMES[i]: float(p) for i, p in enumerate(cur_proba)},
        )

    def _make_obs(self, returns: np.ndarray) -> np.ndarray:
        """HMM 입력 특성: [수익률, 변동성] — z-score 표준화로 수렴 안정화"""
        vol = pd.Series(returns).rolling(10).std().bfill().values
        obs = np.column_stack([returns, vol]).astype(float)
        mu = obs.mean(axis=0)
        sd = obs.std(axis=0)
        sd[sd == 0] = 1.0
        return (obs - mu) / sd

    def _volatility_regime(self, returns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """HMM 폴백: 변동성 + 수익률 기반 단순 레짐"""
        n = len(returns)
        vol = pd.Series(returns).rolling(20).std().bfill().values
        mu  = pd.Series(returns).rolling(20).mean().bfill().values
        vol_med = np.median(vol)

        states = np.zeros(n, dtype=int)
        for i in range(n):
            if vol[i] > vol_med * 1.3 and mu[i] < 0:
                states[i] = 0   # 하락장
            elif vol[i] < vol_med * 0.7 and mu[i] > 0:
                states[i] = 2   # 상승장
            else:
                states[i] = 1   # 횡보장

        # 간단한 확률 추정
        probas = np.zeros((n, 3))
        for i in range(n):
            s = states[i]
            probas[i, s] = 0.6
            for j in range(3):
                if j != s:
                    probas[i, j] = 0.2

        return states, probas


# ─────────────────────────────────────────────────────────────────────────────
# ② 특성 생성 (기술적 + 레짐 + 가치)
# ─────────────────────────────────────────────────────────────────────────────

def build_features(
    df: pd.DataFrame,
    value_score: float = 50.0,
    fundamentals: Optional[dict] = None,
) -> pd.DataFrame:
    """
    기술적 + 레짐 + 가치 + 펀더멘털 통합 피처 생성.

    현재 대비 추가된 피처:
      - 레짐 확률 (HMM 출력)
      - 다중 타임프레임 모멘텀 (1/3/6/12개월)
      - 팩터 기반: 가치(Value), 품질(Quality), 저변동성(Low-Vol)
      - 가치투자 점수 (ValueScore → 정규화)
      - 펀더멘털 (PEG, 지속 ROE, FCF 수익률 등 — fundamental_features.py)

    fundamentals 는 종목당 스칼라라 단일 종목 시계열 학습에서는 분산이 0이지만
    (트리 모델이 자동 무시), 여러 종목을 합쳐 크로스섹션 학습할 때 식별력을 갖는다.
    단일 종목 예측에서는 predict() 의 펀더멘털 틸트가 실질 보정을 담당한다.
    """
    c = df["close"].values
    h = df["high"].values
    l = df["low"].values
    v = df["volume"].values if "volume" in df.columns else np.ones(len(c))
    n = len(c)

    features = pd.DataFrame(index=df.index)

    # ── 수익률 팩터 ──────────────────────────────────────────────────────────
    for d in [1, 5, 10, 21, 63, 126, 252]:
        features[f"ret_{d}d"] = pd.Series(c).pct_change(d).values

    # ── 기술적 지표 ──────────────────────────────────────────────────────────
    # RSI (7/14/21일)
    for p in [7, 14, 21]:
        features[f"rsi_{p}"] = _rsi(c, p)

    # 이동평균 대비 위치
    for p in [10, 20, 50, 100, 200]:
        ma = pd.Series(c).rolling(p).mean().values
        features[f"ma_ratio_{p}"] = c / (ma + 1e-9) - 1

    # MACD
    ema12 = pd.Series(c).ewm(span=12).mean().values
    ema26 = pd.Series(c).ewm(span=26).mean().values
    macd  = ema12 - ema26
    sig   = pd.Series(macd).ewm(span=9).mean().values
    features["macd_hist"]  = macd - sig
    features["macd_cross"] = np.sign(macd - sig) - np.sign(
        np.roll(macd - sig, 1))

    # 볼린저 밴드 %B
    bb_mid = pd.Series(c).rolling(20).mean().values
    bb_std = pd.Series(c).rolling(20).std().values
    features["bb_pct_b"] = (c - (bb_mid - 2*bb_std)) / (4*bb_std + 1e-9)
    features["bb_width"] = 4 * bb_std / (bb_mid + 1e-9)

    # ATR (변동성)
    tr = np.maximum(h - l, np.maximum(
        np.abs(h - np.roll(c, 1)), np.abs(l - np.roll(c, 1))))
    for p in [14, 21]:
        features[f"atr_{p}"] = pd.Series(tr).rolling(p).mean().values / c

    # ── 변동성 팩터 ──────────────────────────────────────────────────────────
    ret1d = pd.Series(c).pct_change().values
    for p in [5, 21, 63]:
        features[f"vol_{p}d"] = pd.Series(ret1d).rolling(p).std().values * np.sqrt(252)

    # 변동성의 변동성 (VoV)
    features["vov"] = pd.Series(
        pd.Series(ret1d).rolling(21).std().values).rolling(21).std().values

    # ── 거래량 팩터 ──────────────────────────────────────────────────────────
    features["vol_ratio"] = pd.Series(v).rolling(20).apply(
        lambda x: x[-1] / (x[:-1].mean() + 1e-9), raw=True).values
    # OBV Z-score
    obv = np.cumsum(np.sign(np.diff(np.append(c[0], c))) * v)
    features["obv_z"] = pd.Series(obv).rolling(20).apply(
        lambda x: (x[-1] - x.mean()) / (x.std() + 1e-9), raw=True).values

    # ── 모멘텀 팩터 (Fama-French UMD 스타일) ─────────────────────────────────
    # skip-1 모멘텀: t-252 ~ t-21 수익률
    for skip, look in [(21, 252), (21, 126), (5, 63), (1, 21)]:
        if n > look + skip:
            mom = pd.Series(c).shift(skip).pct_change(look - skip).values
        else:
            mom = np.zeros(n)
        features[f"mom_{look}d"] = mom

    # 52주 최고가 대비 위치 (52W High effect)
    wk52_high = pd.Series(h).rolling(252, min_periods=50).max().values
    features["pct_from_52h"] = c / (wk52_high + 1e-9) - 1

    # ── 품질 팩터 (Quality: 추세 안정성) ─────────────────────────────────────
    # 수익률 일관성 (양수 수익일 비율)
    for p in [21, 63]:
        features[f"up_ratio_{p}"] = pd.Series(ret1d > 0).rolling(p).mean().values

    # ── 가치투자 점수 (외부 입력, 정규화) ────────────────────────────────────
    # value_score: 0~100 → -1~+1 정규화
    features["value_score_norm"] = (value_score - 50) / 50

    # ── 레짐 피처 ────────────────────────────────────────────────────────────
    # 단순 트렌드 레짐 (복잡한 HMM 없이)
    ma50  = pd.Series(c).rolling(50).mean().values
    ma200 = pd.Series(c).rolling(200, min_periods=50).mean().values
    features["trend_regime"] = np.sign(ma50 - ma200)   # +1=상승, -1=하락

    # ── 펀더멘털 피처 (종목당 스칼라 브로드캐스트) ──────────────────────────
    if fundamentals:
        for key, val in fundamentals.items():
            try:
                features[f"fund_{key}"] = float(val)
            except (TypeError, ValueError):
                continue

    return features.replace([np.inf, -np.inf], np.nan).fillna(0)


def _rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
    d = np.diff(prices)
    g = np.where(d > 0, d, 0)
    l = np.where(d < 0, -d, 0)
    ag = pd.Series(np.append(0, g)).ewm(alpha=1/period, adjust=False).mean().values
    al = pd.Series(np.append(0, l)).ewm(alpha=1/period, adjust=False).mean().values
    rs = ag / (al + 1e-9)
    return 100 - 100 / (1 + rs)


# ─────────────────────────────────────────────────────────────────────────────
# ③ 업그레이드 앙상블 모델
# ─────────────────────────────────────────────────────────────────────────────

class ImprovedEnsembleModel:
    """
    현대 퀀트 앙상블 모델.

    구성:
      L1 기저 모델 3개:
        - XGBoost     (트리 기반, 비선형 패턴)
        - LightGBM    (빠른 그래디언트 부스팅)
        - Random Forest (배깅, 다양성)
      L2 메타 모델:
        - Logistic Regression (확률 보정)
      레짐 필터:
        - HMM으로 현재 레짐 감지 → 신뢰도 조정

    학습 방법:
      Walk-Forward (확장 윈도우):
        훈련: t=0 ~ t=k
        테스트: t=k+1 ~ t=k+step
        k를 늘려가며 반복 → OOF 예측 누적
    """

    FORWARD_DAYS   = 5     # 몇 일 후 수익률 예측
    MIN_TRAIN_DAYS = 252   # 최소 학습 기간 (1년)
    WALK_STEP      = 63    # 워크포워드 스텝 (분기)

    # ── 하이퍼파라미터 최적화 (주력 L1 모델 XGBoost 대상) ──────────────────────
    # XGBoost 고정 기본값 — optimize_hyperparams() 의 비교 기준선이자
    # 튜닝 전 default. 기본 조합이 그리드 안에 포함돼 향상도(Δ)를 직접 비교한다.
    XGB_BASE_PARAMS = dict(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="logloss", verbosity=0,
    )
    # 워크포워드 AUC 로 탐색할 그리드 (3×3×2 = 18 조합). 기본값 포함.
    XGB_PARAM_GRID = {
        "max_depth":     [3, 5, 7],
        "learning_rate": [0.02, 0.05, 0.1],
        "n_estimators":  [200, 300],
    }

    def __init__(self):
        self._models      = {}
        self._meta_model  = None
        self._scaler      = None
        self._regime_det  = RegimeDetector()
        self._fitted      = False
        self._feature_imp = {}
        self._validation  = None   # OOF 검증 기록 (fit 후 채워짐)
        self._best_params = None   # 튜닝된 XGB 파라미터 (optimize_hyperparams 후)
        self._tuning      = None   # 튜닝 리포트 (최적·기본·향상도·리더보드)

    def validation_metrics(self) -> Optional[dict]:
        """
        Walk-forward OOF 예측에 대한 out-of-sample 성능 지표(성적표).

        반환 dict:
            n              검증 표본 수
            base_rate      실제 상승 비율(기준선)
            accuracy       방향 적중률 (proba>0.5 기준)
            edge           적중률 − 다수예측 기준선 (양수면 정보 있음)
            auc            ROC-AUC (0.5=무작위, >0.55면 유의미)
            brier          Brier score (낮을수록 좋음, 0.25=무작위)
            calibration    [(예측확률평균, 실제상승빈도, 표본수), ...] 5구간
            ret_by_bin     [(확률구간라벨, 평균 N일 수익률%, 표본수), ...]
            long_edge_pct  proba>0.55 일 때 평균수익 − 전체 평균수익 (%p)
            grade          AUC 기반 등급 (A~F)
        """
        v = self._validation
        if not v or len(v["proba"]) < 20:
            return None
        proba = np.asarray(v["proba"], dtype=float)
        y = np.asarray(v["y"], dtype=int)
        fwd = np.asarray(v["fwd_ret"], dtype=float)

        n = int(len(proba))
        base_rate = float(y.mean())
        pred = (proba > 0.5).astype(int)
        accuracy = float((pred == y).mean())
        edge = float(accuracy - max(base_rate, 1 - base_rate))

        try:
            from sklearn.metrics import roc_auc_score, brier_score_loss
            auc = float(roc_auc_score(y, proba)) if len(np.unique(y)) > 1 else float("nan")
            brier = float(brier_score_loss(y, proba))
        except Exception:
            auc, brier = float("nan"), float("nan")

        # 칼리브레이션 (5구간)
        calib = []
        edges = np.linspace(0.0, 1.0, 6)
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (proba >= lo) & (proba < hi if hi < 1.0 else proba <= hi)
            if m.sum() > 0:
                calib.append((float(proba[m].mean()), float(y[m].mean()), int(m.sum())))

        # 확률 구간별 평균 N일 수익률 (정보가 있으면 단조 증가)
        ret_by_bin = []
        bins = [(0.0, 0.45, "<45%"), (0.45, 0.55, "45–55%"), (0.55, 1.01, ">55%")]
        valid_fwd = ~np.isnan(fwd)
        for lo, hi, lbl in bins:
            m = (proba >= lo) & (proba < hi) & valid_fwd
            if m.sum() > 0:
                ret_by_bin.append((lbl, float(fwd[m].mean() * 100), int(m.sum())))

        long_m = (proba > 0.55) & valid_fwd
        long_edge = (float(fwd[long_m].mean() - fwd[valid_fwd].mean()) * 100
                     if long_m.sum() > 0 and valid_fwd.sum() > 0 else 0.0)

        grade = ("A" if auc >= 0.60 else "B" if auc >= 0.56 else "C" if auc >= 0.53
                 else "D" if auc >= 0.50 else "F") if auc == auc else "N/A"

        return dict(n=n, base_rate=base_rate, accuracy=accuracy, edge=edge,
                    auc=auc, brier=brier, calibration=calib, ret_by_bin=ret_by_bin,
                    long_edge_pct=long_edge, grade=grade)

    # ─────────────────────────────────────────────────────────────────────────
    # 하이퍼파라미터 최적화 (워크포워드 그리드 탐색)
    # ─────────────────────────────────────────────────────────────────────────

    def _xgb_params(self) -> dict:
        """현재 학습에 쓸 XGBoost 파라미터 — 튜닝됐으면 최적값, 아니면 기본값."""
        p = dict(self.XGB_BASE_PARAMS)
        if self._best_params:
            p.update(self._best_params)
        return p

    def best_hyperparams(self) -> Optional[dict]:
        """optimize_hyperparams() 가 채운 튜닝 리포트 반환 (없으면 None)."""
        return self._tuning

    def optimize_hyperparams(
        self,
        frame,
        fundamentals: Optional[dict] = None,
        grid: Optional[dict] = None,
    ) -> Optional[dict]:
        """
        워크포워드 out-of-sample AUC 기준으로 XGBoost 하이퍼파라미터를 그리드 탐색.

        주력 L1 모델인 XGBoost만 튜닝한다(피처 중요도·신호의 주 동인이라 비용 대비
        효과가 가장 큼). fit() 과 동일한 walk-forward 분할을 써 각 조합의 OOF 확률을
        만들고 AUC 로 채점 → 최적 조합을 self._best_params 에 저장한다. 이후 fit() 은
        _xgb_params() 를 통해 이 최적값을 자동으로 사용한다(룩어헤드 없음).

        Returns dict (없으면 None):
            best_params   최적 조합 {max_depth, learning_rate, n_estimators}
            best_auc      최적 조합의 워크포워드 AUC
            base_params   기본(고정) 조합
            base_auc      기본 조합 AUC (향상도 비교 기준선)
            improvement   best_auc − base_auc (양수면 튜닝이 유효)
            n_combos      AUC 산출에 성공한 조합 수
            n_samples     OOF 평가 표본 수(일)
            leaderboard   [(params, auc), ...] AUC 내림차순 상위 8개
        """
        from sklearn.metrics import roc_auc_score
        try:
            from xgboost import XGBClassifier
        except ImportError:
            return None   # 그리드 탐색은 실제 XGBoost 가 있을 때만 의미 있음

        grid = grid or self.XGB_PARAM_GRID
        df = frame.data.copy()
        df.index = pd.to_datetime(df.index).tz_localize(None)

        X_all = build_features(df, fundamentals=fundamentals)
        y_all = (df["close"].pct_change(self.FORWARD_DAYS)
                 .shift(-self.FORWARD_DAYS) > 0).astype(int)
        valid = X_all.notna().all(axis=1) & y_all.notna()
        X_all, y_all = X_all[valid], y_all[valid]

        n = len(X_all)
        if n < self.MIN_TRAIN_DAYS + self.FORWARD_DAYS + 10:
            return None   # 데이터 부족 → 튜닝 의미 없음

        splits = []
        start = self.MIN_TRAIN_DAYS
        while start + self.WALK_STEP <= n - self.FORWARD_DAYS:
            splits.append((start, min(start + self.WALK_STEP, n - self.FORWARD_DAYS)))
            start += self.WALK_STEP
        if not splits:
            return None

        import itertools
        keys   = list(grid.keys())
        combos = [dict(zip(keys, vals)) for vals in itertools.product(*grid.values())]

        results = []   # (combo, auc, n_eval)
        for combo in combos:
            params = dict(self.XGB_BASE_PARAMS)
            params.update(combo)
            auc, n_eval = self._wf_oof_auc(
                X_all, y_all, splits, XGBClassifier, params, roc_auc_score)
            if auc == auc:   # NaN 제외
                results.append((combo, auc, n_eval))

        if not results:
            return None

        results.sort(key=lambda r: -r[1])
        best_combo, best_auc, n_samples = results[0]

        base_combo = {k: self.XGB_BASE_PARAMS[k] for k in keys}
        base_auc   = next((a for c, a, _ in results if c == base_combo), float("nan"))

        self._best_params = best_combo
        self._tuning = dict(
            best_params = best_combo,
            best_auc    = best_auc,
            base_params = base_combo,
            base_auc    = base_auc,
            improvement = (best_auc - base_auc) if base_auc == base_auc else float("nan"),
            n_combos    = len(results),
            n_samples   = n_samples,
            leaderboard = [(c, a) for c, a, _ in results[:8]],
        )
        return self._tuning

    @staticmethod
    def _wf_oof_auc(X_all, y_all, splits, XGBClassifier, params, roc_auc_score):
        """한 파라미터 조합의 walk-forward OOF AUC 와 평가 표본 수를 반환."""
        from sklearn.preprocessing import RobustScaler
        oof = np.full(len(X_all), np.nan)
        y   = y_all.values
        try:
            for tr_end, vl_end in splits:
                sc   = RobustScaler()
                X_tr = sc.fit_transform(X_all.iloc[:tr_end])
                X_vl = sc.transform(X_all.iloc[tr_end:vl_end])
                m    = XGBClassifier(**params)
                m.fit(X_tr, y_all.iloc[:tr_end])
                oof[tr_end:vl_end] = m.predict_proba(X_vl)[:, 1]
        except Exception:
            return float("nan"), 0
        mask = ~np.isnan(oof)
        if mask.sum() < 20 or len(np.unique(y[mask])) < 2:
            return float("nan"), int(mask.sum())
        return float(roc_auc_score(y[mask], oof[mask])), int(mask.sum())

    def fit(self, frame, fundamentals: Optional[dict] = None) -> "ImprovedEnsembleModel":
        """Walk-Forward 교차검증으로 모델 학습"""
        from sklearn.preprocessing import RobustScaler
        from sklearn.linear_model import LogisticRegression
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.ensemble import RandomForestClassifier

        try:
            from xgboost import XGBClassifier
            from lightgbm import LGBMClassifier
        except ImportError:
            from sklearn.ensemble import GradientBoostingClassifier as XGBClassifier
            LGBMClassifier = RandomForestClassifier

        df = frame.data.copy()
        # 타임존 제거
        df.index = pd.to_datetime(df.index).tz_localize(None)

        # 피처 생성
        X_all = build_features(df, fundamentals=fundamentals)
        # 레이블: N일 후 수익률 > 0 이면 1, 아니면 0
        y_all = (df["close"].pct_change(self.FORWARD_DAYS)
                 .shift(-self.FORWARD_DAYS) > 0).astype(int)

        # 공통 인덱스 (NaN 제거)
        valid = X_all.notna().all(axis=1) & y_all.notna()
        X_all = X_all[valid]
        y_all = y_all[valid]

        n = len(X_all)
        if n < self.MIN_TRAIN_DAYS + self.FORWARD_DAYS + 10:
            # 데이터 부족 → 단순 모드
            self._simple_fit(X_all, y_all)
            return self

        # ── Walk-Forward OOF 예측 생성 ───────────────────────────────────────
        oof_preds = np.zeros((n, 3))   # [xgb, lgbm, rf] 예측 확률
        oof_y     = y_all.values.copy()

        self._scaler = RobustScaler()

        splits = []
        start = self.MIN_TRAIN_DAYS
        while start + self.WALK_STEP <= n - self.FORWARD_DAYS:
            splits.append((start, min(start + self.WALK_STEP, n - self.FORWARD_DAYS)))
            start += self.WALK_STEP

        for train_end, val_end in splits:
            X_tr = X_all.iloc[:train_end]
            y_tr = y_all.iloc[:train_end]
            X_vl = X_all.iloc[train_end:val_end]

            X_tr_s = self._scaler.fit_transform(X_tr)
            X_vl_s = self._scaler.transform(X_vl)

            # XGBoost — 튜닝됐으면 최적 파라미터, 아니면 기본값
            xgb = XGBClassifier(**self._xgb_params())
            xgb.fit(X_tr_s, y_tr)
            oof_preds[train_end:val_end, 0] = xgb.predict_proba(X_vl_s)[:, 1]

            # LightGBM
            lgbm = LGBMClassifier(
                n_estimators=300, num_leaves=31, learning_rate=0.05,
                subsample=0.8, verbose=-1,
            )
            lgbm.fit(X_tr_s, y_tr)
            oof_preds[train_end:val_end, 1] = lgbm.predict_proba(X_vl_s)[:, 1]

            # Random Forest
            rf = RandomForestClassifier(
                n_estimators=200, max_depth=6, min_samples_leaf=5,
                random_state=42, n_jobs=-1,
            )
            rf.fit(X_tr_s, y_tr)
            oof_preds[train_end:val_end, 2] = rf.predict_proba(X_vl_s)[:, 1]

        # ── 전체 데이터로 최종 모델 학습 ─────────────────────────────────────
        X_s = self._scaler.fit_transform(X_all)

        self._models["xgb"] = XGBClassifier(**self._xgb_params())
        self._models["xgb"].fit(X_s, oof_y)

        self._models["lgbm"] = LGBMClassifier(
            n_estimators=300, num_leaves=31, learning_rate=0.05,
            subsample=0.8, verbose=-1,
        )
        self._models["lgbm"].fit(X_s, oof_y)

        self._models["rf"] = RandomForestClassifier(
            n_estimators=200, max_depth=6, min_samples_leaf=5,
            random_state=42, n_jobs=-1,
        )
        self._models["rf"].fit(X_s, oof_y)

        # ── 메타 모델 (Platt Scaling으로 확률 보정) ───────────────────────────
        valid_oof = np.any(oof_preds != 0, axis=1)
        if valid_oof.sum() > 20:
            meta_lr = LogisticRegression(C=1.0, max_iter=1000)
            meta_lr.fit(oof_preds[valid_oof], oof_y[valid_oof])
            self._meta_model = meta_lr
        else:
            self._meta_model = None

        # ── 검증용 OOF 기록 (out-of-sample · 룩어헤드 없음) ──────────────────
        # walk-forward 검증 구간의 예측 확률 / 실제 레이블 / 실제 N일 수익률을
        # 저장 → validation_metrics() 가 적중률·칼리브레이션·구간수익을 계산.
        if valid_oof.sum() > 20:
            if self._meta_model is not None:
                oof_proba = self._meta_model.predict_proba(oof_preds[valid_oof])[:, 1]
            else:
                oof_proba = oof_preds[valid_oof].mean(axis=1)
            fwd_ret_all = (df["close"].pct_change(self.FORWARD_DAYS)
                           .shift(-self.FORWARD_DAYS)).reindex(X_all.index).values
            self._validation = {
                "proba": oof_proba,
                "y": oof_y[valid_oof].astype(int),
                "fwd_ret": fwd_ret_all[valid_oof],
            }
        else:
            self._validation = None

        # ── 피처 중요도 (XGBoost) ────────────────────────────────────────────
        try:
            imp = self._models["xgb"].feature_importances_
            feat_names = X_all.columns.tolist()
            top_idx    = np.argsort(imp)[-10:][::-1]
            self._feature_imp = {feat_names[i]: float(imp[i]) for i in top_idx}
        except Exception:
            self._feature_imp = {}

        # ── 레짐 감지 학습 ────────────────────────────────────────────────────
        ret = df["close"].pct_change().dropna().values
        self._regime_det.fit(ret)

        self._fitted      = True
        self._feature_cols = X_all.columns.tolist()
        return self

    def _simple_fit(self, X, y):
        """데이터 부족 시 단순 Logistic Regression"""
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import RobustScaler
        self._scaler = RobustScaler()
        X_s = self._scaler.fit_transform(X)
        lr  = LogisticRegression(C=1.0, max_iter=1000)
        lr.fit(X_s, y)
        self._models["lr"]   = lr
        self._meta_model     = None
        self._fitted         = True
        self._feature_cols   = X.columns.tolist()

    def predict(
        self,
        frame,
        value_score: float = 50.0,
        fundamentals: Optional[dict] = None,
    ) -> ModelPrediction:
        """현재 시점 예측"""
        symbol = frame.symbol if hasattr(frame, "symbol") else "UNKNOWN"
        df = frame.data.copy()
        df.index = pd.to_datetime(df.index).tz_localize(None)

        # 피처 생성 (최근 1행)
        X = build_features(df, value_score=value_score, fundamentals=fundamentals)
        X_last = X.iloc[[-1]]

        if not self._fitted or self._scaler is None:
            return self._fallback_prediction(symbol, df)

        # 모델 컬럼 정렬
        for col in self._feature_cols:
            if col not in X_last.columns:
                X_last[col] = 0.0
        X_last = X_last[self._feature_cols].fillna(0)

        X_s = self._scaler.transform(X_last)

        # 각 모델 예측
        model_probs = {}
        for name, model in self._models.items():
            try:
                prob = model.predict_proba(X_s)[0, 1]
                model_probs[name] = float(prob)
            except Exception:
                pass

        if not model_probs:
            return self._fallback_prediction(symbol, df)

        probs_arr = np.array(list(model_probs.values()))

        # 메타 모델로 최종 확률 보정
        if self._meta_model is not None:
            try:
                meta_input  = np.zeros((1, 3))
                names = ["xgb", "lgbm", "rf", "lr"]
                for i, n in enumerate(names[:3]):
                    meta_input[0, i] = model_probs.get(n, probs_arr.mean())
                up_prob = float(self._meta_model.predict_proba(meta_input)[0, 1])
            except Exception:
                up_prob = float(probs_arr.mean())
        else:
            up_prob = float(probs_arr.mean())

        # 모델 간 일치도 (낮은 분산 = 높은 일치)
        agreement = float(1.0 - np.clip(probs_arr.std() * 4, 0, 1))

        # 레짐 감지
        ret = df["close"].pct_change().dropna().values
        try:
            regime_name, regime_proba = self._regime_det.current_regime(ret)
        except Exception:
            regime_name   = "횡보장"
            regime_proba  = {"상승장": 0.33, "횡보장": 0.34, "하락장": 0.33}

        # 레짐 기반 신뢰도 조정
        # 상승장에서 매수 신호 → 신뢰도 up / 하락장에서 매수 → 신뢰도 down
        regime_boost = 0.0
        if regime_name == "상승장" and up_prob > 0.5:
            regime_boost = 0.05
        elif regime_name == "하락장" and up_prob < 0.5:
            regime_boost = 0.05
        elif regime_name in ["상승장", "하락장"] and (
            (regime_name == "상승장" and up_prob < 0.5) or
            (regime_name == "하락장" and up_prob > 0.5)
        ):
            regime_boost = -0.05

        # 펀더멘털 틸트 — regime_boost 와 같은 방식의 사후 확률 보정.
        # 단일 종목 시계열에서 스칼라 펀더멘털 피처는 분산이 0이라 트리가 무시하므로,
        # PEG·지속ROE·FCF 기반 종합 점수를 베이지안 프라이어처럼 확률에 가산한다.
        fund_tilt = self._fundamental_tilt(fundamentals)
        up_prob = float(np.clip(up_prob + fund_tilt, 0.02, 0.98))

        # 최종 신뢰도
        raw_confidence  = abs(up_prob - 0.5) * 2   # 불확실도를 신뢰도로 변환
        confidence      = float(np.clip(raw_confidence * agreement + regime_boost, 0.1, 0.95))

        # 신호 결정 (레짐 조건 추가)
        if up_prob > 0.55 and regime_name != "하락장":
            signal = "매수"
        elif up_prob < 0.45 and regime_name != "상승장":
            signal = "매도"
        else:
            signal = "중립"

        component_signals = {k: f"{v:.1%}" for k, v in model_probs.items()}
        if fund_tilt != 0.0:
            component_signals["펀더멘털틸트"] = f"{fund_tilt:+.1%}"

        return ModelPrediction(
            symbol          = symbol,
            regime          = regime_name,
            regime_proba    = regime_proba,
            signal          = signal,
            up_probability  = up_prob,
            confidence      = confidence,
            feature_importance = self._feature_imp,
            component_signals  = component_signals,
            model_agreement = agreement,
            fundamental_tilt = fund_tilt,
        )

    @staticmethod
    def _fundamental_tilt(fundamentals: Optional[dict]) -> float:
        """펀더멘털 종합 점수(0~100)를 ±0.06 범위의 확률 틸트로 변환."""
        if not fundamentals:
            return 0.0
        try:
            from app.ml.features.fundamental_features import fundamental_score
            score = fundamental_score(fundamentals)          # 0~100, 50=중립
            return float(np.clip((score - 50.0) / 50.0 * 0.06, -0.06, 0.06))
        except Exception:
            return 0.0

    def _fallback_prediction(self, symbol: str, df: pd.DataFrame) -> ModelPrediction:
        """모델 미학습 시 기술적 지표 기반 단순 예측"""
        c    = df["close"].values
        ret  = np.diff(c) / c[:-1]
        rsi  = _rsi(c, 14)[-1]
        ma20 = c[-20:].mean() if len(c) >= 20 else c.mean()
        ma50 = c[-50:].mean() if len(c) >= 50 else c.mean()

        score = 0.5
        if rsi < 30:   score += 0.15
        elif rsi > 70: score -= 0.15
        if c[-1] > ma20 > ma50: score += 0.1
        elif c[-1] < ma20 < ma50: score -= 0.1

        score = float(np.clip(score, 0.2, 0.8))
        return ModelPrediction(
            symbol          = symbol,
            regime          = "횡보장",
            regime_proba    = {"상승장": 0.33, "횡보장": 0.34, "하락장": 0.33},
            signal          = "매수" if score > 0.55 else "매도" if score < 0.45 else "중립",
            up_probability  = score,
            confidence      = abs(score - 0.5) * 1.5,
            feature_importance = {},
            component_signals  = {"기술적분석": f"{score:.1%}"},
            model_agreement = 0.5,
        )
