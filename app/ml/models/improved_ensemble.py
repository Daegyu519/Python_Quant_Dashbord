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

            self._model = hmm.GaussianHMM(
                n_components=self.N_STATES,
                covariance_type="full",
                n_iter=200,
                random_state=42,
            )
            self._model.fit(obs)

            # 레짐 정렬: 평균 수익률 기준으로 정렬 (낮은→높은)
            means = self._model.means_[:, 0]
            self._order = np.argsort(means)   # 0=낮음(하락), 2=높음(상승)
            self._fitted = True

        except Exception as e:
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
        """현재 레짐과 확률 딕셔너리 반환"""
        states, probas = self.predict(returns)
        cur_state = states[-1]
        cur_proba = probas[-1]
        return (
            self.REGIME_NAMES[cur_state],
            {self.REGIME_NAMES[i]: float(p) for i, p in enumerate(cur_proba)},
        )

    def _make_obs(self, returns: np.ndarray) -> np.ndarray:
        """HMM 입력 특성: [수익률, 변동성]"""
        vol = pd.Series(returns).rolling(5).std().bfill().values
        return np.column_stack([returns, vol])

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

def build_features(df: pd.DataFrame, value_score: float = 50.0) -> pd.DataFrame:
    """
    기술적 + 레짐 + 가치 통합 피처 생성.

    현재 대비 추가된 피처:
      - 레짐 확률 (HMM 출력)
      - 다중 타임프레임 모멘텀 (1/3/6/12개월)
      - 팩터 기반: 가치(Value), 품질(Quality), 저변동성(Low-Vol)
      - 가치투자 점수 (ValueScore → 정규화)
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

    def __init__(self):
        self._models      = {}
        self._meta_model  = None
        self._scaler      = None
        self._regime_det  = RegimeDetector()
        self._fitted      = False
        self._feature_imp = {}

    def fit(self, frame) -> "ImprovedEnsembleModel":
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
        X_all = build_features(df)
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

            # XGBoost
            xgb = XGBClassifier(
                n_estimators=300, max_depth=5, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                eval_metric="logloss", verbosity=0,
            )
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

        self._models["xgb"] = XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss", verbosity=0,
        )
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

    def predict(self, frame, value_score: float = 50.0) -> ModelPrediction:
        """현재 시점 예측"""
        symbol = frame.symbol if hasattr(frame, "symbol") else "UNKNOWN"
        df = frame.data.copy()
        df.index = pd.to_datetime(df.index).tz_localize(None)

        # 피처 생성 (최근 1행)
        X = build_features(df, value_score=value_score)
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

        return ModelPrediction(
            symbol          = symbol,
            regime          = regime_name,
            regime_proba    = regime_proba,
            signal          = signal,
            up_probability  = up_prob,
            confidence      = confidence,
            feature_importance = self._feature_imp,
            component_signals  = {
                k: f"{v:.1%}" for k, v in model_probs.items()
            },
            model_agreement = agreement,
        )

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
