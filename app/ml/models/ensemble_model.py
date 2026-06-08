"""
=============================================================================
ML Ensemble Model — XGBoost + LightGBM + Stacking
=============================================================================
기관급 ML 앙상블 예측 모델.
Walk-forward validation으로 데이터 누수 방지.
Optuna 하이퍼파라미터 최적화 내장.
=============================================================================
"""

from __future__ import annotations

import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import lightgbm as lgb
import optuna

from app.core.base_classes import BaseMLModel
from app.config.logging_config import get_logger, PerformanceLogger
from app.config.settings import get_settings

logger = get_logger(__name__)
settings = get_settings()

# Optuna 로그 조용히
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ─────────────────────────────────────────────
# XGBoost 예측 모델
# ─────────────────────────────────────────────
class XGBoostAlphaModel(BaseMLModel):
    """
    XGBoost 기반 알파 생성 모델.

    예측 목표:
    - 분류: 다음 N일 후 상승/하락 예측
    - 회귀: 다음 N일 후 수익률 예측

    특징:
    - GPU 지원 (tree_method='gpu_hist')
    - Early stopping
    - SHAP 피처 중요도
    - Walk-forward validation
    """

    def __init__(
        self,
        model_id: str = "xgb_alpha",
        task: str = "classification",  # or "regression"
        n_days_ahead: int = 5,
        n_estimators: int = 500,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        use_gpu: bool = False,
    ) -> None:
        super().__init__(model_id=model_id, name=f"XGBoost Alpha ({task})")
        self.task = task
        self.n_days_ahead = n_days_ahead
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.use_gpu = use_gpu and settings.ml.use_gpu

        self._model: Optional[xgb.XGBClassifier | xgb.XGBRegressor] = None
        self._threshold: float = 0.5  # 분류 임계값

    def _build_model(self, **kwargs) -> Any:
        """XGBoost 모델 생성"""
        device = "cuda" if self.use_gpu else "cpu"
        base_params = {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "device": device,
            "random_state": 42,
            "n_jobs": -1,
            **kwargs,
        }

        if self.task == "classification":
            return xgb.XGBClassifier(
                **base_params,
                objective="binary:logistic",
                eval_metric="auc",
                use_label_encoder=False,
            )
        else:
            return xgb.XGBRegressor(
                **base_params,
                objective="reg:squarederror",
                eval_metric="rmse",
            )

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        eval_set: Optional[Tuple[pd.DataFrame, pd.Series]] = None,
    ) -> "XGBoostAlphaModel":
        """
        모델 학습.

        Args:
            X: 피처 행렬 (시계열 순서 유지 필수)
            y: 타겟 (분류: 0/1, 회귀: float)
            eval_set: 검증 데이터 (early stopping용)
        """
        self._feature_names = list(X.columns)
        X_arr = X.to_numpy(dtype=np.float32)

        # NaN 처리
        X_arr = np.nan_to_num(X_arr, nan=0.0, posinf=0.0, neginf=0.0)

        with PerformanceLogger("xgboost_fit", n_samples=len(X), n_features=len(X.columns)):
            self._model = self._build_model()

            callbacks = []
            if eval_set is not None:
                X_val, y_val = eval_set
                X_val_arr = np.nan_to_num(X_val.to_numpy(dtype=np.float32))
                eval_sets = [(X_arr, y.to_numpy()), (X_val_arr, y_val.to_numpy())]
                callbacks.append(xgb.callback.EarlyStopping(
                    rounds=50, metric_name="auc" if self.task == "classification" else "rmse",
                    save_best=True
                ))
                self._model.fit(
                    X_arr, y.to_numpy(),
                    eval_set=eval_sets,
                    verbose=False,
                )
            else:
                self._model.fit(X_arr, y.to_numpy(), verbose=False)

        self._is_fitted = True

        # 성과 로깅
        if self.task == "classification":
            train_pred = self._model.predict_proba(X_arr)[:, 1]
            train_auc = roc_auc_score(y.to_numpy(), train_pred)
            logger.info(
                "xgboost_fitted",
                train_auc=round(train_auc, 4),
                n_features=len(self._feature_names),
            )

        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """예측"""
        if not self._is_fitted:
            raise RuntimeError("Must call fit() first")

        X_arr = np.nan_to_num(X.to_numpy(dtype=np.float32))

        if self.task == "classification":
            predictions = self._model.predict(X_arr)
        else:
            predictions = self._model.predict(X_arr)

        return pd.Series(predictions, index=X.index)

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        """확률 예측 (분류 모델)"""
        if not self._is_fitted:
            raise RuntimeError("Must call fit() first")

        if self.task != "classification":
            raise ValueError("predict_proba is only for classification models")

        X_arr = np.nan_to_num(X.to_numpy(dtype=np.float32))
        proba = self._model.predict_proba(X_arr)

        return pd.DataFrame(
            proba,
            index=X.index,
            columns=["prob_down", "prob_up"],
        )

    def get_feature_importance(self) -> pd.Series:
        """피처 중요도 (gain 기반)"""
        if not self._is_fitted:
            raise RuntimeError("Must call fit() first")

        importance = self._model.feature_importances_
        return pd.Series(
            importance,
            index=self._feature_names,
            name="importance",
        ).sort_values(ascending=False)

    def get_shap_values(self, X: pd.DataFrame) -> np.ndarray:
        """SHAP 값 계산 (AI 설명가능성)"""
        try:
            import shap
            explainer = shap.TreeExplainer(self._model)
            X_arr = np.nan_to_num(X.to_numpy(dtype=np.float32))
            return explainer.shap_values(X_arr)
        except ImportError:
            logger.warning("shap not installed, falling back to feature importance")
            return self._model.feature_importances_


# ─────────────────────────────────────────────
# LightGBM 예측 모델
# ─────────────────────────────────────────────
class LightGBMAlphaModel(BaseMLModel):
    """LightGBM 기반 알파 생성 모델 (XGBoost보다 빠름)"""

    def __init__(
        self,
        model_id: str = "lgb_alpha",
        task: str = "classification",
        n_estimators: int = 500,
        max_depth: int = -1,
        num_leaves: int = 63,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
    ) -> None:
        super().__init__(model_id=model_id, name=f"LightGBM Alpha ({task})")
        self.task = task
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.num_leaves = num_leaves
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self._model = None

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        eval_set: Optional[Tuple[pd.DataFrame, pd.Series]] = None,
    ) -> "LightGBMAlphaModel":
        """모델 학습"""
        self._feature_names = list(X.columns)
        X_arr = np.nan_to_num(X.to_numpy(dtype=np.float32))

        params = {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "num_leaves": self.num_leaves,
            "learning_rate": self.learning_rate,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "n_jobs": -1,
            "random_state": 42,
            "verbose": -1,
        }

        if self.task == "classification":
            self._model = lgb.LGBMClassifier(**params, objective="binary")
        else:
            self._model = lgb.LGBMRegressor(**params, objective="regression")

        callbacks = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=-1)]

        if eval_set is not None:
            X_val, y_val = eval_set
            X_val_arr = np.nan_to_num(X_val.to_numpy(dtype=np.float32))
            self._model.fit(
                X_arr, y.to_numpy(),
                eval_set=[(X_val_arr, y_val.to_numpy())],
                callbacks=callbacks,
            )
        else:
            self._model.fit(X_arr, y.to_numpy())

        self._is_fitted = True
        logger.info("lgb_fitted", n_features=len(self._feature_names))
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        X_arr = np.nan_to_num(X.to_numpy(dtype=np.float32))
        return pd.Series(self._model.predict(X_arr), index=X.index)

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        X_arr = np.nan_to_num(X.to_numpy(dtype=np.float32))
        proba = self._model.predict_proba(X_arr)
        return pd.DataFrame(proba, index=X.index, columns=["prob_down", "prob_up"])

    def get_feature_importance(self) -> pd.Series:
        return pd.Series(
            self._model.feature_importances_,
            index=self._feature_names,
        ).sort_values(ascending=False)


# ─────────────────────────────────────────────
# Stacking Ensemble
# ─────────────────────────────────────────────
class StackingAlphaModel(BaseMLModel):
    """
    Stacking 앙상블 (레이어 2 메타 모델).

    L1 모델: XGBoost + LightGBM + (선택적) LSTM
    L2 메타 모델: Logistic Regression 또는 Linear

    Walk-forward cross-validation으로 stacking 예측 생성.
    """

    def __init__(
        self,
        model_id: str = "stacking_alpha",
        task: str = "classification",
        n_splits: int = 5,
    ) -> None:
        super().__init__(model_id=model_id, name="Stacking Ensemble")
        self.task = task
        self.n_splits = n_splits

        # L1 모델 정의
        self._l1_models = {
            "xgb": XGBoostAlphaModel(task=task),
            "lgb": LightGBMAlphaModel(task=task),
        }

        # L2 메타 모델
        if task == "classification":
            from sklearn.linear_model import LogisticRegression
            self._meta_model = LogisticRegression(C=1.0, max_iter=1000)
        else:
            from sklearn.linear_model import Ridge
            self._meta_model = Ridge(alpha=1.0)

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        eval_set: Optional[Tuple[pd.DataFrame, pd.Series]] = None,
    ) -> "StackingAlphaModel":
        """
        Walk-forward stacking 학습.

        시계열 교차 검증으로 메타 피처 생성 (데이터 누수 방지).
        """
        self._feature_names = list(X.columns)
        n = len(X)

        # Walk-forward CV로 OOF(out-of-fold) 예측 생성
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        oof_predictions = np.zeros((n, len(self._l1_models)))

        with PerformanceLogger("stacking_fit", n_samples=n, n_models=len(self._l1_models)):
            for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X)):
                X_train = X.iloc[train_idx]
                y_train = y.iloc[train_idx]
                X_val = X.iloc[val_idx]

                for col_idx, (name, model) in enumerate(self._l1_models.items()):
                    model.fit(X_train, y_train)
                    if self.task == "classification":
                        preds = model.predict_proba(X_val)["prob_up"].to_numpy()
                    else:
                        preds = model.predict(X_val).to_numpy()
                    oof_predictions[val_idx, col_idx] = preds

                logger.info(
                    "stacking_fold_complete",
                    fold=fold_idx + 1,
                    train_size=len(train_idx),
                    val_size=len(val_idx),
                )

            # 전체 데이터로 L1 모델 재학습
            for model in self._l1_models.values():
                model.fit(X, y)

            # L2 메타 모델 학습 (OOF 예측 기반)
            self._meta_model.fit(oof_predictions, y.to_numpy())

        self._is_fitted = True
        logger.info("stacking_fitted")
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        l1_preds = self._get_l1_predictions(X)
        meta_preds = self._meta_model.predict(l1_preds)
        return pd.Series(meta_preds, index=X.index)

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.task != "classification":
            raise ValueError("Only for classification")
        l1_preds = self._get_l1_predictions(X)
        proba = self._meta_model.predict_proba(l1_preds)
        return pd.DataFrame(proba, index=X.index, columns=["prob_down", "prob_up"])

    def _get_l1_predictions(self, X: pd.DataFrame) -> np.ndarray:
        """L1 모델 예측 수집"""
        predictions = []
        for model in self._l1_models.values():
            if self.task == "classification":
                pred = model.predict_proba(X)["prob_up"].to_numpy()
            else:
                pred = model.predict(X).to_numpy()
            predictions.append(pred)
        return np.column_stack(predictions)

    def get_feature_importance(self) -> pd.Series:
        """L1 모델 평균 피처 중요도"""
        importances = []
        for model in self._l1_models.values():
            importances.append(model.get_feature_importance())

        avg_importance = pd.concat(importances, axis=1).mean(axis=1)
        return avg_importance.sort_values(ascending=False)


# ─────────────────────────────────────────────
# Optuna 기반 하이퍼파라미터 최적화
# ─────────────────────────────────────────────
class HyperparameterOptimizer:
    """
    Optuna 기반 자동 하이퍼파라미터 최적화.

    시계열 교차 검증으로 과적합 방지.
    """

    def __init__(
        self,
        model_class: type,
        task: str = "classification",
        n_trials: int = 100,
        n_cv_splits: int = 5,
        metric: str = "auc",
    ) -> None:
        self.model_class = model_class
        self.task = task
        self.n_trials = n_trials
        self.n_cv_splits = n_cv_splits
        self.metric = metric

    def optimize(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> dict[str, Any]:
        """
        최적 하이퍼파라미터 탐색.

        Returns:
            최적 파라미터 딕셔너리
        """
        tscv = TimeSeriesSplit(n_splits=self.n_cv_splits)

        def objective(trial: optuna.Trial) -> float:
            # 파라미터 공간 정의
            params = self._get_param_space(trial)
            cv_scores = []

            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                model = self.model_class(task=self.task, **params)
                model.fit(X_train, y_train)

                if self.task == "classification":
                    proba = model.predict_proba(X_val)["prob_up"].to_numpy()
                    score = roc_auc_score(y_val.to_numpy(), proba)
                else:
                    preds = model.predict(X_val).to_numpy()
                    score = -mean_squared_error(y_val.to_numpy(), preds, squared=False)

                cv_scores.append(score)

            return np.mean(cv_scores)

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
        )

        with PerformanceLogger("hyperparameter_optimization", n_trials=self.n_trials):
            study.optimize(
                objective,
                n_trials=self.n_trials,
                timeout=settings.ml.optuna_timeout,
                show_progress_bar=False,
            )

        logger.info(
            "optimization_completed",
            best_value=round(study.best_value, 4),
            best_params=study.best_params,
            n_trials=len(study.trials),
        )

        return study.best_params

    def _get_param_space(self, trial: optuna.Trial) -> dict[str, Any]:
        """파라미터 탐색 공간"""
        if self.model_class == XGBoostAlphaModel:
            return {
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            }
        elif self.model_class == LightGBMAlphaModel:
            return {
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                "num_leaves": trial.suggest_int("num_leaves", 20, 300),
                "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            }
        return {}
