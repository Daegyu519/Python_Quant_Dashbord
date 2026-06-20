"""
=============================================================================
딥러닝 확률 모델 — MPS(Apple Silicon) 우선 PyTorch 아키텍처
=============================================================================
프로젝트 하드웨어 정책 (claude.md):
  - 추론/대시보드(기본): Apple Silicon 로컬 실행 → `mps` 를 명시적 기본값으로.
  - 학습(조건부): 클라우드 GPU 가정 → `cuda` 우선, 로컬 검증 시 `mps`/`cpu` 폴백.

구성:
  resolve_device()        디바이스 결정 (추론=mps 우선 / 학습=cuda 우선)
  ProbabilisticMLP        드롭아웃 MLP — 로짓 출력
  mc_dropout_predict()    MC-Dropout 으로 상승확률 평균 + 불확실성(표준편차)
  train_probability_model() 디바이스 불문(agnostic) 학습 루프

torch 는 무거워서 requirements-full.txt 에서도 선택 설치 항목이다.
미설치 환경에서도 패키지 임포트가 깨지지 않도록 torch 임포트는 함수/클래스
내부에서 지연(lazy) 처리한다.

주의: MPS 백엔드는 float64 미지원 → 모든 텐서는 float32 로 캐스팅한다.

사용 예 (improved_ensemble 의 L1 모델로 추가 가능):
    from app.ml.models.deep_model import (
        resolve_device, ProbabilisticMLP, train_probability_model, mc_dropout_predict)

    model = train_probability_model(X_train, y_train)          # cuda→mps→cpu
    p_up, sigma = mc_dropout_predict(model, X_last)            # mps 추론 (기본)
=============================================================================
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:                       # 타입 힌트 전용 — 런타임 임포트 없음
    import torch
    from torch import nn


def _require_torch():
    """torch 지연 임포트. 미설치 시 설치 안내와 함께 명확히 실패."""
    try:
        import torch
        return torch
    except ImportError as e:
        raise ImportError(
            "PyTorch가 설치되어 있지 않습니다. 딥러닝 모듈은 선택 사항입니다.\n"
            "  Apple Silicon: pip install torch  (MPS 지원 내장)"
        ) from e


def resolve_device(training: bool = False) -> "torch.device":
    """
    실행 디바이스 결정.

    training=False (기본, 추론/대시보드):
        mps → cuda → cpu  — 로컬 Apple Silicon 에서 항상 MPS 가 기본.
    training=True (조건부 학습):
        cuda → mps → cpu  — 클라우드 GPU(P100 등) 우선, 로컬 검증 폴백.
    """
    torch = _require_torch()
    if training:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    # 추론 기본값: Metal Performance Shaders
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _build_mlp_class():
    """nn.Module 서브클래스를 지연 생성 (torch 미설치 환경 보호)."""
    torch = _require_torch()
    from torch import nn

    class ProbabilisticMLP(nn.Module):
        """
        드롭아웃 기반 확률 MLP.

        - 출력: 로짓 1개 → sigmoid 로 상승확률.
        - 추론 시에도 드롭아웃을 켜서(MC-Dropout) N회 샘플링하면
          확률의 평균(점추정)과 표준편차(모형 불확실성)를 함께 얻는다.
          → 이진 점예측이 아닌 분포 기반 확률 출력.
        """

        def __init__(self, input_dim: int, hidden: tuple[int, ...] = (64, 32),
                     dropout: float = 0.3):
            super().__init__()
            layers: list[nn.Module] = []
            prev = input_dim
            for h in hidden:
                layers += [nn.Linear(prev, h), nn.GELU(), nn.Dropout(dropout)]
                prev = h
            layers.append(nn.Linear(prev, 1))
            self.net = nn.Sequential(*layers)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return self.net(x).squeeze(-1)

    return ProbabilisticMLP


def __getattr__(name: str):
    """`from deep_model import ProbabilisticMLP` 지원 (PEP 562 지연 로딩)."""
    if name == "ProbabilisticMLP":
        return _build_mlp_class()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def train_probability_model(
    X: np.ndarray,
    y: np.ndarray,
    hidden: tuple[int, ...] = (64, 32),
    dropout: float = 0.3,
    epochs: int = 60,
    lr: float = 1e-3,
    batch_size: int = 128,
    device: Optional["torch.device"] = None,
):
    """
    디바이스 불문 학습 루프. device 미지정 시 학습 정책(cuda→mps→cpu) 적용.

    Args:
        X: (n_samples, n_features) — RobustScaler 등으로 스케일링된 피처
        y: (n_samples,) — 0/1 레이블 (N일 후 상승 여부)
    Returns:
        학습 완료된 ProbabilisticMLP (eval 모드, 지정 디바이스 상주)
    """
    torch = _require_torch()
    from torch import nn

    dev = device or resolve_device(training=True)
    mlp_cls = _build_mlp_class()
    model = mlp_cls(input_dim=X.shape[1], hidden=hidden, dropout=dropout).to(dev)

    # MPS 는 float64 미지원 → float32 강제
    Xt = torch.as_tensor(np.asarray(X, dtype=np.float32), device=dev)
    yt = torch.as_tensor(np.asarray(y, dtype=np.float32), device=dev)

    # 클래스 불균형 보정 (상승/하락 비율)
    pos = float(yt.mean().item())
    pos_weight = torch.tensor([(1 - pos) / max(pos, 1e-6)], device=dev)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    model.train()
    n = len(Xt)
    for _ in range(epochs):
        perm = torch.randperm(n, device=dev)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            optimizer.zero_grad()
            loss = criterion(model(Xt[idx]), yt[idx])
            loss.backward()
            optimizer.step()

    model.eval()
    return model


def mc_dropout_predict(
    model,
    X: np.ndarray,
    n_samples: int = 30,
    device: Optional["torch.device"] = None,
) -> tuple[float, float]:
    """
    MC-Dropout 확률 추론 — 추론 기본 디바이스는 MPS.

    드롭아웃을 켠 채 n_samples 회 forward → sigmoid 확률 분포에서
    (평균 상승확률, 표준편차=모형 불확실성)을 반환한다.
    표준편차가 크면 모델 스스로 확신이 낮다는 뜻 → 신뢰도 할인에 사용.
    """
    torch = _require_torch()

    dev = device or resolve_device(training=False)   # 기본 = mps
    model = model.to(dev)
    Xt = torch.as_tensor(np.asarray(X, dtype=np.float32), device=dev)
    if Xt.ndim == 1:
        Xt = Xt.unsqueeze(0)

    model.train()                                    # 드롭아웃 활성화 (MC-Dropout)
    probs: list[float] = []
    with torch.no_grad():
        for _ in range(n_samples):
            p = torch.sigmoid(model(Xt)).mean()
            probs.append(float(p.item()))
    model.eval()

    arr = np.asarray(probs)
    return float(arr.mean()), float(arr.std())
