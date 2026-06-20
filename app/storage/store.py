"""
=============================================================================
로컬 JSON 영속 저장 — 세션 간 데이터 유지 (모의 포트폴리오 등)
=============================================================================
프로젝트 루트의 `.data/` 디렉터리에 JSON 파일로 저장한다.
  - `.data/` 는 .gitignore 에 등록 → 깃에 올라가지 않음.
  - 단일 사용자 로컬 실행 전제(파일 락 없음). 동시 쓰기는 가정하지 않음.

⚠️ Streamlit Cloud 등 컨테이너 환경은 파일시스템이 재시작 시 초기화되므로
   영속이 보장되지 않는다(로컬 실행에서만 안정적). 데이터 손실 방지가 중요하면
   외부 DB로 교체할 것.

사용:
    from app.storage import load_json, save_json
    positions = load_json("portfolio.json", default=[])
    save_json("portfolio.json", positions)
=============================================================================
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

# 프로젝트 루트(.../클로드가 만드는 마음대로 투자 머신)의 .data/
_DATA_DIR = Path(__file__).resolve().parents[2] / ".data"


def data_path(filename: str) -> Path:
    """`.data/<filename>` 절대 경로. 디렉터리는 필요 시 생성."""
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    # 경로 탈출 방지 — 파일명만 사용
    return _DATA_DIR / os.path.basename(filename)


def load_json(filename: str, default: Any = None) -> Any:
    """JSON 로드. 파일이 없거나 손상되면 default 반환."""
    path = data_path(filename)
    if not path.exists():
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return default


def save_json(filename: str, data: Any) -> bool:
    """JSON 저장(원자적 쓰기). 성공 여부 반환."""
    path = data_path(filename)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)   # 원자적 교체
        return True
    except OSError:
        return False
