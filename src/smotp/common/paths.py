"""Một nơi duy nhất quyết định đường dẫn, để cùng một code chạy được ở local, WSL và Kaggle.

Đặt biến môi trường (xem .env.example):
  SMOTP4SB_DATA  thư mục dữ liệu thô (chứa train/, pub_test/, annotations/, ...)
  SMOTP_WORK     nơi chứa cache sinh ra + runs (mặc định: ./work)
  SMOTP_MODELS   nơi chứa trọng số (mặc định: ./work/models)
"""
from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def on_kaggle() -> bool:
    return Path("/kaggle/input").exists()


def data_root() -> Path:
    v = os.environ.get("SMOTP4SB_DATA")
    if not v:
        raise RuntimeError("Chưa đặt SMOTP4SB_DATA (xem .env.example).")
    p = Path(v)
    if not p.exists():
        raise FileNotFoundError(f"SMOTP4SB_DATA không tồn tại: {p}")
    return p


def work_root() -> Path:
    p = Path(os.environ.get("SMOTP_WORK", REPO / "work"))
    p.mkdir(parents=True, exist_ok=True)
    return p


def models_root() -> Path:
    p = Path(os.environ.get("SMOTP_MODELS", work_root() / "models"))
    p.mkdir(parents=True, exist_ok=True)
    return p


def seq_dir(split: str, seq_id: str) -> Path:
    return data_root() / split / seq_id


def list_sequences(split: str) -> list[str]:
    d = data_root() / split
    return sorted(p.name for p in d.iterdir() if p.is_dir())


def derived(kind: str, config_hash: str) -> Path:
    """Cache sinh ra (tile, detection, track...). Tên kèm hash cấu hình => xoá/tạo lại được."""
    p = work_root() / "derived" / kind / config_hash
    p.mkdir(parents=True, exist_ok=True)
    return p
