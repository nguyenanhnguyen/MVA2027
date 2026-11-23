"""Dung chung cho CẢ HAI track: đọc/ghi và kiểm tra định dạng file nộp.

Quy ước (theo trang Codabench, cần đối chiếu lại với tools/validate_submission.py của repo baseline):
  Track 1 / tracking:  frame_id, track_id, x, y, w, h, conf       (frame đánh số từ 1, x,y = góc trên-trái)
  Track 2 / prediction: window_start, frame_id, track_id, mode_k, cx, cy
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

OBS = 60          # số frame quan sát
HOR = 150         # số frame dự đoán
WINDOW = OBS + HOR  # 210
STRIDE = 150      # window_start = 1, 151, 301, ...
K = 6             # số mode


def window_starts(n_frames: int) -> list[int]:
    """Các window_start hợp lệ: cửa sổ 210 frame phải nằm trọn trong sequence."""
    out, ws = [], 1
    while ws + WINDOW - 1 <= n_frames:
        out.append(ws)
        ws += STRIDE
    return out


def write_mot(path: str | Path, rows: Iterable[Sequence[float]]) -> None:
    """rows: (frame_id, track_id, x, y, w, h, conf). Dùng cho Track 1 và tracking/ của Track 2."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="\n") as f:
        for fr, tid, x, y, w, h, c in rows:
            f.write(f"{int(fr)},{int(tid)},{x:.2f},{y:.2f},{w:.2f},{h:.2f},{c:.4f}\n")


def read_mot(path: str | Path) -> list[tuple[int, int, float, float, float, float, float]]:
    rows = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        p = line.split(",")
        rows.append((int(p[0]), int(p[1]), *map(float, p[2:7])))
    return rows


def write_pred(path: str | Path, rows: Iterable[Sequence[float]]) -> None:
    """rows: (window_start, frame_id, track_id, mode_k, cx, cy)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="\n") as f:
        for ws, fr, tid, k, cx, cy in rows:
            f.write(f"{int(ws)},{int(fr)},{int(tid)},{int(k)},{cx:.2f},{cy:.2f}\n")


def read_pred(path: str | Path) -> list[tuple[int, int, int, int, float, float]]:
    rows = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        p = line.split(",")
        rows.append((int(p[0]), int(p[1]), int(p[2]), int(p[3]), float(p[4]), float(p[5])))
    return rows


def check_pred(pred_rows, tracking_rows, n_frames: int) -> list[str]:
    """Kiểm tra nhanh trước khi chạy validator chính thức. Trả về danh sách lỗi (rỗng = ổn)."""
    errs: list[str] = []
    valid_ws = set(window_starts(n_frames))
    seen = set()
    tracked = {(r[1]) for r in tracking_rows}  # track_id có trong tracking/
    for ws, fr, tid, k, cx, cy in pred_rows:
        if ws not in valid_ws:
            errs.append(f"window_start {ws} không hợp lệ (hợp lệ: {sorted(valid_ws)[:5]}...)")
        if not (ws + OBS <= fr <= ws + WINDOW - 1):
            errs.append(f"frame {fr} ngoài tầm dự đoán của cửa sổ {ws}")
        if not (0 <= k < K):
            errs.append(f"mode_k {k} ngoài 0..{K - 1}")
        key = (ws, fr, tid, k)
        if key in seen:
            errs.append(f"dòng trùng {key}")
        seen.add(key)
        if tid not in tracked:
            errs.append(f"track_id {tid} không có trong tracking/ (sẽ không được chấm)")
    # loại trùng thông báo
    return sorted(set(errs))
