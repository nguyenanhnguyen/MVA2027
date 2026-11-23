"""Định dạng file nộp của cả hai track + kiểm tra nhanh.

Theo trang Codabench (cần đối chiếu với tools/validate_submission.py của repo baseline khi có repo):
  Track 1 và tracking/ của Track 2:  frame_id, track_id, x, y, w, h, conf    (frame từ 1, x,y = góc trên-trái)
  prediction/ của Track 2:           window_start, frame_id, track_id, mode_k, cx, cy
"""
from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Iterable, Sequence

OBS, HOR, K = 60, 150, 6        # frame quan sát, frame dự đoán, số mode
WINDOW = OBS + HOR              # 210
STRIDE = 150                    # window_start = 1, 151, 301, ...


def window_starts(n_frames: int) -> list[int]:
    """Cửa sổ 210 frame phải nằm trọn trong sequence."""
    return [ws for ws in range(1, n_frames + 1, STRIDE) if ws + WINDOW - 1 <= n_frames]


def _write(path, lines):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes("".join(lines).encode())   # ghi bytes để luôn dùng "\n", kể cả trên Windows


def write_mot(path, rows: Iterable[Sequence[float]]) -> None:
    _write(path, (f"{int(f)},{int(t)},{x:.2f},{y:.2f},{w:.2f},{h:.2f},{c:.4f}\n" for f, t, x, y, w, h, c in rows))


def write_pred(path, rows: Iterable[Sequence[float]]) -> None:
    _write(path, (f"{int(ws)},{int(f)},{int(t)},{int(k)},{cx:.2f},{cy:.2f}\n" for ws, f, t, k, cx, cy in rows))


def _read(path, kinds):
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(tuple(k(v) for k, v in zip(kinds, line.split(","))))
    return out


def read_mot(path):
    return _read(path, (int, int, float, float, float, float, float))


def read_pred(path):
    return _read(path, (int, int, int, int, float, float))


def check_pred(pred_rows, tracking_rows, n_frames: int) -> list[str]:
    """Bắt các lỗi hay gặp trước khi chạy validator chính thức. Rỗng = ổn."""
    errs, seen = set(), set()
    valid_ws = set(window_starts(n_frames))
    tracked = {r[1] for r in tracking_rows}
    frames_of_mode: dict[tuple, dict[int, set]] = {}
    for ws, fr, tid, k, _, _ in pred_rows:
        if ws not in valid_ws:
            errs.add(f"window_start {ws} không hợp lệ")
        if not ws + OBS <= fr <= ws + WINDOW - 1:
            errs.add(f"frame {fr} ngoài tầm dự đoán của cửa sổ {ws}")
        if not 0 <= k < K:
            errs.add(f"mode_k {k} ngoài 0..{K - 1}")
        if (ws, fr, tid, k) in seen:
            errs.add(f"dòng trùng {(ws, fr, tid, k)}")
        seen.add((ws, fr, tid, k))
        if tid not in tracked:
            errs.add(f"track_id {tid} không có trong tracking/ (sẽ không được chấm)")
        frames_of_mode.setdefault((ws, tid), {}).setdefault(k, set()).add(fr)
    for key, modes in frames_of_mode.items():     # validator chính thức cũng cảnh báo điều này
        if len(modes) != K or len({frozenset(v) for v in modes.values()}) > 1:
            errs.add(f"cảnh báo: {key} thiếu mode hoặc các mode phủ frame khác nhau")
    return sorted(errs)


def zip_dir(src: str | Path, out_zip: str | Path) -> Path:
    """Nén nội dung thư mục (không có thư mục bọc ngoài)."""
    src, out_zip = Path(src), Path(out_zip)
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(src.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(src).as_posix())
    return out_zip
