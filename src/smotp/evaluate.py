"""Đánh giá. Hai lớp:
  * `proxy_tp`: ước lượng NHANH để chọn tham số (vectorized, không cần repo baseline). Chỉ là DetA của SO-HOTA-TP
    với đầu vào đã khớp ID; KHÔNG gồm SO-AssA-TP và việc khớp ID bằng tracker => không so được với leaderboard.
  * `run_official`: gọi evaluator chính thức trong third_party (nguồn chân lý cho số liệu báo cáo).
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

from .data import REPO, S_MEAN
from .formats import HOR

ALPHAS = np.round(np.arange(0.05, 0.96, 0.05), 2)


def dotd(d, S: float = S_MEAN):
    return np.exp(-np.asarray(d) / S)


def proxy_tp(pos: np.ndarray, emit: np.ndarray, fut: np.ndarray, S: float = S_MEAN) -> dict:
    """pos (N,K,H,2) px, emit (N,H) bool: frame có phát dự đoán, fut (N,H,2) px (NaN = chim không còn thấy).
    Quy tắc chấm theo trang Evaluation: chọn mode có DotD trung bình cao nhất trên frame còn thấy (frame không phát
    tính 0); với mỗi alpha: thấy+phát+đủ gần = TP; thấy+phát+xa = FN+FP; thấy+không phát = FN; mất+phát = FP."""
    vis = ~np.isnan(fut[..., 0])
    sim = dotd(np.linalg.norm(pos - np.nan_to_num(fut)[:, None], axis=-1), S) * (vis & emit)[:, None]
    best = sim[np.arange(len(sim)), (sim.sum(-1) / np.maximum(vis.sum(-1), 1)[:, None]).argmax(1)]   # (N,H)
    deta = []
    for a in ALPHAS:
        tp = (vis & emit & (best >= a)).sum()
        fp = (emit & ~(vis & (best >= a))).sum()
        fn = (vis & ~(emit & (best >= a))).sum()
        deta.append(tp / max(tp + fn + fp, 1))
    d = np.linalg.norm(pos - np.nan_to_num(fut)[:, None], axis=-1)
    ade = np.where(vis[:, None], d, 0).sum(-1) / np.maximum(vis.sum(-1), 1)[:, None]
    return {"proxy_deta_tp": float(np.mean(deta)), "minADE6_px": float(ade.min(1).mean()),
            "n_windows": int(len(pos)), "emit_rate": float(emit.mean())}


def run_official(kind: str, **kw) -> None:
    """kind='t1': gt_dir, tracker_dir, seqmap | kind='t2': pred, gt, tracks, output_json.
    Cú pháp lấy từ trang Codabench/README repo baseline; kiểm tra lại bằng `--help` khi có repo."""
    cmd = [sys.executable, "-m"]
    cmd += (["evaluation.eval_tracking", "--gt-dir", kw["gt_dir"], "--tracker-dir", kw["tracker_dir"], "--seqmap", kw["seqmap"]]
            if kind == "t1" else
            ["evaluation.so_hota_tp", "--pred", kw["pred"], "--gt", kw["gt"], "--tracks", kw["tracks"],
             "--output-json", kw["output_json"]])
    subprocess.run(cmd, cwd=Path(REPO) / "third_party" / "MVA2027-SMOTP4SB", check=True)
