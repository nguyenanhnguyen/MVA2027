"""Tracker một lõi dùng cho cả hai track (Track 1: cả sequence; Track 2: từng cửa sổ 60 frame).

Thiết kế theo YOLOv8-SMOT (arXiv 2507.12087): OC-SORT/ByteTrack không dùng đặc trưng ngoại hình, thêm
  (1) EMA cho vận tốc (hướng chuyển động), (2) mở rộng bbox trước khi tính IoU, (3) phạt khoảng cách tâm kiểu DIoU:
      sim = (IoU_mở_rộng - khoảng_cách_chuẩn_hóa + 1) / 2
Đây là BẢN VIẾT LẠI theo mô tả của paper (công thức chuẩn hóa khoảng cách paper không nêu rõ; ở đây dùng d²/c² của
hộp bao như DIoU). Chưa được đối chiếu số liệu với bản gốc: phải tái lập ablation trước khi tin vào tham số.
Chạy offline trên cả đoạn: lọc track ngắn và nội suy khoảng trống sau khi xong.
"""
from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np
from scipy.optimize import linear_sum_assignment

Tracks = dict[int, dict[int, tuple]]     # {track_id: {frame: (x, y, w, h, conf)}}  (x,y = góc trên-trái)


@dataclass
class TrackCfg:
    hi: float = 0.25            # ngưỡng conf giai đoạn 1
    lo: float = 0.05            # conf thấp nhất cho giai đoạn 2 (kiểu ByteTrack)
    new: float = 0.25           # conf tối thiểu để mở track mới
    sim_thr: float = 0.25
    sim_thr_low: float = 0.17   # giai đoạn 2: ngưỡng thấp hơn 0.08 như paper
    ema: float = 0.8
    expand: float = 2.0
    max_age: int = 30
    min_len: int = 5
    min_mean_conf: float = 0.0
    interp_gap: int = 10        # nội suy khoảng trống tối đa (frame); 0 = tắt

    @classmethod
    def from_dict(cls, d: dict) -> "TrackCfg":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})


def similarity(a: np.ndarray, b: np.ndarray, expand: float) -> np.ndarray:
    """a (M,4), b (N,4) dạng cx,cy,w,h -> ma trận (M,N): (IoU_mở_rộng - d²/c² + 1) / 2."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    aw, ah = a[:, 2:3] * expand, a[:, 3:4] * expand
    bw, bh = (b[:, 2] * expand)[None], (b[:, 3] * expand)[None]
    ax1, ay1, ax2, ay2 = a[:, 0:1] - aw / 2, a[:, 1:2] - ah / 2, a[:, 0:1] + aw / 2, a[:, 1:2] + ah / 2
    bx1, by1, bx2, by2 = b[:, 0][None] - bw / 2, b[:, 1][None] - bh / 2, b[:, 0][None] + bw / 2, b[:, 1][None] + bh / 2
    inter = np.clip(np.minimum(ax2, bx2) - np.maximum(ax1, bx1), 0, None) * \
        np.clip(np.minimum(ay2, by2) - np.maximum(ay1, by1), 0, None)
    iou = inter / (aw * ah + bw * bh - inter + 1e-9)
    c2 = (np.maximum(ax2, bx2) - np.minimum(ax1, bx1)) ** 2 + (np.maximum(ay2, by2) - np.minimum(ay1, by1)) ** 2
    d2 = (a[:, 0:1] - b[:, 0][None]) ** 2 + (a[:, 1:2] - b[:, 1][None]) ** 2
    return (iou - d2 / (c2 + 1e-9) + 1) / 2


def _assign(sim: np.ndarray, thr: float):
    """Hungarian một-một; trả (cặp khớp, hàng chưa khớp, cột chưa khớp)."""
    if sim.size == 0:
        return [], list(range(sim.shape[0])), list(range(sim.shape[1]))
    r, c = linear_sum_assignment(-sim)
    m = [(i, j) for i, j in zip(r, c) if sim[i, j] >= thr]
    mi, mj = {i for i, _ in m}, {j for _, j in m}
    return m, [i for i in range(sim.shape[0]) if i not in mi], [j for j in range(sim.shape[1]) if j not in mj]


def _cxcywh(d: np.ndarray) -> np.ndarray:
    return np.c_[d[:, 0] + d[:, 2] / 2, d[:, 1] + d[:, 3] / 2, d[:, 2], d[:, 3]] if len(d) else np.zeros((0, 4))


class _T:
    def __init__(self, tid: int, f: int, det: np.ndarray):
        self.id, self.last, self.age, self.v = tid, f, 0, np.zeros(2)
        self.hist: dict[int, tuple] = {}
        self.c, self.wh = np.array([det[0] + det[2] / 2, det[1] + det[3] / 2]), det[2:4].copy()
        self.hist[f] = tuple(det)
        self.fresh = True

    def predict(self, f: int, max_age: int) -> np.ndarray:
        return np.r_[self.c + self.v * min(f - self.last, max_age), self.wh]

    def update(self, f: int, det: np.ndarray, ema: float):
        c = np.array([det[0] + det[2] / 2, det[1] + det[3] / 2])
        inst = (c - self.c) / max(f - self.last, 1)
        self.v = inst if self.fresh else ema * self.v + (1 - ema) * inst
        self.fresh, self.c, self.wh, self.last, self.age = False, c, det[2:4].copy(), f, 0
        self.hist[f] = tuple(det)


def track(dets: dict[int, np.ndarray], frames, cfg: TrackCfg) -> Tracks:
    """dets: {frame: (n,5) x,y,w,h,conf}. frames: các frame theo thứ tự. Trả về Tracks."""
    live: list[_T] = []
    done: list[_T] = []
    nid = 1
    for f in frames:
        d = dets.get(f)
        d = np.zeros((0, 5)) if d is None else np.asarray(d, dtype=float).reshape(-1, 5)
        hi, lo = d[d[:, 4] >= cfg.hi], d[(d[:, 4] >= cfg.lo) & (d[:, 4] < cfg.hi)]
        pred = np.array([t.predict(f, cfg.max_age) for t in live]).reshape(-1, 4)
        m1, ut, ud = _assign(similarity(pred, _cxcywh(hi), cfg.expand), cfg.sim_thr)
        m2, ut2, _ = _assign(similarity(pred[ut], _cxcywh(lo), cfg.expand), cfg.sim_thr_low)
        for i, j in m1:
            live[i].update(f, hi[j], cfg.ema)
        for i, j in m2:
            live[ut[i]].update(f, lo[j], cfg.ema)
        for i in (ut[k] for k in ut2):
            live[i].age += 1
        for j in ud:
            if hi[j, 4] >= cfg.new:
                live.append(_T(nid, f, hi[j]))
                nid += 1
        done += [t for t in live if t.age > cfg.max_age]
        live = [t for t in live if t.age <= cfg.max_age]
    return _finalize(done + live, cfg)


def _finalize(tracks: list[_T], cfg: TrackCfg) -> Tracks:
    out: Tracks = {}
    for t in sorted(tracks, key=lambda t: min(t.hist)):
        if len(t.hist) < cfg.min_len or np.mean([h[4] for h in t.hist.values()]) < cfg.min_mean_conf:
            continue
        h = dict(t.hist)
        fr = sorted(h)
        for a, b in zip(fr, fr[1:]):
            gap = b - a - 1
            if 0 < gap <= cfg.interp_gap:
                for k in range(1, gap + 1):
                    w = k / (gap + 1)
                    h[a + k] = tuple((1 - w) * np.array(h[a]) + w * np.array(h[b]))
        out[len(out) + 1] = h
    return out


def to_rows(tracks: Tracks, id_offset: int = 0) -> list[tuple]:
    """Tracks -> các dòng MOT (frame, id, x, y, w, h, conf) sắp theo frame."""
    return sorted((f, tid + id_offset, *b) for tid, tr in tracks.items() for f, b in tr.items())
