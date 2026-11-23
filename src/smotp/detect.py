"""Phát hiện chim: tạo tile SliceTrain, huấn luyện YOLOv8 (Ultralytics), suy luận full-frame, cache detection.

Chỉ file này cần ultralytics/GPU; import trễ để phần còn lại chạy được trên máy không có chúng.
Lưu ý giấy phép: Ultralytics là AGPL-3.0 (người nhận giải vẫn phải công khai code theo luật cuộc thi).
"""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import numpy as np

from .data import content_key, data_root, derived, frames_of, image_size, load_gt, models_root


# ================================ SliceTrain: tile huấn luyện ====================================
def tile_grid(W: int, H: int, tile: int, overlap: float) -> list[tuple[int, int, int, int]]:
    """Lưới tile (x, y, w, h) phủ kín ảnh, chồng nhau `overlap`."""
    stride = max(1, int(tile * (1 - overlap)))

    def axis(L):
        if L <= tile:
            return [0]
        xs = list(range(0, L - tile + 1, stride))
        return xs + [L - tile] if xs[-1] + tile < L else xs

    return [(x, y, min(tile, W - x), min(tile, H - y)) for y in axis(H) for x in axis(W)]


def boxes_in_tile(boxes: np.ndarray, tx: int, ty: int, tw: int, th: int, min_vis: float = 0.5) -> np.ndarray:
    """boxes (N,4) xywh trong ảnh -> các hộp (xywh, tọa độ tile) còn nhìn thấy >= min_vis diện tích."""
    if len(boxes) == 0:
        return boxes.reshape(0, 4)
    x1, y1 = np.maximum(boxes[:, 0], tx), np.maximum(boxes[:, 1], ty)
    x2, y2 = np.minimum(boxes[:, 0] + boxes[:, 2], tx + tw), np.minimum(boxes[:, 1] + boxes[:, 3], ty + th)
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    keep = (boxes[:, 2] * boxes[:, 3] > 0) & (inter >= min_vis * boxes[:, 2] * boxes[:, 3])
    return np.c_[x1 - tx, y1 - ty, x2 - x1, y2 - y1][keep]


def make_tiles(cfg: dict, folds: dict[str, int], val_fold: int, root: Path | None = None) -> Path:
    """Cắt frame train thành tile 1280 (chồng 20%), giữ tile có chim + một tỷ lệ tile nền; ghi theo định dạng YOLO.
    Lấy mẫu thưa theo thời gian (frame_step) vì frame liên tiếp gần như trùng nhau. Trả về đường dẫn data.yaml."""
    from PIL import Image
    d = cfg["det"]
    out = derived("tiles", hashlib.sha1(json.dumps([d["tile"], d["overlap"], d["frame_step"], d["val_frame_step"],
                                                    d["neg_ratio"], val_fold]).encode()).hexdigest()[:8])
    if (out / "data.yaml").exists():
        return out / "data.yaml"
    root = Path(root) if root else data_root()
    rng, gt = random.Random(0), load_gt(root)
    for seq, tracks in gt.items():
        part = "val" if folds[seq] == val_fold else "train"
        step = d["val_frame_step"] if part == "val" else d["frame_step"]
        (out / "images" / part).mkdir(parents=True, exist_ok=True)
        (out / "labels" / part).mkdir(parents=True, exist_ok=True)
        by_frame: dict[int, list] = {}
        for tr in tracks.values():
            for f, b in tr.items():
                by_frame.setdefault(f, []).append(b)
        for p in frames_of(root / "train" / seq)[::step]:
            f = int(p.stem)
            boxes = np.array(by_frame.get(f, []), dtype=float).reshape(-1, 4)
            with Image.open(p) as im:
                im.load()
                grid = tile_grid(*im.size, d["tile"], d["overlap"])
                tl = [(g, boxes_in_tile(boxes, *g)) for g in grid]
                pos = [t for t in tl if len(t[1])]
                neg = [t for t in tl if not len(t[1])]
                chosen = pos + rng.sample(neg, min(len(neg), int(np.ceil(d["neg_ratio"] * max(len(pos), 1)))))
                for (x, y, w, h), bx in chosen:
                    name = f"{seq}_{f:05d}_{x}_{y}"
                    im.crop((x, y, x + w, y + h)).save(out / "images" / part / f"{name}.jpg", quality=95)
                    (out / "labels" / part / f"{name}.txt").write_text("".join(
                        f"0 {(b[0] + b[2] / 2) / w:.6f} {(b[1] + b[3] / 2) / h:.6f} {b[2] / w:.6f} {b[3] / h:.6f}\n"
                        for b in bx))
    (out / "data.yaml").write_text(f"path: {out.as_posix()}\ntrain: images/train\nval: images/val\nnames:\n  0: bird\n")
    return out / "data.yaml"


def train(cfg: dict, data_yaml: Path, val_fold: int, project: Path | None = None) -> Path:
    """Huấn luyện YOLOv8 trên tile. Đây là bản Ultralytics của công thức repo baseline (AdamW, cosine, Mosaic tắt ở
    các epoch cuối); repo gốc dùng mmyolo nên kết quả chưa chắc trùng. Trả về đường dẫn trọng số đã chép vào models/."""
    from ultralytics import YOLO
    d = cfg["det"]
    project = project or derived("runs", "det")
    r = YOLO(d["weights"]).train(
        data=str(data_yaml), imgsz=d["imgsz"], epochs=d["epochs"], batch=d["batch"], optimizer="AdamW",
        lr0=d["lr0"], weight_decay=d["weight_decay"], cos_lr=True, close_mosaic=d["close_mosaic"],
        project=str(project), name=f"fold{val_fold}", exist_ok=True)
    best = Path(r.save_dir) / "weights" / "best.pt"
    dst = models_root() / f"det_fold{val_fold}.pt"
    dst.write_bytes(best.read_bytes())
    return dst


# ================================ suy luận + cache ===============================================
_DETECTORS: dict[str, object] = {}


def _detector(weights: str):
    if weights not in _DETECTORS:
        from ultralytics import YOLO
        _DETECTORS[weights] = YOLO(weights)
    return _DETECTORS[weights]


def _predict(weights: str, d: dict, path: Path, size: tuple[int, int]) -> np.ndarray:
    imgsz = int(np.ceil(max(size) * d["infer_scale"] / 32) * 32)       # full-frame, kích thước gốc x scale
    r = _detector(weights).predict(str(path), imgsz=imgsz, conf=d["conf"], iou=d["nms_iou"], verbose=False)[0]
    b, c = r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()
    return np.c_[b[:, 0], b[:, 1], b[:, 2] - b[:, 0], b[:, 3] - b[:, 1], c].reshape(-1, 5)


def detect_frames(paths: list[Path], cfg: dict, weights: str | Path | None = None) -> dict[int, np.ndarray]:
    """Detection cho danh sách frame -> {vị trí 1..N: mảng (n,5) x,y,w,h,conf}. Cache theo NỘI DUNG + trọng số."""
    d = cfg["det"]
    weights = str(weights or d.get("infer_weights") or models_root() / "det_final.pt")
    wid = hashlib.sha1(json.dumps([Path(weights).name, d["infer_scale"], d["conf"], d["nms_iou"]]).encode()).hexdigest()[:8]
    f = derived("det", wid) / f"{content_key(paths)}.npz"
    if f.exists():
        z = np.load(f)
        return {i + 1: a for i, a in enumerate(np.split(z["box"], np.searchsorted(z["fr"], np.arange(2, int(z["n"]) + 1))))}
    size = image_size(paths[0])
    out = {i: _predict(weights, d, p, size) for i, p in enumerate(paths, 1)}
    np.savez_compressed(f, n=len(paths), box=np.concatenate(list(out.values())),
                        fr=np.concatenate([np.full(len(a), i) for i, a in out.items()]).astype(np.int32))
    return out
