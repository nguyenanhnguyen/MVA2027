"""Dữ liệu: đường dẫn, danh sách sequence, nhãn train, khóa nội dung, ánh xạ ID giữa hai gói, chia fold.

Biến môi trường (xem .env.example, hoặc file .env ở gốc repo, tự nạp):
  SMOTP4SB_DATA     gói dữ liệu Track 1 (chứa train/, pub_test/, annotations/)
  SMOTP4SB_DATA_T2  gói Track 2 (train giống hệt; pub_test đánh số khác). Bỏ trống = dùng chung gói 1
  SMOTP_WORK        cache sinh ra, tile, runs        (mặc định ./work)
  SMOTP_MODELS      trọng số                          (mặc định <work>/models)
"""
from __future__ import annotations

import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
S_MEAN = 24.15            # kích thước bbox trung bình (trang Evaluation); evaluator chính thức tính lại trên split
FRAME_4K = (3840, 2160)

# ---- nạp .env (không ghi đè biến đã có) -------------------------------------------------------
_env = REPO / ".env"
if _env.exists():
    for _l in _env.read_text(encoding="utf-8").splitlines():
        if "=" in _l and not _l.lstrip().startswith("#"):
            _k, _v = _l.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())


# ---- đường dẫn --------------------------------------------------------------------------------
def data_root(track: int = 1) -> Path:
    var = "SMOTP4SB_DATA_T2" if track == 2 and os.environ.get("SMOTP4SB_DATA_T2") else "SMOTP4SB_DATA"
    v = os.environ.get(var)
    if not v:
        raise RuntimeError(f"Chưa đặt biến môi trường {var} (xem .env.example)")
    p = Path(v)
    if not p.exists():
        raise FileNotFoundError(f"{var} không tồn tại: {p}")
    return p


def _dir(var: str, default: Path) -> Path:
    p = Path(os.environ.get(var, default))
    p.mkdir(parents=True, exist_ok=True)
    return p


def work_root() -> Path:
    return _dir("SMOTP_WORK", REPO / "work")


def models_root() -> Path:
    return _dir("SMOTP_MODELS", work_root() / "models")


def derived(kind: str, key: str) -> Path:
    """Cache sinh ra, tên kèm hash cấu hình => xóa và tạo lại được."""
    p = work_root() / "derived" / kind / key
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---- sequence ---------------------------------------------------------------------------------
def list_sequences(split: str, track: int = 1) -> list[str]:
    d = data_root(track) / split
    return sorted(p.name for p in d.iterdir() if p.is_dir())


def frames_of(seq_dir: Path) -> list[Path]:
    """Các frame JPEG theo thứ tự; frame_id = số trong tên file (đánh số từ 1, liên tục)."""
    fs = sorted(Path(seq_dir).glob("*.jpg"), key=lambda p: int(p.stem))
    if fs and (int(fs[0].stem) != 1 or int(fs[-1].stem) != len(fs)):
        raise ValueError(f"{seq_dir}: frame không liên tục từ 00001 ({fs[0].name}..{fs[-1].name}, {len(fs)} file)")
    return fs


def image_size(path: Path) -> tuple[int, int]:
    from PIL import Image
    with Image.open(path) as im:
        return im.size            # (W, H)


def content_key(paths: list[Path]) -> str:
    """Khóa theo NỘI DUNG (số frame + byte của frame đầu/giữa/cuối), không phụ thuộc seq_id.
    Dùng để cache detection dùng chung giữa hai gói có đánh số khác nhau."""
    h = hashlib.sha1(str(len(paths)).encode())
    for i in sorted({0, len(paths) // 2, len(paths) - 1}):
        h.update(Path(paths[i]).read_bytes())
    return h.hexdigest()[:16]


def idmap(root_a: Path, root_b: Path, split: str = "pub_test") -> dict:
    """So hai gói theo nội dung: {'a_to_b': {...}, 'only_a': [...], 'only_b': [...]}."""
    def keys(root):
        return {content_key(frames_of(d)): d.name for d in sorted((Path(root) / split).iterdir()) if d.is_dir()}
    ka, kb = keys(root_a), keys(root_b)
    return {"a_to_b": {ka[k]: kb[k] for k in ka if k in kb},
            "only_a": sorted(ka[k] for k in ka if k not in kb),
            "only_b": sorted(kb[k] for k in kb if k not in ka)}


# ---- nhãn train -------------------------------------------------------------------------------
def load_gt(root: Path | None = None) -> dict[str, dict[int, dict[int, tuple]]]:
    """annotations/train.json (COCO-MOT) -> {seq_id: {track_id: {frame_id: (x, y, w, h)}}}.
    Giả định (CẦN KIỂM TRA khi có dữ liệu): file_name dạng '.../{seq_id}/{frame}.jpg'; mỗi ảnh có 'frame_id'
    (nếu không có thì lấy từ tên file); mỗi annotation có image_id, track_id, bbox=[x,y,w,h]."""
    root = Path(root) if root else data_root()
    js = json.loads((root / "annotations" / "train.json").read_text(encoding="utf-8"))
    img = {}
    for im in js["images"]:
        p = Path(im["file_name"])
        if len(p.parts) < 2:
            raise ValueError(f"file_name không có thư mục sequence: {im['file_name']}")
        img[im["id"]] = (p.parts[-2], int(im.get("frame_id", p.stem)))
    gt: dict = defaultdict(lambda: defaultdict(dict))
    for a in js["annotations"]:
        seq, fr = img[a["image_id"]]
        gt[seq][int(a["track_id"])][fr] = tuple(a["bbox"])
    return {s: dict(t) for s, t in gt.items()}


def make_folds(k: int = 5, root: Path | None = None) -> dict[str, int]:
    """Chia sequence train thành k fold. Phân tầng theo (độ phân giải, có IMU), mỗi tầng chia thành các khối
    LIÊN TIẾP theo seq_id (giả định: sequence cùng địa điểm có id gần nhau; thay bằng nhãn cảnh thật nếu có)."""
    root = Path(root) if root else data_root()
    strata = defaultdict(list)
    for d in sorted((root / "train").iterdir()):
        if d.is_dir():
            strata[(image_size(frames_of(d)[0]), (d / "imu.csv").exists())].append(d.name)
    return {s: i * k // len(ss) for ss in strata.values() for i, s in enumerate(sorted(ss))}
