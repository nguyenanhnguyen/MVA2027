"""Sinh MANIFEST.json cho dữ liệu thô: số frame + dung lượng mỗi sequence.

Dùng:  python scripts/manifest.py            (đọc SMOTP4SB_DATA)
       python scripts/manifest.py --hash     (thêm sha256 từng sequence, chậm)
So sánh MANIFEST.json giữa các máy/đồng đội để chắc dữ liệu giống nhau.
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from smotp.common.paths import data_root  # noqa: E402


def seq_info(d: Path, do_hash: bool) -> dict:
    jpgs = sorted(d.glob("*.jpg"))
    info = {"frames": len(jpgs), "bytes": sum(p.stat().st_size for p in jpgs),
            "has_imu": (d / "imu.csv").exists()}
    if do_hash:
        h = hashlib.sha256()
        for p in jpgs:
            h.update(p.name.encode())
            h.update(str(p.stat().st_size).encode())
        info["fingerprint"] = h.hexdigest()  # tên+kích thước, đủ để so sánh nhanh
    return info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hash", action="store_true")
    a = ap.parse_args()
    root = data_root()
    out = {}
    for split in ("train", "pub_test"):
        d = root / split
        if d.exists():
            out[split] = {s.name: seq_info(s, a.hash) for s in sorted(d.iterdir()) if s.is_dir()}
    tot = {k: sum(v["frames"] for v in out[k].values()) for k in out}
    out["_summary"] = {"frames": tot, "sequences": {k: len(v) for k, v in out.items() if not k.startswith("_")}}
    (root / "MANIFEST.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out["_summary"], indent=1))


if __name__ == "__main__":
    main()
