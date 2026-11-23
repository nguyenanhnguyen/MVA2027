"""Điểm vào duy nhất:  python -m smotp.cli <lệnh> ...   (xem README.md)

Chuẩn bị : manifest | idmap | folds
Detector : tiles | train-det | detect
Track 1  : track1            (public: tải lên CodaBench)     | run-track1 IN OUT  (private: bên trong gói code)
Track 2  : train-pred | eval-pred | track2                   | run-track2 IN OUT
Cả hai track dùng CHUNG detector, cache detection (theo nội dung) và lõi tracker; chỉ khác lớp cuối.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
import tomllib
from pathlib import Path

import numpy as np

for _s in (sys.stdout, sys.stderr):          # console cp1252 (Windows) hoặc locale POSIX (Docker) không in được tiếng Việt
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

from . import data as D
from .detect import detect_frames, make_tiles, train
from .evaluate import proxy_tp
from .formats import HOR, OBS, check_pred, window_starts, write_mot, write_pred, zip_dir
from .predict import build_windows, concat_windows, load_net_predictor, predict_cv, predict_window, train_net
from .track import TrackCfg, to_rows, track

CFG_PATH = D.REPO / "configs" / "default.toml"
FOLDS = D.REPO / "splits" / "folds.json"


def load_cfg(path=None) -> dict:
    return tomllib.loads(Path(path or CFG_PATH).read_text(encoding="utf-8"))


def folds() -> dict[str, int]:
    return json.loads(FOLDS.read_text(encoding="utf-8"))


def ledger(track: int, run_id: str, where: str, score: float | None, note: str = "") -> None:
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=D.REPO, capture_output=True, text=True).stdout.strip()
    except OSError:
        commit = ""
    p = D.REPO / "experiments" / "ledger.csv"
    with open(p, "a", newline="") as f:
        csv.writer(f).writerow([time.strftime("%Y-%m-%d"), track, run_id, commit, "", where, score, "", "", note])


# ================================ nguồn đầu vào ==================================================
def public_sequences(split: str, track: int):
    root = D.data_root(track) / split
    return [(d.name, D.frames_of(d)) for d in sorted(root.iterdir()) if d.is_dir()]


def private_windows(inp: Path, cfg: dict, weights):
    """Bố cục private Track 2: {seq}/{window_start}/00001..00060.jpg  (frame tương đối -> tuyệt đối)."""
    for sd in sorted(p for p in inp.iterdir() if p.is_dir()):
        for wd in sorted((p for p in sd.iterdir() if p.is_dir()), key=lambda p: int(p.name)):
            ws, paths = int(wd.name), D.frames_of(wd)
            dets = detect_frames(paths, cfg, weights)
            yield sd.name, ws, D.image_size(paths[0]), {ws - 1 + i: a for i, a in dets.items()}


def public_windows(seqs, cfg: dict, weights):
    for seq, paths in seqs:
        dets, wh = detect_frames(paths, cfg, weights), D.image_size(paths[0])
        for ws in window_starts(len(paths)):
            yield seq, ws, wh, {f: dets[f] for f in range(ws, ws + OBS)}


# ================================ chạy hai track =================================================
def run_track1(seqs, cfg: dict, out: Path, weights=None) -> Path:
    tc = TrackCfg.from_dict(cfg["track1"])
    for seq, paths in seqs:
        dets = detect_frames(paths, cfg, weights)
        write_mot(out / f"{seq}.txt", to_rows(track(dets, range(1, len(paths) + 1), tc)))
        print(f"[track1] {seq}: {len(paths)} frame", flush=True)
    return out


def get_predictor(cfg: dict):
    ck = D.models_root() / cfg["pred"]["weights"]
    if ck.exists():
        return load_net_predictor(ck)
    print(f"[cảnh báo] chưa có {ck}: dùng bộ dự đoán vận tốc hằng số", flush=True)
    return predict_cv


def run_track2(windows, cfg: dict, out: Path, seq_ids: list[str]) -> Path:
    tc, predict, tau = TrackCfg.from_dict(cfg["track2"]), get_predictor(cfg), cfg["pred"]["tau"]
    trk_rows = {s: [] for s in seq_ids}
    prd_rows = {s: [] for s in seq_ids}
    last_ws = 1
    for seq, ws, wh, dets in windows:
        tr = track(dets, range(ws, ws + OBS), tc)
        off = ws * 1000                                     # id duy nhất giữa các cửa sổ
        trk_rows[seq] += to_rows(tr, off)
        prd_rows[seq] += predict_window({t + off: v for t, v in tr.items()}, ws, wh, predict, tau)
        last_ws = max(last_ws, ws)
    for s in seq_ids:                                       # sequence thiếu file bị tính toàn miss: luôn ghi file
        write_mot(out / "tracking" / f"{s}.txt", trk_rows[s])
        write_pred(out / "prediction" / f"{s}.txt", prd_rows[s])
        errs = check_pred(prd_rows[s], trk_rows[s], last_ws + 209)
        if errs:
            print(f"[track2] {s}: {errs[:3]}", flush=True)
    return out


# ================================ các lệnh =======================================================
def cmd_manifest(a, cfg):
    out = {}
    for split in ("train", "pub_test"):
        if (D.data_root(a.track) / split).exists():
            out[split] = {}
            for seq, paths in public_sequences(split, a.track):
                out[split][seq] = {"frames": len(paths), "size": D.image_size(paths[0]), "key": D.content_key(paths),
                                   "imu": (paths[0].parent / "imu.csv").exists()}
    p = D.work_root() / f"MANIFEST_t{a.track}.json"
    p.write_text(json.dumps(out, indent=1))
    print(p, {k: (len(v), sum(x["frames"] for x in v.values())) for k, v in out.items()})


def cmd_idmap(a, cfg):
    m = D.idmap(D.data_root(1), D.data_root(2), a.split)
    (D.REPO / "splits" / f"idmap_{a.split}.json").write_text(json.dumps(m, indent=1))
    print(f"trùng nội dung: {len(m['a_to_b'])} | chỉ gói 1: {len(m['only_a'])} | chỉ gói 2: {len(m['only_b'])}")


def cmd_folds(a, cfg):
    FOLDS.parent.mkdir(exist_ok=True)
    f = D.make_folds(cfg["data"]["fold_k"])
    FOLDS.write_text(json.dumps(f, indent=1))
    print({k: sum(v == k for v in f.values()) for k in range(cfg["data"]["fold_k"])})


def cmd_tiles(a, cfg):
    print(make_tiles(cfg, folds(), a.val_fold))


def cmd_train_det(a, cfg):
    print(train(cfg, make_tiles(cfg, folds(), a.val_fold), a.val_fold))


def cmd_detect(a, cfg):
    for seq, paths in public_sequences(a.split, a.track):
        detect_frames(paths, cfg, a.weights)
        print(f"[detect] {a.split}/{seq}", flush=True)


def out_dir(a, name):
    p = Path(a.out) if a.out else D.REPO / "submissions" / f"{name}_{time.strftime('%Y%m%d_%H%M%S')}"
    p.mkdir(parents=True, exist_ok=True)
    return p


def cmd_track1(a, cfg):
    o = run_track1(public_sequences(a.split, 1), cfg, out_dir(a, "track1"), a.weights)
    print("zip:", zip_dir(o, o.with_suffix(".zip")))


def cmd_track2(a, cfg):
    seqs = public_sequences(a.split, 2)
    o = run_track2(public_windows(seqs, cfg, a.weights), cfg, out_dir(a, "track2"), [s for s, _ in seqs])
    print("zip:", zip_dir(o, o.with_suffix(".zip")))


def _windows_for(cfg, a, stride, part):
    """Mẫu cửa sổ của các sequence train: part='val' (fold val-fold), 'train' (các fold còn lại)."""
    f, gt, root, tc = folds(), D.load_gt(), D.data_root(), TrackCfg.from_dict(cfg["track2"])
    parts = []
    for seq, tracks in gt.items():
        if (f[seq] == a.val_fold) != (part == "val"):
            continue
        paths = D.frames_of(root / "train" / seq)
        dets = detect_frames(paths, cfg, a.oof.format(fold=f[seq])) if a.source == "tracker" else None
        parts.append(build_windows(tracks, D.image_size(paths[0]), len(paths), stride, dets, tc))
    return concat_windows(parts)


def cmd_train_pred(a, cfg):
    p = cfg["pred"]
    tr, va = _windows_for(cfg, a, p["stride"], "train"), _windows_for(cfg, a, p["stride"], "val")
    print(f"cửa sổ train {len(tr['obs'])}, val {len(va['obs'])}")
    train_net(tr, va, D.models_root() / p["weights"], p["epochs"])


def cmd_eval_pred(a, cfg):
    """Ước lượng nhanh trên fold val với bước 150 như cửa sổ thật; so bộ dự đoán hiện có với vận tốc hằng số."""
    va, tau0 = _windows_for(cfg, a, HOR, "val"), cfg["pred"]["tau"]
    score = None
    for name, fn in (("cv", predict_cv), ("hiện tại", get_predictor(cfg))):
        pos, pv = fn(va["obs"], va["sz"], va["wh"])
        for tau in sorted({0.05, 0.1, 0.2, 0.3, 0.5, tau0}):
            r = proxy_tp(pos, ~np.maximum.accumulate(pv < tau, axis=1), va["fut"])
            score = r["proxy_deta_tp"] if (name != "cv" and tau == tau0) else score
            print(f"{name:9s} tau={tau:<4} proxy_DetA-TP={r['proxy_deta_tp']:.4f}  minADE6={r['minADE6_px']:.1f}px  "
                  f"phát={r['emit_rate']:.2f}  n={r['n_windows']}")
    ledger(2, f"eval_{a.source}_f{a.val_fold}", "local", score, f"proxy DetA-TP, tau={tau0}")


def cmd_run1(a, cfg):
    inp = Path(a.input)
    seqs = [(d.name, D.frames_of(d)) for d in sorted(inp.iterdir()) if d.is_dir()]
    run_track1(seqs, cfg, Path(a.output))


def cmd_run2(a, cfg):
    inp = Path(a.input)
    run_track2(private_windows(inp, cfg, None), cfg, Path(a.output), [d.name for d in sorted(inp.iterdir()) if d.is_dir()])


def main(argv=None):
    ap = argparse.ArgumentParser(prog="smotp", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add(name, fn, *args):
        p = sub.add_parser(name)
        p.set_defaults(fn=fn)
        for k, v in args:
            p.add_argument(k, **v)
        return p

    track = ("--track", dict(type=int, default=1, choices=(1, 2)))
    split = ("--split", dict(default="pub_test"))
    fold = ("--val-fold", dict(type=int, default=0))
    weights = ("--weights", dict(default=None, help="trọng số detector (mặc định models/det_final.pt)"))
    out = ("--out", dict(default=None))
    source = ("--source", dict(default="gt", choices=("gt", "tracker")))
    oof = ("--oof", dict(default=str(D.models_root() / "det_fold{fold}.pt"), help="mẫu đường dẫn detector OOF theo fold"))

    add("manifest", cmd_manifest, track)
    add("idmap", cmd_idmap, split)
    add("folds", cmd_folds)
    add("tiles", cmd_tiles, fold)
    add("train-det", cmd_train_det, fold)
    add("detect", cmd_detect, split, track, weights)
    add("track1", cmd_track1, split, weights, out)
    add("track2", cmd_track2, split, weights, out)
    add("train-pred", cmd_train_pred, fold, source, oof)
    add("eval-pred", cmd_eval_pred, fold, source, oof)
    add("run-track1", cmd_run1, ("input", {}), ("output", {}))
    add("run-track2", cmd_run2, ("input", {}), ("output", {}))
    a = ap.parse_args(argv)
    a.fn(a, load_cfg(a.config))


if __name__ == "__main__":
    main()
