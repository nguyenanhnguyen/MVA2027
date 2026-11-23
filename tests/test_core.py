"""Kiểm thử logic lõi bằng dữ liệu tổng hợp (không cần dữ liệu thật, GPU, torch hay ultralytics).
Chạy:  python -m pytest tests -q      hoặc      python tests/test_core.py   (không cần pytest)"""
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.environ.setdefault("SMOTP_WORK", tempfile.mkdtemp(prefix="smotp_test_"))

from smotp import cli, data as D, formats as F  # noqa: E402
from smotp.detect import boxes_in_tile, tile_grid  # noqa: E402
from smotp.evaluate import proxy_tp  # noqa: E402
from smotp.predict import build_windows, match_to_gt, predict_cv, predict_window  # noqa: E402
from smotp.track import TrackCfg, to_rows, track  # noqa: E402

WH = (3840, 2160)


def _bird(x0, y0, vx, vy, n, w=16, exit_at_edge=True):
    """Quỹ đạo thẳng đều; ngừng có nhãn khi tâm ra khỏi khung hình."""
    tr = {}
    for f in range(1, n + 1):
        cx, cy = x0 + vx * (f - 1), y0 + vy * (f - 1)
        if exit_at_edge and not (0 <= cx <= WH[0] and 0 <= cy <= WH[1]):
            break
        tr[f] = (cx - w / 2, cy - w / 2, w, w)
    return tr


# ---------------------------------------------------------------- định dạng
def test_window_starts():
    assert F.window_starts(209) == [] and F.window_starts(210) == [1]
    assert F.window_starts(360) == [1, 151] and F.window_starts(359) == [1]


def test_roundtrip_and_check(tmp_path=None):
    tmp = Path(tmp_path or tempfile.mkdtemp())
    F.write_mot(tmp / "a.txt", [(1, 7, 10.0, 20.0, 16.0, 16.0, 0.9)])
    assert F.read_mot(tmp / "a.txt")[0][:2] == (1, 7)
    rows = [(1, 61 + t, 7, k, 1.0, 1.0) for t in range(5) for k in range(6)]
    F.write_pred(tmp / "p.txt", rows)
    assert F.read_pred(tmp / "p.txt") == rows
    trk = [(1, 7, 0, 0, 1, 1, 1.0)]
    assert F.check_pred(rows, trk, 360) == []
    bad = rows[:-1] + [(1, 60, 9, 6, 1.0, 1.0)]
    assert len(F.check_pred(bad, trk, 360)) >= 3
    z = F.zip_dir(tmp, tmp.parent / (tmp.name + ".zip"))
    assert z.exists()


# ---------------------------------------------------------------- tile
def test_tiles():
    for W, H in ((3840, 2160), (1920, 1080), (1000, 800)):
        cover = np.zeros((H, W), bool)
        for x, y, w, h in tile_grid(W, H, 1280, 0.2):
            assert x + w <= W and y + h <= H
            cover[y:y + h, x:x + w] = True
        assert cover.all()
    assert len(tile_grid(3840, 2160, 1280, 0.2)) == 8
    b = np.array([[1270.0, 10, 20, 20], [100, 100, 10, 10]])
    out = boxes_in_tile(b, 0, 0, 1280, 1280)
    assert len(out) == 2 and abs(out[0, 2] - 10) < 1e-6          # hộp bị cắt, còn 50% => giữ, đã cắt cho vừa
    assert len(boxes_in_tile(b, 1280, 0, 1280, 1280)) == 1 or len(boxes_in_tile(b, 1280, 0, 1280, 1280)) == 0


# ---------------------------------------------------------------- tracker
def test_tracker_two_birds():
    rng = np.random.default_rng(0)
    gt = {1: _bird(100, 100, 3, 0, 120), 2: _bird(500, 300, -2, 1, 120)}
    dets = {}
    for f in range(1, 121):
        rows = []
        for tid, tr in gt.items():
            if f in tr and rng.random() > 0.15:                          # 15% bỏ sót
                x, y, w, h = tr[f]
                rows.append((x + rng.normal(0, .5), y + rng.normal(0, .5), w, h, 0.8))
        dets[f] = np.array(rows).reshape(-1, 5)
    out = track(dets, range(1, 121), TrackCfg())
    assert len(out) == 2
    for tr in out.values():
        assert len(tr) >= 115                                             # nội suy lấp khoảng trống
    owners = set()                                                       # mỗi track phải bám ĐÚNG MỘT con chim
    for tr in out.values():
        near = [all(np.hypot(b[0] - gt[g][f][0], b[1] - gt[g][f][1]) < 5 for f, b in tr.items() if f in gt[g])
                for g in gt]
        assert sum(near) == 1
        owners.add(near.index(True))
    assert owners == {0, 1}
    rows = to_rows(out)
    assert rows == sorted(rows)


def test_tracker_edge_cases():
    assert track({}, range(1, 10), TrackCfg()) == {}
    assert track({1: np.zeros((0, 5))}, range(1, 3), TrackCfg()) == {}


# ---------------------------------------------------------------- dữ liệu
def test_content_key_and_idmap(tmp_path=None):
    from PIL import Image
    tmp = Path(tmp_path or tempfile.mkdtemp())
    rng = np.random.default_rng(1)
    for root, name, seed in (("a", "0001", 1), ("a", "0002", 2), ("b", "0900", 1), ("b", "0901", 3)):
        d = tmp / root / "pub_test" / name
        d.mkdir(parents=True)
        r = np.random.default_rng(seed)
        for i in range(1, 5):
            Image.fromarray(r.integers(0, 255, (32, 32, 3), dtype=np.uint8)).save(d / f"{i:05d}.jpg")
    m = D.idmap(tmp / "a", tmp / "b")
    assert m["a_to_b"] == {"0001": "0900"} and m["only_a"] == ["0002"] and m["only_b"] == ["0901"]
    bad = tmp / "a" / "pub_test" / "0001" / "00002.jpg"
    bad.rename(bad.with_name("00009.jpg"))
    try:
        D.frames_of(bad.parent)
        raise AssertionError("phải báo lỗi frame không liên tục")
    except ValueError:
        pass


# ---------------------------------------------------------------- dự đoán
def test_match_and_windows_and_proxy():
    n = 400
    gt = {5: _bird(1000, 1000, 2, 1, n), 9: _bird(3000, 500, -4, 0, n)}      # bird 9 rời khung ở ~frame 750? (không), bird 5 ở lại
    gt[9] = _bird(1000, 500, -8, 0, n)                                          # ra khỏi khung hình (x<0) quanh frame 126
    assert max(gt[9]) < 130
    w = build_windows(gt, WH, n, stride=150)
    assert w["obs"].shape[1:] == (60, 2) and w["fut"].shape[1:] == (150, 2) and len(w["obs"]) >= 2
    # khớp tracker->GT: tracker đặt id khác nhưng cùng vị trí
    trk = {1: {f: (*b, 1.0) for f, b in gt[5].items()}, 2: {f: (*b, 1.0) for f, b in gt[9].items()}}
    assert match_to_gt(trk, gt, 1) == {1: 5, 2: 9}
    # CV với đầu vào GT trên quỹ đạo thẳng đều: gần như hoàn hảo
    pos, pv = predict_cv(w["obs"], w["sz"], w["wh"])
    assert pos.shape == (len(w["obs"]), 6, 150, 2) and pv.shape == (len(w["obs"]), 150)
    emit = ~np.maximum.accumulate(pv < 0.2, axis=1)
    r = proxy_tp(pos, emit, w["fut"])
    assert r["proxy_deta_tp"] > 0.95, r
    # phát thừa sau khi chim đã mất => điểm giảm (tính chất cốt lõi của SO-HOTA-TP)
    r2 = proxy_tp(pos, np.ones_like(emit), w["fut"])
    assert r2["proxy_deta_tp"] < r["proxy_deta_tp"]
    # đứng yên thì kém hơn đi thẳng
    still = np.repeat(np.nan_to_num(w["obs"][:, -1])[:, None, None, :], 6, 1).repeat(150, 2)
    assert proxy_tp(still, emit, w["fut"])["proxy_deta_tp"] < r["proxy_deta_tp"]


def test_augment_keeps_geometry():
    from smotp.predict import _augment, _targets, features
    n = 400
    w = build_windows({1: _bird(1000, 1000, 2, 1, n)}, WH, n, stride=50)
    rng = np.random.default_rng(0)
    o, s, f = _augment(w, np.arange(len(w["obs"])), rng, p_drop=0.1, noise_px=0.0)
    x, anchor = features(o, s, w["wh"])
    rel, fvis = _targets(f, anchor)
    assert x.shape == (len(o), 60, 8) and rel.shape == (len(o), 150, 2) and np.isfinite(x).all()
    # chim đi thẳng đều: lật ảnh/bỏ frame không được làm vỡ quan hệ "đích = neo + vận tốc * thời gian" (|v| = sqrt(5))
    i, last = 0, 59 - np.argmax(~np.isnan(o[0, :, 0])[::-1])
    gap = 59 - last
    assert abs(np.linalg.norm(rel[i, 0]) * 24.15 - np.sqrt(5) * (gap + 1)) < 1e-3
    assert fvis.min() == 1                                              # chim còn nhìn thấy suốt 150 frame


def test_predict_window_rows():
    tr = {1: {f: (*b, 0.9) for f, b in _bird(1000, 1000, 2, 1, 60).items()}}
    rows = predict_window(tr, 1, WH, predict_cv, tau=0.2)
    assert rows and {r[3] for r in rows} == set(range(6))
    assert min(r[1] for r in rows) == 61 and max(r[1] for r in rows) <= 210
    assert F.check_pred(rows, [(f, 1, 0, 0, 1, 1, 1) for f in range(1, 61)], 210) == []
    # track quá ngắn / mất quá lâu không được dự đoán
    short = {1: {f: (*b, 0.9) for f, b in _bird(1000, 1000, 2, 1, 3).items()}}
    assert predict_window(short, 1, WH, predict_cv) == []
    old = {1: {f: (*b, 0.9) for f, b in _bird(1000, 1000, 2, 1, 40).items()}}
    assert predict_window(old, 1, WH, predict_cv) == []                       # mất 20 frame cuối > MAX_GAP


# ---------------------------------------------------------------- tích hợp (detector giả)
def test_pipeline_with_fake_detector(tmp_path=None):
    tmp = Path(tmp_path or tempfile.mkdtemp())
    n = 420
    gt = _bird(800, 800, 2, 1, n)

    def fake_detect(paths, cfg, weights=None):
        return {f: np.array([[*gt[f], 0.9]]) if f in gt else np.zeros((0, 5)) for f in range(1, len(paths) + 1)}

    orig = cli.detect_frames
    cli.detect_frames = fake_detect
    try:
        cfg = cli.load_cfg()
        paths = [Path(f"{i:05d}.jpg") for i in range(1, n + 1)]
        # Track 1
        o1 = cli.run_track1([("0001", paths)], cfg, tmp / "t1")
        rows = F.read_mot(o1 / "0001.txt")
        assert len({r[1] for r in rows}) == 1 and len(rows) == n
        # Track 2: các cửa sổ 1 và 151 (cần 210 frame) => n=420 cho 2 cửa sổ
        wins = [("0001", ws, WH, {f: np.array([[*gt[f], 0.9]]) for f in range(ws, ws + 60)}) for ws in F.window_starts(n)]
        o2 = cli.run_track2(iter(wins), cfg, tmp / "t2", ["0001", "0002"])
        pred, trk = F.read_pred(o2 / "prediction" / "0001.txt"), F.read_mot(o2 / "tracking" / "0001.txt")
        assert {r[0] for r in pred} == {1, 151} and F.check_pred(pred, trk, n) == []
        assert (o2 / "prediction" / "0002.txt").read_text() == ""             # sequence không có chim vẫn có file
    finally:
        cli.detect_frames = orig


def test_torch_optional():
    try:
        import torch  # noqa: F401
    except ImportError:
        print("  (bỏ qua: chưa cài torch)")
        return
    from smotp.predict import load_net_predictor, train_net
    n = 400
    gt = {1: _bird(1000, 1000, 2, 1, n), 2: _bird(2000, 700, -3, 1, n)}
    w = build_windows(gt, WH, n, stride=20)
    out = Path(tempfile.mkdtemp()) / "m.pt"
    train_net(w, w, out, epochs=2, bs=32, log=lambda *_: None)
    pos, pv = load_net_predictor(out)(w["obs"][:3], w["sz"][:3], w["wh"][:3])
    assert pos.shape == (3, 6, 150, 2) and pv.shape == (3, 150)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("OK ", name)
