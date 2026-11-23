import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from smotp.common import formats as F  # noqa: E402


def test_window_starts():
    assert F.window_starts(209) == []
    assert F.window_starts(210) == [1]
    assert F.window_starts(360) == [1, 151]
    assert F.window_starts(359) == [1]


def test_roundtrip(tmp_path):
    rows = [(1, 7, 10.0, 20.0, 16.0, 16.0, 0.9)]
    F.write_mot(tmp_path / "a.txt", rows)
    assert F.read_mot(tmp_path / "a.txt")[0][:2] == (1, 7)
    p = [(1, 61, 7, 0, 100.0, 200.0)]
    F.write_pred(tmp_path / "p.txt", p)
    assert F.read_pred(tmp_path / "p.txt") == p


def test_check_pred():
    trk = [(1, 7, 0, 0, 1, 1, 1.0)]
    ok = [(1, 61, 7, 0, 1.0, 1.0), (1, 210, 7, 5, 1.0, 1.0)]
    assert F.check_pred(ok, trk, 360) == []
    bad = [(1, 60, 7, 0, 1.0, 1.0), (1, 61, 9, 6, 1.0, 1.0), (1, 61, 9, 6, 1.0, 1.0)]
    assert len(F.check_pred(bad, trk, 360)) >= 3
