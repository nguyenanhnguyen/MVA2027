"""Track 2: dựng cửa sổ, đặc trưng, bộ dự đoán 6 mode, huấn luyện theo SO-HOTA-TP, quy tắc phát/dừng.

Quyết định thiết kế (lý do trong PLAN.md mục 2 và 9):
  * Làm việc trên TỌA ĐỘ ẢNH THÔ, không dùng IMU (bằng chứng trong docs/cmc.md của repo baseline: mọi đặc trưng camera
    làm điểm giảm; phân rã camera/chim thua vận tốc hằng số thô).
  * Loss = winner-takes-all của 1 - exp(-d/S) trung bình trên frame còn nhìn thấy (chính là thứ metric thưởng),
    cộng BCE cho đầu "còn nhìn thấy" (để biết khi nào dừng).
  * Mọi mode dùng CHUNG một điểm dừng => các mode phủ cùng tập frame (validator không cảnh báo).
  * Hai bộ dự đoán cùng giao diện: `cv` (numpy, vận tốc hằng số + 5 biến thể) và `net` (LSTM, cần torch).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from .data import FRAME_4K, S_MEAN
from .formats import HOR, K, OBS, WINDOW
from .track import TrackCfg, track

MIN_OBS, MAX_GAP = 5, 10          # tối thiểu số frame thấy trong cửa sổ; tối đa số frame mất ở cuối cửa sổ


# ================================ cửa sổ và đặc trưng ============================================
def _dense(tr: dict, n: int) -> tuple[np.ndarray, np.ndarray]:
    """track {frame: (x,y,w,h,...)} -> mảng dày đánh chỉ số theo số frame: tâm (n+WINDOW+2, 2) và cỡ sqrt(wh)."""
    pos = np.full((n + WINDOW + 2, 2), np.nan)
    sz = np.full(n + WINDOW + 2, np.nan)
    for f, b in tr.items():
        if 0 < f < len(pos):
            pos[f], sz[f] = (b[0] + b[2] / 2, b[1] + b[3] / 2), np.sqrt(max(b[2] * b[3], 0))
    return pos, sz


def _usable(obs: np.ndarray) -> bool:
    vis = ~np.isnan(obs[:, 0])
    return vis.sum() >= MIN_OBS and OBS - 1 - np.flatnonzero(vis)[-1] <= MAX_GAP


def features(pos: np.ndarray, sz: np.ndarray, wh: np.ndarray):
    """pos (N,OBS,2) có NaN, sz (N,OBS), wh (N,2) kích thước ảnh -> (feats (N,OBS,8), anchor (N,2)).
    Đặc trưng/bước: vị trí so với điểm neo (theo S), cờ thấy, vị trí tuyệt đối chuẩn hóa (để biết gần mép khung
    hình), cỡ chim (theo S), kích thước ảnh. Điểm neo = vị trí thấy cuối cùng."""
    vis = ~np.isnan(pos[..., 0])
    last = OBS - 1 - np.argmax(vis[:, ::-1], axis=1)
    anchor = pos[np.arange(len(pos)), last]
    rel = np.where(vis[..., None], (pos - anchor[:, None]) / S_MEAN, 0)
    absn = np.where(vis[..., None], pos / wh[:, None], 0)
    szn = np.where(vis, np.nan_to_num(sz) / S_MEAN, 0)[..., None]
    meta = np.broadcast_to(wh[:, None] / np.array(FRAME_4K), (len(pos), OBS, 2))
    return np.concatenate([rel, vis[..., None], absn, szn, meta], -1).astype(np.float32), anchor


def match_to_gt(trk: dict, gt: dict, ws: int, min_shared: int = 5) -> dict[int, int]:
    """Khớp track của tracker với track ground-truth như scorer: Hungarian theo khoảng cách tâm trung bình trên
    OBS frame quan sát; loại cặp xa hơn -S*ln(0.05) hoặc chung < 5 frame. Trả {id_tracker: id_gt}."""
    ti, gi = list(trk), list(gt)
    cost = np.full((len(ti), len(gi)), 1e9)
    for a, ta in enumerate(ti):
        for b, tb in enumerate(gi):
            ds = [np.hypot(trk[ta][f][0] + trk[ta][f][2] / 2 - gt[tb][f][0] - gt[tb][f][2] / 2,
                           trk[ta][f][1] + trk[ta][f][3] / 2 - gt[tb][f][1] - gt[tb][f][3] / 2)
                  for f in range(ws, ws + OBS) if f in trk[ta] and f in gt[tb]]
            if len(ds) >= min_shared:
                cost[a, b] = np.mean(ds)
    if not cost.size:
        return {}
    r, c = linear_sum_assignment(cost)
    return {ti[i]: gi[j] for i, j in zip(r, c) if cost[i, j] <= -S_MEAN * np.log(0.05)}


def build_windows(gt: dict, wh: tuple[int, int], n: int, stride: int = 5,
                  dets: dict | None = None, tcfg: TrackCfg | None = None) -> dict[str, np.ndarray]:
    """Mẫu huấn luyện của MỘT sequence. Đích luôn từ ground-truth (frame có nhãn = còn nhìn thấy).
    dets=None: đầu vào là quỹ đạo ground-truth. Có dets: đầu vào là đầu ra của tracker trên 60 frame
    (đúng phân bố lúc thi; để không rò rỉ, dets nên do detector KHÔNG huấn luyện trên sequence này tạo ra)."""
    gd = {g: _dense(tr, n) for g, tr in gt.items()}
    obs, osz, fut = [], [], []
    # chỉ lấy cửa sổ đủ 210 frame như scorer: nếu không, "hết video" bị nhầm thành "chim biến mất" và dạy dừng sai
    for ws in range(1, n - WINDOW + 2, stride):
        if dets is None:
            pairs = [(g, gd[g]) for g in gt]
        else:
            trk = track(dets, range(ws, ws + OBS), tcfg or TrackCfg())
            pairs = [(g, _dense(trk[t], n)) for t, g in match_to_gt(trk, gt, ws).items()]
        for g, (sp, ss) in pairs:
            o = sp[ws:ws + OBS]
            if _usable(o):
                obs.append(o), osz.append(ss[ws:ws + OBS]), fut.append(gd[g][0][ws + OBS:ws + OBS + HOR])
    z = lambda x, s: np.array(x, dtype=np.float64).reshape(s)
    return {"obs": z(obs, (-1, OBS, 2)), "sz": z(osz, (-1, OBS)), "fut": z(fut, (-1, HOR, 2)),
            "wh": np.tile(np.array(wh, dtype=np.float64), (len(obs), 1))}


def concat_windows(parts: list[dict]) -> dict[str, np.ndarray]:
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


# ================================ bộ dự đoán =====================================================
_ROT = np.deg2rad([0, 0, 0, 0, 12, -12])
_MUL = np.array([1.0, 0.6, 1.4, 0.0, 1.0, 1.0])


def predict_cv(obs, sz, wh):
    """Vận tốc hằng số + 5 biến thể (chậm, nhanh, đứng yên, xoay ±12°). Trả (pos (N,K,HOR,2), pvis (N,HOR)).
    pvis = 1 khi mode 0 còn trong khung hình."""
    N = len(obs)
    feats, anchor = features(obs, sz, wh)
    vis = feats[..., 2] > 0
    v = np.zeros((N, 2))
    for i in range(N):
        idx = np.flatnonzero(vis[i])[-10:]
        if len(idx) >= 2:
            v[i] = (obs[i, idx[-1]] - obs[i, idx[0]]) / (idx[-1] - idx[0])
    gap = np.argmax(vis[:, ::-1], axis=1)                              # số frame mất ở cuối cửa sổ
    t = (gap[:, None] + np.arange(1, HOR + 1)[None])[:, None, :, None]  # (N,1,HOR,1)
    c, s = np.cos(_ROT), np.sin(_ROT)
    vk = np.stack([v[:, 0, None] * c - v[:, 1, None] * s, v[:, 0, None] * s + v[:, 1, None] * c], -1) * _MUL[None, :, None]
    pos = anchor[:, None, None, :] + vk[:, :, None, :] * t
    inside = (pos[:, 0] >= 0).all(-1) & (pos[:, 0, :, 0] <= wh[:, None, 0]) & (pos[:, 0, :, 1] <= wh[:, None, 1])
    return pos, inside.astype(np.float32)


def _make_net(hid: int = 128):
    import torch
    from torch import nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = nn.LSTM(8, hid, num_layers=2, batch_first=True)
            self.traj = nn.Sequential(nn.Linear(hid, 256), nn.ReLU(), nn.Linear(256, K * HOR * 2))
            self.vis = nn.Sequential(nn.Linear(hid, 128), nn.ReLU(), nn.Linear(128, HOR))

        def forward(self, x):
            h = self.enc(x)[0][:, -1]
            inc = self.traj(h).view(-1, K, HOR, 2) * 0.1          # bước nhỏ trong đơn vị S
            return torch.cumsum(inc, 2), self.vis(h)              # vị trí tương đối so với điểm neo, logit còn thấy
    return Net()


def sohota_loss(pos, vis_logit, fut, fvis, eps: float = 0.05, lam: float = 0.2):
    """pos (B,K,H,2) và fut (B,H,2) theo đơn vị S, tương đối điểm neo; fvis (B,H) 0/1.
    Chọn mode tốt nhất theo CHÍNH đại lượng được chấm; eps giữ cho mode khác không chết."""
    import torch
    import torch.nn.functional as F
    d = torch.linalg.norm(pos - fut[:, None], dim=-1)
    l = (1 - torch.exp(-d)) * fvis[:, None]
    per_mode = l.sum(-1) / fvis.sum(-1, keepdim=True).clamp(min=1)       # (B,K)
    wta = per_mode.min(1).values.mean() + eps * per_mode.mean()
    return wta + lam * F.binary_cross_entropy_with_logits(vis_logit, fvis)


def _augment(d: dict, idx, rng, p_drop=0.1, noise_px=1.5):
    """Giả lập nhiễu của tracker + lật ảnh. Trả bản sao (obs, sz, fut)."""
    obs, sz, fut, wh = d["obs"][idx].copy(), d["sz"][idx].copy(), d["fut"][idx].copy(), d["wh"][idx]
    keep = rng.random(obs.shape[:2]) > p_drop
    dropped = obs.copy()
    dropped[~keep] = np.nan
    ok = np.array([_usable(o) for o in dropped])
    obs[ok] = dropped[ok]
    obs += rng.normal(0, noise_px, obs.shape)
    for ax in (0, 1):
        flip = rng.random(len(obs)) < 0.5
        for a in (obs, fut):
            a[flip, ..., ax] = wh[flip, ax].reshape(-1, *([1] * (a.ndim - 2))) - a[flip, ..., ax]
    return obs, sz, fut


def _targets(fut, anchor):
    fvis = (~np.isnan(fut[..., 0])).astype(np.float32)
    rel = np.nan_to_num((fut - anchor[:, None]) / S_MEAN).astype(np.float32)
    return rel, fvis


def train_net(train_d: dict, val_d: dict | None, out: Path, epochs: int = 40, bs: int = 512, lr: float = 2e-3,
              seed: int = 0, log=print) -> Path:
    """Huấn luyện LSTM. Chọn checkpoint theo loss validation (nếu có). Cần torch."""
    import torch
    rng, dev = np.random.default_rng(seed), "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    net = _make_net().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    N = len(train_d["obs"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=epochs * ((N + bs - 1) // bs))

    def batch(d, idx, aug):
        o, s, f = _augment(d, idx, rng) if aug else (d["obs"][idx], d["sz"][idx], d["fut"][idx])
        x, anc = features(o, s, d["wh"][idx])
        rel, fv = _targets(f, anc)
        return [torch.from_numpy(a).to(dev) for a in (x, rel, fv)]

    best, out = float("inf"), Path(out)
    for ep in range(epochs):
        net.train()
        perm, tot = rng.permutation(N), 0.0
        for i in range(0, N, bs):
            x, rel, fv = batch(train_d, perm[i:i + bs], True)
            pos, vl = net(x)
            loss = sohota_loss(pos, vl, rel, fv)
            opt.zero_grad(), loss.backward(), opt.step(), sched.step()
            tot += loss.item() * len(x)
        msg = f"epoch {ep + 1}/{epochs} train {tot / N:.4f}"
        if val_d is not None:
            net.eval()
            nv, v = len(val_d["obs"]), 0.0
            with torch.no_grad():
                for i in range(0, nv, bs):
                    x, rel, fv = batch(val_d, np.arange(i, min(i + bs, nv)), False)
                    pos, vl = net(x)
                    v += sohota_loss(pos, vl, rel, fv).item() * len(x)
            v /= nv
            msg += f" val {v:.4f}"
        else:
            v = tot / N
        if v < best:
            best = v
            out.parent.mkdir(parents=True, exist_ok=True)
            torch.save(net.state_dict(), out)
        log(msg)
    return out


def load_net_predictor(path: Path):
    """Trả hàm predict(obs, sz, wh) -> (pos (N,K,HOR,2) px, p_còn_thấy (N,HOR)) dùng mô hình đã huấn luyện."""
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = _make_net().to(dev)
    net.load_state_dict(torch.load(path, map_location=dev))
    net.eval()

    def predict(obs, sz, wh):
        x, anchor = features(obs, sz, wh)
        with torch.no_grad():
            pos, vl = net(torch.from_numpy(x).to(dev))
        return pos.cpu().numpy() * S_MEAN + anchor[:, None, None], torch.sigmoid(vl).cpu().numpy()
    return predict


# ================================ phát / dừng ====================================================
def predict_window(tracks: dict, ws: int, wh: tuple[int, int], predict, tau: float = 0.2) -> list[tuple]:
    """tracks: Tracks của cửa sổ (frame tuyệt đối). Trả các dòng prediction (ws, frame, id, mode, cx, cy).
    Mỗi track phát đủ K mode trên các frame t < điểm dừng; dừng ở frame đầu tiên có p_còn_thấy < tau
    (xấp xỉ: phát nếu p > DetA/(1+DetA) ~ 0.1-0.2; tau nên tinh chỉnh bằng CV với evaluator)."""
    ids, obs, sz = [], [], []
    for tid, tr in tracks.items():
        o, s = (a[ws:ws + OBS] for a in _dense(tr, ws + OBS))
        if _usable(o):
            ids.append(tid), obs.append(o), sz.append(s)
    if not ids:
        return []
    pos, pv = predict(np.array(obs), np.array(sz), np.tile(np.array(wh, dtype=float), (len(ids), 1)))
    rows = []
    for i, tid in enumerate(ids):
        dead = pv[i] < tau
        n_emit = int(np.argmax(dead)) if dead.any() else HOR
        rows += [(ws, ws + OBS + t, tid, k, *pos[i, k, t]) for t in range(n_emit) for k in range(K)]
    return rows
