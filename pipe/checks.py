"""Frame error checks for one generated clip against its start/end paintings.
Usage: python3 checks.py --clip X.mp4 --start A.png --end B.png [--cfg thresholds.json] [--out report.json]
Every metric is computed locally (OpenCV + an offline segmentation net); no API cost."""
import argparse, json, os, sys, hashlib
import numpy as np, cv2
from skimage.metrics import structural_similarity as ssim
from skimage.morphology import skeletonize

DEFAULT = {
    "fps_ok": [15, 31], "min_frames": 40, "min_dur": 2.5, "max_dur": 11.0,
    "start_ssim": 0.55, "end_ssim": 0.50,          # first/last frame must resemble the paintings
    "black_luma": 12, "flicker_jump": 14.0,          # mean-luma jump between frames
    "residual_z": 6.0, "residual_abs": 9.0,          # appearance change not explained by motion, in character region
    "area_tol": 0.18, "area_jump": 0.12,             # silhouette area vs painting range / frame-to-frame
    "ends_jump": 6, "parts_jump": 2,                 # skeleton end-points / mask pieces change vs clip median
    "chatter_ratio": 0.40, "chatter_min": 0.35,      # in-place 2.5-7 Hz motion energy in small cells
    "color_de": 9.0,                                  # drift of character mean colour (a*,b*)
    "cam_shift": 6.0, "cam_scale": 0.035, "cam_jitter": 0.9,
    "dup_area": 0.35,                                 # extra large character blob, fraction of the main blob
    "seg_every": 6, "seg_w": 384,
}

_seg = None
def seg_mask(rgb):
    global _seg
    from rembg import remove, new_session
    from PIL import Image
    if _seg is None: _seg = new_session('isnet-general-use')
    m = np.array(remove(Image.fromarray(rgb), session=_seg, only_mask=True))
    return (m > 128).astype(np.uint8)

def read_video(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened(): return None, 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    fr = []
    while True:
        ok, f = cap.read()
        if not ok: break
        fr.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    return fr, fps

def rs(img, w):
    h = int(round(w * 9 / 16));  return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)

def robust_z(x):
    x = np.asarray(x, float); med = np.median(x); mad = np.median(np.abs(x - med)) * 1.4826 + 1e-6
    return (x - med) / mad

def lab_ab(rgb, m):
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(float)
    if m.sum() < 50: return np.array([np.nan, np.nan])
    return lab[m > 0][:, 1:].mean(0)

def skel_ends(m):
    if m.sum() < 50: return 0
    sk = skeletonize(m > 0).astype(np.uint8)
    nb = cv2.filter2D(sk, -1, np.ones((3, 3), np.float32)) * sk
    return int((nb == 2).sum())

def parts(m, frac=0.02):
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    a = sorted([s[4] for s in st[1:]], reverse=True)
    if not a: return 0, a
    return sum(1 for x in a if x > frac * a[0]), a

def run(clip, start, end, cfg):
    C = dict(DEFAULT); C.update(cfg or {})
    R = {"clip": clip, "checks": {}, "fail": [], "worst": {}}
    def chk(name, ok, value, worst=None, note=""):
        R["checks"][name] = {"pass": bool(ok), "value": value, "note": note}
        if worst is not None: R["worst"][name] = int(worst)
        if not ok: R["fail"].append(name)
    if not os.path.exists(clip) or os.path.getsize(clip) < 1000:
        chk("file", False, "missing/empty"); return R
    R["sha1"] = hashlib.sha1(open(clip, 'rb').read()).hexdigest()
    fr, fps = read_video(clip)
    if not fr: chk("file", False, "undecodable"); return R
    n = len(fr); dur = n / fps if fps else 0
    chk("fps", C["fps_ok"][0] <= fps <= C["fps_ok"][1], round(fps, 2))
    chk("length", n >= C["min_frames"] and C["min_dur"] <= dur <= C["max_dur"], {"frames": n, "sec": round(dur, 2)})
    if n < 8: return R
    S = rs(cv2.cvtColor(cv2.imread(start), cv2.COLOR_BGR2RGB), 344); E = rs(cv2.cvtColor(cv2.imread(end), cv2.COLOR_BGR2RGB), 344)
    sm = [rs(f, 344) for f in fr]; g = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in sm]
    s0 = ssim(cv2.cvtColor(S, cv2.COLOR_RGB2GRAY), g[0], data_range=255); s1 = ssim(cv2.cvtColor(E, cv2.COLOR_RGB2GRAY), g[-1], data_range=255)
    chk("start_match", s0 >= C["start_ssim"], round(float(s0), 3)); chk("end_match", s1 >= C["end_ssim"], round(float(s1), 3))
    luma = np.array([x.mean() for x in g]); black = np.where(luma < C["black_luma"])[0]
    chk("black_frames", len(black) == 0, int(len(black)), black[0] if len(black) else None)
    jl = np.abs(np.diff(luma)); chk("flicker", jl.max() < C["flicker_jump"], round(float(jl.max()), 2), int(jl.argmax()) + 1)
    # ---- segmentation of the character(s): paintings + sampled frames
    W = C["seg_w"]; mS = seg_mask(rs(cv2.cvtColor(cv2.imread(start), cv2.COLOR_BGR2RGB), W)); mE = seg_mask(rs(cv2.cvtColor(cv2.imread(end), cv2.COLOR_BGR2RGB), W))
    idx = list(range(0, n, C["seg_every"]))
    if idx[-1] != n - 1: idx.append(n - 1)
    M = {i: seg_mask(rs(fr[i], W)) for i in idx}
    aS, aE = mS.sum(), mE.sum(); lo, hi = min(aS, aE) * (1 - C["area_tol"]), max(aS, aE) * (1 + C["area_tol"])
    area = np.array([M[i].sum() for i in idx], float)
    bad = [i for i, a in zip(idx, area) if not (lo <= a <= hi)]
    chk("silhouette_area", not bad, {"start": int(aS), "end": int(aE), "min": int(area.min()), "max": int(area.max())}, bad[0] if bad else None)
    ra = np.abs(np.diff(area)) / (np.maximum(area[:-1], 1)); j = int(ra.argmax())
    chk("silhouette_jump", ra.max() < C["area_jump"], round(float(ra.max()), 3), idx[j + 1])
    ends = np.array([skel_ends(M[i]) for i in idx]); pc = np.array([parts(M[i])[0] for i in idx])
    de = np.abs(ends - np.median(ends)); dp = np.abs(pc - np.median(pc))
    chk("limb_ends", de.max() <= C["ends_jump"], {"median": float(np.median(ends)), "maxdev": int(de.max())}, idx[int(de.argmax())])
    chk("body_parts", dp.max() <= C["parts_jump"], {"median": float(np.median(pc)), "maxdev": int(dp.max())}, idx[int(dp.argmax())])
    # duplicate character: a second big blob that the paintings don't have
    def big2(m):
        _, a = parts(m); return (a[1] / a[0]) if len(a) > 1 else 0
    ref2 = max(big2(mS), big2(mE)); d2 = np.array([big2(M[i]) for i in idx])
    chk("extra_character", d2.max() < max(C["dup_area"], ref2 + 0.15), round(float(d2.max()), 3), idx[int(d2.argmax())])
    # colour drift
    abS = lab_ab(rs(cv2.cvtColor(cv2.imread(start), cv2.COLOR_BGR2RGB), W), mS); abE = lab_ab(rs(cv2.cvtColor(cv2.imread(end), cv2.COLOR_BGR2RGB), W), mE)
    dE = []
    for i in idx:
        ab = lab_ab(rs(fr[i], W), M[i]); dE.append(min(np.linalg.norm(ab - abS), np.linalg.norm(ab - abE)) if not np.isnan(ab).any() else 0)
    dE = np.array(dE); chk("color_drift", dE.max() < C["color_de"], round(float(dE.max()), 2), idx[int(dE.argmax())])
    # ---- optical flow: camera lock, residual (unexplained change), chatter
    def up(m):  # mask to flow resolution, dilated
        return cv2.dilate(cv2.resize(m, (344, 194), interpolation=cv2.INTER_NEAREST), np.ones((9, 9), np.uint8))
    mask_at = lambda k: up(M[min(idx, key=lambda i: abs(i - k))])
    H, Wd = g[0].shape; gx, gy = np.meshgrid(np.arange(Wd), np.arange(H))
    cam_t, cam_s, resid, cells = [], [], [], []
    for k in range(1, n):
        fl = cv2.calcOpticalFlowFarneback(g[k - 1], g[k], None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mk = cv2.resize(mask_at(k), (Wd, H), interpolation=cv2.INTER_NEAREST)
        bg = mk == 0
        pts = np.stack([gx[bg], gy[bg]], 1).astype(np.float32)[::7]; v = fl[bg][::7]
        if len(pts) > 30:
            A, _ = cv2.estimateAffinePartial2D(np.ascontiguousarray(pts.reshape(-1,1,2),np.float32), np.ascontiguousarray((pts + v).reshape(-1,1,2),np.float32), method=cv2.RANSAC, ransacReprojThreshold=1.0)
        else: A = None
        if A is None: A = np.array([[1, 0, 0], [0, 1, 0]], float)
        cam_t.append(A[:, 2].copy()); cam_s.append(np.sqrt(abs(np.linalg.det(A[:, :2]))))
        warped = cv2.remap(g[k - 1], (gx + fl[..., 0]).astype(np.float32), (gy + fl[..., 1]).astype(np.float32), cv2.INTER_LINEAR)
        rr = np.abs(warped.astype(float) - g[k]); resid.append(rr[mk > 0].mean() if mk.sum() > 30 else 0)
        # in-place motion per 8x8 cell inside character
        mag = fl[..., 1] if True else None
        cells.append(cv2.resize(fl, (Wd // 8, H // 8), interpolation=cv2.INTER_AREA) * (cv2.resize(mk, (Wd // 8, H // 8), interpolation=cv2.INTER_NEAREST)[..., None] > 0))
    cam_t = np.array(cam_t); cam_s = np.array(cam_s); cum = np.cumsum(cam_t, 0)
    jit = np.abs(cam_t - cv2.GaussianBlur(cam_t, (1, 9), 0)).mean(1)
    chk("camera_locked", np.abs(cum).max() < C["cam_shift"] and abs(np.prod(cam_s) - 1) < C["cam_scale"] and jit.max() < C["cam_jitter"],
        {"drift_px": round(float(np.abs(cum).max()), 2), "scale": round(float(np.prod(cam_s)), 4), "jitter": round(float(jit.max()), 2)}, int(np.abs(cum).max(1).argmax()) + 1)
    resid = np.array(resid); z = robust_z(resid); kk = int(np.argmax(z))
    chk("morph_residual", not (z.max() > C["residual_z"] and resid.max() > C["residual_abs"]), {"max": round(float(resid.max()), 2), "z": round(float(z.max()), 1)}, kk + 1)
    # chatter: cells whose motion oscillates 2.5-7 Hz while their mean displacement is ~0
    cs = np.stack(cells, 0)  # T,h,w,2
    T = cs.shape[0]; ratio_best, amp_best = 0, 0
    if T >= 16 and fps:
        f = np.fft.rfftfreq(T, 1 / fps); band = (f >= 2.5) & (f <= 7)
        for comp in (0, 1):
            x = cs[..., comp]; x = x - x.mean(0)
            P = np.abs(np.fft.rfft(x, axis=0)) ** 2
            tot = P[1:].sum(0) + 1e-6; rb = P[band].sum(0) / tot; amp = x.std(0)
            sel = amp > C["chatter_min"]
            if sel.any():
                r = rb[sel].max(); ratio_best = max(ratio_best, float(r)); amp_best = max(amp_best, float(amp[sel].max()))
    chk("mouth_chatter", ratio_best < C["chatter_ratio"], {"band_ratio": round(ratio_best, 3), "amp": round(amp_best, 2)})
    R["n_frames"] = n; R["fps"] = fps
    R["score"] = float(resid.max() + 10 * ra.max() + 5 * ratio_best)  # lower is better among passing clips
    return R

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--clip'); ap.add_argument('--start'); ap.add_argument('--end'); ap.add_argument('--cfg'); ap.add_argument('--out')
    a = ap.parse_args(); cfg = json.load(open(a.cfg)) if a.cfg and os.path.exists(a.cfg) else {}
    r = run(a.clip, a.start, a.end, cfg); s = json.dumps(r, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
    if a.out: open(a.out, 'w').write(s)
    print(s)
