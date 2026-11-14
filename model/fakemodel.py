#!/usr/bin/env python3
"""Adversarial stand-in for an image-to-video model (QA stress test).

python3 fakemodel.py --start PATH --end PATH --prompt TEXT --seed INT --out PATH.mp4

Deterministic per (NN, seed, sha256(prompt)). Hidden truth -> key.jsonl next to this file.
"""
import argparse, hashlib, json, os, sys, shutil, subprocess, fcntl, time
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
KEY = os.path.join(HERE, "key.jsonl")
INDEX = os.path.join(HERE, "index.npz")
CLIPS = "/tmp/clips"
INPUTS = "/tmp/dry/inputs"
NNS = ["01", "02", "03", "04", "07", "10", "12", "13", "14", "15", "16"]
W, H, FPS = 1280, 704, 30
COST = 0.30

# Approximate character boxes at frame 0 (1280x704 coords), from looking at frames.
CHARS = {  # (name, body box, face box) at frame 0
    "01": [("Gus", [160, 290, 440, 350], [370, 370, 150, 130]), ("Ana", [790, 260, 90, 130], [800, 270, 50, 45])],
    "02": [("Ana", [300, 90, 360, 510], [460, 220, 170, 150])],
    "03": [("Gus", [300, 90, 640, 490], [710, 226, 150, 150])],
    "04": [("Gus", [170, 90, 450, 520], [370, 256, 170, 160]), ("Ana", [710, 96, 330, 460], [750, 186, 120, 120])],
    "07": [("Ana", [690, 110, 350, 460], [750, 230, 110, 110])],
    "10": [("Gus", [300, 170, 570, 460], [400, 290, 190, 160])],
    "12": [("Gus", [300, 136, 340, 560], [450, 246, 140, 130])],
    "13": [("Ana", [430, 36, 290, 590], [520, 176, 200, 150])],
    "14": [("Ana", [250, 150, 310, 390], [390, 220, 130, 110]), ("Gus", [800, 220, 300, 420], [810, 320, 100, 100])],
    "15": [("Gus", [290, 280, 240, 390], [400, 300, 120, 100]), ("Ana", [760, 250, 220, 410], [770, 290, 120, 110])],
    "16": [("Gus", [350, 136, 290, 480], [500, 236, 110, 110]), ("Ana", [710, 146, 230, 450], [730, 236, 130, 120])],
}
# Base fault probability per NN (varies around 0.5)
BASE_P = {"01": .40, "02": .55, "03": .35, "04": .60, "07": .50, "10": .45,
          "12": .55, "13": .40, "14": .65, "15": .50, "16": .55}
SOURCE_FAULT = {
    "12": "figure peeking from the door at the end is green/yellow (reads as a second grasshopper, not rust-red Ana); "
          "also present in end_12.png - judged from low-res frames",
}

CONTENT_FAULTS = {  # type: weight
    "missing_limb": 8, "extra_limb": 7, "merged_limbs": 6, "body_melt": 7, "extra_eyes": 6,
    "mouth_chatter": 7, "chewing": 4, "colour_drift_yellow": 6, "camera_shake": 5,
    "camera_zoom": 4, "camera_pan": 4, "flicker": 6, "texture_boil": 4, "duplicate_character": 4,
}
STRUCT_FAULTS = {"wrong_ending": 6, "wrong_first_frame": 4, "wrong_fps": 3, "wrong_length": 3,
                 "black_frames": 4, "truncated_mp4": 2}
API_FAULTS = {"timeout": 3, "rate_limited": 3, "internal_error": 2, "duplicate_output": 2}


# ---------------------------------------------------------------- utils
def thumb(img):
    g = cv2.cvtColor(cv2.resize(img, (64, 36), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    return g.astype(np.float32)


def read_frames(nn):
    cap = cv2.VideoCapture(os.path.join(CLIPS, f"clip{nn}.mp4"))
    fr = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if f.shape[:2] != (H, W):
            f = cv2.resize(f, (W, H))
        fr.append(f)
    return fr


def build_index():
    data = {}
    for nn in NNS:
        cap = cv2.VideoCapture(os.path.join(CLIPS, f"clip{nn}.mp4"))
        ok, f = cap.read()
        data["clip" + nn] = thumb(f)
        p = os.path.join(INPUTS, f"start_{nn}.png")
        if os.path.exists(p):
            data["start" + nn] = thumb(cv2.imread(p))
    np.savez(INDEX, **data)


def identify(img):
    if not os.path.exists(INDEX):
        build_index()
    idx = np.load(INDEX)
    t = thumb(img)
    best, bd = None, 1e9
    for k in idx.files:
        d = float(np.mean(np.abs(idx[k] - t)))
        if d < bd:
            bd, best = d, k[-2:]
    return best, bd


def wchoice(rng, d):
    ks = list(d)
    w = np.array([d[k] for k in ks], float)
    return ks[rng.choice(len(ks), p=w / w.sum())]


def clampbox(b):
    x, y, w, h = [int(v) for v in b]
    x = max(0, min(W - 8, x)); y = max(0, min(H - 8, y))
    w = max(8, min(W - x, w)); h = max(8, min(H - y, h))
    return [x, y, w, h]


def track(frames, box, a, b):
    """Follow a box from frame a to b with coarse template matching. Returns list of boxes."""
    s = 4
    x, y, w, h = box
    out = [list(box)]
    g0 = cv2.cvtColor(cv2.resize(frames[a], (W // s, H // s)), cv2.COLOR_BGR2GRAY)
    tpl = g0[y // s:(y + h) // s, x // s:(x + w) // s]
    cx, cy = x, y
    for i in range(a + 1, b + 1):
        g = cv2.cvtColor(cv2.resize(frames[i], (W // s, H // s)), cv2.COLOR_BGR2GRAY)
        r = 15
        x0 = max(0, cx // s - r); y0 = max(0, cy // s - r)
        x1 = min(W // s, (cx + w) // s + r); y1 = min(H // s, (cy + h) // s + r)
        win = g[y0:y1, x0:x1]
        if win.shape[0] > tpl.shape[0] and win.shape[1] > tpl.shape[1] and tpl.size > 0:
            res = cv2.matchTemplate(win, tpl, cv2.TM_CCOEFF_NORMED)
            _, mv, _, ml = cv2.minMaxLoc(res)
            if mv > 0.4:
                cx, cy = (x0 + ml[0]) * s, (y0 + ml[1]) * s
        out.append(clampbox([cx, cy, w, h]))
    return out


def feather(h, w, k=0.25):
    m = np.zeros((h, w), np.float32)
    cv2.ellipse(m, (w // 2, h // 2), (max(1, int(w * .45)), max(1, int(h * .45))), 0, 0, 360, 1, -1)
    ks = max(3, int(min(h, w) * k) | 1)
    return cv2.GaussianBlur(m, (ks, ks), 0)[..., None]


def ramp(i, a, b, rise=0.3):
    """Envelope 0..1..0 inside [a,b] (quick rise, hold, quick fall)."""
    n = max(1, b - a)
    t = (i - a) / n
    r = max(1e-3, rise)
    return float(np.clip(min(t / r, (1 - t) / r, 1.0), 0, 1)) if n > 2 else 1.0


def char_mask(patch, who):
    """Soft mask of the character's own colours inside a patch (green for Gus, rust for Ana)."""
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    hh, ss, vv = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    if who == "Gus":
        m = (hh >= 22) & (hh <= 50) & (ss > 70) & (vv > 50)
    else:
        m = ((hh <= 16) | (hh >= 170)) & (ss > 70) & (vv > 40)
    m = m.astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    m = cv2.dilate(m, np.ones((5, 5), np.uint8))
    return cv2.GaussianBlur(m.astype(np.float32) / 255, (7, 7), 0)[..., None]


def find_eyes(frame, box):
    x, y, w, h = box
    roi = frame[y:y + h, x:x + w]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    m = ((hsv[..., 2] > 200) & (hsv[..., 1] < 60)).astype(np.uint8)
    n, lab, st, cen = cv2.connectedComponentsWithStats(m)
    cands = [st[i] for i in range(1, n) if max(60, w * h * .004) < st[i][4] < w * h * 0.15]
    if not cands:
        return None
    cands.sort(key=lambda s: -s[4])
    ex, ey, ew, eh = cands[0][:4]
    if len(cands) > 1:  # union of two biggest whites = eye pair
        bx, by, bw, bh = cands[1][:4]
        if abs(by - ey) < eh * 1.5:
            nx, ny = min(ex, bx), min(ey, by)
            ew, eh = max(ex + ew, bx + bw) - nx, max(ey + eh, by + bh) - ny
            ex, ey = nx, ny
    return [x + ex, y + ey, ew, eh]


def warp_roi(frame, box, dx, dy):
    x, y, w, h = box
    roi = frame[y:y + h, x:x + w]
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    dx = np.broadcast_to(np.asarray(dx, np.float32), gx.shape)
    dy = np.broadcast_to(np.asarray(dy, np.float32), gx.shape)
    frame[y:y + h, x:x + w] = cv2.remap(roi, (gx + dx).astype(np.float32), (gy + dy).astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def gauss2d(w, h, cx, cy, sx, sy):
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return np.exp(-(((gx - cx) / sx) ** 2 + ((gy - cy) / sy) ** 2) / 2), gx, gy


# ---------------------------------------------------------------- content faults
def span_for(rng, sev, n):
    L = {1: (3, 6), 2: (10, 30), 3: (30, 90)}[sev]
    d = int(rng.integers(L[0], L[1] + 1))
    d = min(d, n - 2)
    a = int(rng.integers(1, max(2, n - d)))
    return a, min(n - 1, a + d)


def apply_content(kind, frames, rng, sev, nn, info):
    n = len(frames)
    who, box0, face0 = CHARS[nn][int(rng.integers(len(CHARS[nn])))]
    a, b = span_for(rng, sev, n)
    if kind in ("camera_zoom", "camera_pan") and sev >= 2:
        a, b = int(rng.integers(0, n // 3)), n - 1
    # box at frame a: track from 0
    boxes0 = track(frames, clampbox(box0), 0, a) if a > 0 else [clampbox(box0)]
    ba = boxes0[-1]
    boxes = track(frames, ba, a, b)
    x, y, w, h = ba
    region = None
    amp = {1: 0.5, 2: 1.0, 3: 1.7}[sev]

    if kind in ("missing_limb", "extra_limb", "merged_limbs"):
        # limb zone: bottom 35% of the body box (legs); pick a sub-strip
        lw = int(w * rng.uniform(0.12, 0.22) * (0.8 + 0.2 * sev))
        ly = int(h * 0.62)
        lh = h - ly
        cm = char_mask(frames[a][y + ly:y + h, x:x + w], who)[..., 0]
        cands = [int(rng.uniform(0.05, 0.95 - lw / w) * w) for _ in range(10)]
        lx = max(cands, key=lambda c: float(cm[:, c:c + lw].mean()))
        rel = [lx, ly, lw, lh]
        region = [x + lx, y + ly, lw, lh]
        shift = int(w * rng.uniform(0.18, 0.35)) * (1 if rng.random() < .5 else -1)
        for k, i in enumerate(range(a, b + 1)):
            bx, by, _, _ = boxes[k]
            r = clampbox([bx + rel[0], by + rel[1], rel[2], rel[3]])
            f = frames[i]
            e = ramp(i, a, b, 0.15 if sev > 1 else 0.01)
            if kind == "missing_limb":
                X, Y, Ww, Hh = r
                side = -1 if (X + Ww / 2) < (boxes[k][0] + w / 2) else 1
                off = int(Ww * 1.6) if k == 0 else off
                SX = int(np.clip(X + side * off, 0, W - Ww))
                if abs(SX - X) < Ww:
                    SX = int(np.clip(X - side * off, 0, W - Ww))
                src = f[Y:Y + Hh, SX:SX + Ww].astype(np.float32)
                dst = f[Y:Y + Hh, X:X + Ww].astype(np.float32)
                fm = feather(Hh, Ww, 0.35) * e
                fm = np.minimum(1, fm * 1.4)
                f[Y:Y + Hh, X:X + Ww] = (dst * (1 - fm) + src * fm).astype(np.uint8)
            elif kind == "extra_limb":
                X, Y, Ww, Hh = r
                tx = int(np.clip(X + shift, 0, W - Ww))
                patch = f[Y:Y + Hh, X:X + Ww].astype(np.float32)
                ang = rng.uniform(-12, 12) if k == 0 else ang
                M = cv2.getRotationMatrix2D((Ww / 2, 0), ang, 1.0)
                patch = cv2.warpAffine(patch, M, (Ww, Hh), borderMode=cv2.BORDER_REFLECT)
                m = feather(Hh, Ww, 0.3) * e
                dst = f[Y:Y + Hh, tx:tx + Ww].astype(np.float32)
                f[Y:Y + Hh, tx:tx + Ww] = (dst * (1 - m) + patch * m).astype(np.uint8)
            else:  # merged: pull a wider leg zone toward its centre line
                X, Y = boxes[k][0], boxes[k][1] + ly
                zb = clampbox([X + int(w * .1), Y, int(w * .8), lh])
                ww, hh = zb[2], zb[3]
                g, gx, gy = gauss2d(ww, hh, ww / 2, hh * .7, ww * .3, hh * .5)
                dxf = (gx - ww / 2) * 0.35 * amp * e * g
                warp_roi(f, zb, dxf, np.zeros_like(dxf))
                region = [x + int(w * .1), y + ly, int(w * .8), lh]

    elif kind == "body_melt":
        cxr, cyr = rng.uniform(.3, .7), rng.uniform(.4, .8)
        region = [x, y, w, h]
        noise = cv2.GaussianBlur(rng.standard_normal((h // 8 + 1, w // 8 + 1)).astype(np.float32), (0, 0), 2)
        noise = cv2.resize(noise, (w, h)) / (np.abs(noise).max() + 1e-6)
        for k, i in enumerate(range(a, b + 1)):
            grow = ((i - a + 1) / (b - a + 1)) ** 1.3  # grows over time
            g, gx, gy = gauss2d(w, h, w * cxr, h * cyr, w * .22 * (0.6 + .4 * grow), h * .22 * (0.6 + .4 * grow))
            mag = 22 * amp * grow
            warp_roi(frames[i], boxes[k], noise * mag * g, -np.abs(noise) * mag * 1.4 * g)

    elif kind in ("extra_eyes", "mouth_chatter", "chewing"):
        fx, fy, fw, fh = face0
        fx, fy = fx + (x - box0[0]), fy + (y - box0[1])  # follow body drift to frame a
        eyes = find_eyes(frames[a], clampbox([fx, fy, fw, int(fh * 1.4)]))
        if eyes is None or eyes[2] < fw * .15:
            eyes = [fx + int(fw * .25), fy + int(fh * .2), int(fw * .5), int(fh * .3)]
        ex, ey, ew, eh = eyes
        ew, eh = max(ew, 12), max(eh, 10)
        mouth_c = (fx + fw // 2 - x, fy + int(fh * .78) - y)
        rel = [ex - x, ey - y, ew, eh]
        if kind == "extra_eyes":
            pad = 4
            dxo = int(rng.choice([-1, 1]) * ew * rng.uniform(0.45, 0.8))
            dyo = int(-eh * rng.uniform(0.0, 0.6))
            sc = rng.uniform(0.6, 0.9) if sev == 1 else rng.uniform(0.8, 1.0)
            region = [ex + dxo, ey + dyo, ew, eh]
            for k, i in enumerate(range(a, b + 1)):
                f = frames[i]
                X, Y = boxes[k][0] + rel[0] - pad, boxes[k][1] + rel[1] - pad
                Ww, Hh = ew + 2 * pad, eh + 2 * pad
                X, Y = max(0, X), max(0, Y)
                patch = f[Y:Y + Hh, X:X + Ww].astype(np.float32)
                if patch.size == 0:
                    continue
                pw, ph = max(4, int(patch.shape[1] * sc)), max(4, int(patch.shape[0] * sc))
                patch = cv2.resize(patch, (pw, ph))
                TX, TY = int(np.clip(X + dxo, 0, W - pw)), int(np.clip(Y + dyo, 0, H - ph))
                m = feather(ph, pw, 0.2) * ramp(i, a, b, 0.1 if sev > 1 else 0.01)
                dst = f[TY:TY + ph, TX:TX + pw].astype(np.float32)
                f[TY:TY + ph, TX:TX + pw] = (dst * (1 - m) + patch * m).astype(np.uint8)
        else:
            mw, mh = int(fw * .45), int(fh * .35)
            mrel = [mouth_c[0] - mw // 2, mouth_c[1] - mh // 2, mw, mh]
            region = [x + mrel[0] - mw // 2, y + mrel[1] - mh // 2, mw * 2, mh * 2]
            hz = rng.uniform(2, 5) if kind == "mouth_chatter" else rng.uniform(1.2, 2.2)
            for k, i in enumerate(range(a, b + 1)):
                zb = clampbox([boxes[k][0] + mrel[0] - mw // 2, boxes[k][1] + mrel[1] - mh // 2, mw * 2, mh * 2])
                ww, hh = zb[2], zb[3]
                g, gx, gy = gauss2d(ww, hh, ww / 2, hh / 2, ww * .25, hh * .25)
                ph = 2 * np.pi * hz * (i - a) / FPS
                e = ramp(i, a, b, 0.1)
                if kind == "mouth_chatter":
                    dyf = (gy - hh / 2) * 0.45 * amp * np.sin(ph) * g * e
                    dxf = np.zeros_like(dyf)
                else:
                    dyf = (gy - hh / 2) * 0.35 * amp * max(0, np.sin(ph)) * g * e + 3 * amp * np.sin(ph) * g * e
                    dxf = 4 * amp * np.sin(ph + 1.2) * g * e
                warp_roi(frames[i], zb, dxf, dyf)

    elif kind == "colour_drift_yellow":
        a = int(rng.integers(0, n // 2)) if sev > 1 else a
        b = n - 1 if sev > 1 else b
        boxes = track(frames, track(frames, clampbox(box0), 0, a)[-1], a, b)
        region = [x, y, w, h]
        m0 = feather(h, w, 0.3)
        for k, i in enumerate(range(a, b + 1)):
            t = (i - a + 1) / (b - a + 1) if sev > 1 else ramp(i, a, b, 0.3)
            s = 0.35 * amp * t
            X, Y, Ww, Hh = boxes[k]
            roi = frames[i][Y:Y + Hh, X:X + Ww].astype(np.float32)
            yel = roi.copy()
            yel[..., 0] *= (1 - 0.6 * s)  # B down
            yel[..., 1] = yel[..., 1] + (255 - yel[..., 1]) * 0.45 * s
            yel[..., 2] = yel[..., 2] + (255 - yel[..., 2]) * 0.35 * s
            m = m0[:Hh, :Ww] * char_mask(frames[i][Y:Y + Hh, X:X + Ww], who)
            frames[i][Y:Y + Hh, X:X + Ww] = np.clip(roi * (1 - m) + yel * m, 0, 255).astype(np.uint8)

    elif kind in ("camera_shake", "camera_zoom", "camera_pan"):
        region = None
        dirx, diry = rng.uniform(-1, 1), rng.uniform(-.4, .4)
        for i in range(a, b + 1):
            t = (i - a) / max(1, b - a)
            if kind == "camera_shake":
                e = ramp(i, a, b, 0.2)
                tx, ty = rng.normal(0, 4 * amp) * e, rng.normal(0, 3 * amp) * e
                z, rot = 1.02 + 0.01 * amp, rng.normal(0, 0.25 * amp) * e
            elif kind == "camera_zoom":
                z = 1 + (0.06 * amp) * (t if sev > 1 else ramp(i, a, b, 0.4))
                tx = ty = rot = 0
            else:
                z = 1.08
                tt = t if sev > 1 else ramp(i, a, b, 0.4)
                tx, ty, rot = dirx * 40 * amp * tt, diry * 20 * amp * tt, 0
            M = cv2.getRotationMatrix2D((W / 2, H / 2), rot, z)
            M[0, 2] += tx; M[1, 2] += ty
            frames[i] = cv2.warpAffine(frames[i], M, (W, H), borderMode=cv2.BORDER_REFLECT)

    elif kind == "flicker":
        local = rng.random() < 0.5
        region = [x, y, w, h] if local else None
        for i in range(a, b + 1):
            g = 1 + rng.choice([-1, 1]) * rng.uniform(0.5, 1.5) * 0.05 * amp * ramp(i, a, b, 0.05)
            if local:
                X, Y, Ww, Hh = clampbox([x, y, w, h])
                m = feather(Hh, Ww, .3)
                roi = frames[i][Y:Y + Hh, X:X + Ww].astype(np.float32)
                frames[i][Y:Y + Hh, X:X + Ww] = np.clip(roi * (1 - m) + roi * g * m, 0, 255).astype(np.uint8)
            else:
                frames[i] = np.clip(frames[i].astype(np.float32) * g, 0, 255).astype(np.uint8)

    elif kind == "texture_boil":
        bw_, bh_ = int(rng.uniform(150, 400)), int(rng.uniform(100, 300))
        if rng.random() < .6:
            r = clampbox([x, y, w, h])
        else:
            r = clampbox([int(rng.integers(0, W - bw_)), int(rng.integers(0, H - bh_)), bw_, bh_])
        region = r
        X, Y, Ww, Hh = r
        for i in range(a, b + 1):
            nz = rng.standard_normal((Hh // 6 + 1, Ww // 6 + 1, 2)).astype(np.float32)
            nz = cv2.resize(nz, (Ww, Hh)) * 1.5 * amp * ramp(i, a, b, 0.1)
            m = feather(Hh, Ww, .3)[..., 0]
            warp_roi(frames[i], r, nz[..., 0] * m, nz[..., 1] * m)

    elif kind == "duplicate_character":
        sc = rng.uniform(0.55, 0.9)
        pw, ph = int(w * sc), int(h * sc)
        tx = int(rng.integers(0, W - pw)); ty = int(np.clip(y + h - ph + rng.integers(-30, 30), 0, H - ph))
        region = [tx, ty, pw, ph]
        flip = rng.random() < .5
        for k, i in enumerate(range(a, b + 1)):
            X, Y, Ww, Hh = boxes[k]
            src = cv2.resize(frames[i][Y:Y + Hh, X:X + Ww], (pw, ph))
            cm = char_mask(src, who)
            patch = src.astype(np.float32)
            if flip:
                patch, cm = patch[:, ::-1], cm[:, ::-1]
            m = feather(ph, pw, 0.15) * cm * ramp(i, a, b, 0.1 if sev > 1 else 0.01)
            dst = frames[i][ty:ty + ph, tx:tx + pw].astype(np.float32)
            frames[i][ty:ty + ph, tx:tx + pw] = (dst * (1 - m) + patch * m).astype(np.uint8)

    dxy = [int(boxes[-1][0] - boxes[0][0]), int(boxes[-1][1] - boxes[0][1])] if region else None
    return {"type": kind, "frames": [int(a), int(b)], "region": [int(v) for v in region] if region else None,
            "region_note": "region at first fault frame; it follows the character by track_shift_at_end" if region else None,
            "track_shift_at_end": dxy,
            "severity": int(sev), "character": who if kind not in ("camera_shake", "camera_zoom", "camera_pan", "flicker") or region else None}


# ---------------------------------------------------------------- main
def encode(frames, out, fps):
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    tmp = out + ".part.mp4"
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
                          "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
                          "-pix_fmt", "yuv420p", "-movflags", "+faststart", tmp], stdin=subprocess.PIPE)
    for f in frames:
        p.stdin.write(np.ascontiguousarray(f).tobytes())
    p.stdin.close()
    if p.wait() != 0:
        raise RuntimeError("ffmpeg failed")
    os.replace(tmp, out)


def log_key(rec):
    with open(KEY, "a") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        fh.write(json.dumps(rec) + "\n")
        fcntl.flock(fh, fcntl.LOCK_UN)


def prev_outputs(exclude):
    outs = []
    if os.path.exists(KEY):
        for line in open(KEY):
            try:
                r = json.loads(line)
            except Exception:
                continue
            o = r.get("out")
            if r.get("status") == "ok" and o and os.path.abspath(o) != os.path.abspath(exclude) and os.path.exists(o):
                outs.append((r.get("call_id"), o))
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True); ap.add_argument("--end", required=True)
    ap.add_argument("--prompt", required=True); ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    A = ap.parse_args()
    t0 = time.time()
    si, ei = cv2.imread(A.start), cv2.imread(A.end)
    if si is None or ei is None:
        print(json.dumps({"status": "error", "error": "invalid_input", "message": "could not read image",
                          "cost_usd": 0.0}), file=sys.stderr)
        sys.exit(2)
    nn, dist = identify(si)
    world_only = float(np.mean(np.abs(thumb(si) - thumb(ei)))) < 1.0
    ph = hashlib.sha256(A.prompt.encode()).hexdigest()
    h = hashlib.sha256(f"{nn}|{A.seed}|{ph}".encode()).hexdigest()
    rng = np.random.default_rng(int(h[:16], 16))
    nprev = sum(1 for _ in open(KEY)) if os.path.exists(KEY) else 0
    call_id = f"c{nprev:04d}-{h[:8]}"
    rec = {"call_id": call_id, "NN": nn, "match_dist": round(dist, 2), "seed": A.seed, "prompt": A.prompt,
           "out": os.path.abspath(A.out), "world_only": world_only, "faults": [], "clean": True,
           "source_fault_suspected": SOURCE_FAULT.get(nn)}

    p_fault = BASE_P[nn] * (0.8 if world_only else 1.0)
    faulty = rng.random() < p_fault
    # ---- API-level failures (subset of faulty)
    if faulty and rng.random() < 0.14:
        kind = wchoice(rng, API_FAULTS)
        if kind == "duplicate_output":
            prev = prev_outputs(A.out)
            if prev:
                cid, src = prev[int(rng.integers(len(prev)))]
                shutil.copyfile(src, A.out)
                rec.update(clean=False, status="ok", faults=[{"type": "duplicate_output", "frames": None,
                           "region": None, "severity": 3, "duplicate_of": cid, "source_out": src}])
                log_key(rec)
                print(json.dumps({"status": "ok", "out": A.out, "cost_usd": COST}))
                return
        else:
            billed = kind != "rate_limited"
            cost = COST if billed else 0.0
            err = {"status": "error", "error": kind, "message": {
                "timeout": "generation exceeded 600s deadline", "rate_limited": "too many requests, retry later",
                "internal_error": "worker crashed (CUDA error: an illegal memory access)"}[kind],
                "cost_usd": cost}
            if kind == "rate_limited":
                err["retry_after_s"] = int(rng.integers(5, 60))
            rec.update(clean=False, status="error", out=None,
                       faults=[{"type": kind, "frames": None, "region": None, "severity": 3, "billed": billed}])
            log_key(rec)
            print(json.dumps(err), file=sys.stderr)
            sys.exit(2)

    frames = read_frames(nn)
    if world_only:
        L = 24  # ping-pong the first ~0.8 s at ~third speed -> barely moving
        seq = []
        t = 0.0; d = 1
        while len(seq) < 161:
            seq.append(int(round(t)))
            t += d * 0.35
            if t >= L - 1 or t <= 0:
                d = -d
        frames = [frames[i].copy() for i in seq]
    fps = FPS

    struct = []
    content = []
    if faulty:
        nf = 1 + int(rng.random() < 0.35) + int(rng.random() < 0.12)
        for _ in range(nf):
            if rng.random() < 0.27:
                k = wchoice(rng, STRUCT_FAULTS)
                if world_only and k == "wrong_ending" and rng.random() < .5:
                    k = "wrong_first_frame"
                if k not in [s for s, _ in struct]:
                    struct.append((k, int(rng.integers(1, 4))))
            else:
                k = wchoice(rng, CONTENT_FAULTS)
                if k not in [c for c, _ in content]:
                    content.append((k, int(rng.choice([1, 1, 2, 2, 3]))))

    sk = dict(struct)
    # length-changing structural faults first so content frame indices refer to output frames
    if "wrong_ending" in sk:
        sev = sk["wrong_ending"]
        n0 = len(frames)
        mode = rng.choice(["early", "reverse", "drift"])
        if mode == "early":  # stops mid-motion, padded out to full length with a hold
            cut = int(n0 * {1: .8, 2: .6, 3: .4}[sev])
            frames = frames[:cut] + [frames[cut - 1].copy() for _ in range(n0 - cut)] if rng.random() < .5 else frames[:cut]
            rec["faults"].append({"type": "wrong_ending", "mode": "ends_early", "frames": [cut, len(frames) - 1] if len(frames) > cut else [cut - 1, cut - 1], "region": None, "severity": sev})
        elif mode == "reverse":  # goes back to start pose
            piv = int(n0 * {1: .85, 2: .65, 3: .5}[sev])
            frames = frames[:piv] + frames[piv - 1::-1][:n0 - piv]
            rec["faults"].append({"type": "wrong_ending", "mode": "returns_toward_start", "frames": [piv, len(frames) - 1], "region": None, "severity": sev})
        else:  # ends on different framing (camera drift at end)
            a = int(n0 * .6)
            for i in range(a, n0):
                t = (i - a) / (n0 - a)
                M = cv2.getRotationMatrix2D((W * .3, H * .5), 0, 1 + 0.12 * sev * t)
                frames[i] = cv2.warpAffine(frames[i], M, (W, H), borderMode=cv2.BORDER_REFLECT)
            rec["faults"].append({"type": "wrong_ending", "mode": "ends_on_different_framing", "frames": [a, n0 - 1], "region": None, "severity": sev})
    if "wrong_length" in sk:
        tgt = int(rng.choice([49, 81, 97, 121, 241]))
        if tgt < len(frames):
            frames = frames[:tgt]
        else:
            base = list(frames)
            seq = base[::-1][1:] + base[1:]
            frames = base + [f.copy() for f in seq[:tgt - len(base)]]
        rec["faults"].append({"type": "wrong_length", "frames": [0, len(frames) - 1], "region": None, "severity": 2, "n_frames": len(frames)})

    for k, sev in content:
        try:
            rec["faults"].append(apply_content(k, frames, rng, sev, nn, None))
        except Exception as ex:  # never crash a call because of a fault
            rec.setdefault("fault_errors", []).append(f"{k}: {ex}")

    n = len(frames)
    if "wrong_first_frame" in sk:
        sev = sk["wrong_first_frame"]
        k = {1: 1, 2: 2, 3: 4}[sev]
        other = [m for m in NNS if m != nn][int(rng.integers(len(NNS) - 1))]
        mode = rng.choice(["other_scene", "late_frame", "exposure"])
        if mode == "other_scene":
            src = read_frames(other)[:k]
            for i in range(k):
                frames[i] = src[i]
        elif mode == "late_frame":
            for i in range(k):
                frames[i] = frames[n - 1 - i].copy()
        else:
            for i in range(k):
                frames[i] = cv2.convertScaleAbs(frames[i], alpha=1.6, beta=30)
        rec["faults"].append({"type": "wrong_first_frame", "mode": mode, "frames": [0, k - 1], "region": None, "severity": sev})
    if "black_frames" in sk:
        sev = sk["black_frames"]
        k = {1: 1, 2: 2, 3: 4}[sev]
        a = int(rng.integers(1, n - k))
        for i in range(a, a + k):
            frames[i] = np.zeros_like(frames[i])
        rec["faults"].append({"type": "black_frames", "frames": [a, a + k - 1], "region": None, "severity": sev})
    if "wrong_fps" in sk:
        fps = int(rng.choice([16, 24, 60]))
        rec["faults"].append({"type": "wrong_fps", "frames": [0, n - 1], "region": None, "severity": 2, "fps": fps})

    encode(frames, A.out, fps)
    if "truncated_mp4" in sk:
        sz = os.path.getsize(A.out)
        if rng.random() < .6:
            keep = int(sz * rng.uniform(0.3, 0.8))
            with open(A.out, "r+b") as fh:
                fh.truncate(keep)
            mode = f"truncated_to_{keep}_bytes"
        else:
            with open(A.out, "r+b") as fh:
                fh.seek(int(sz * .5))
                fh.write(os.urandom(int(sz * .1)))
            mode = "garbage_overwrite_mid"
        rec["faults"].append({"type": "truncated_mp4", "mode": mode, "frames": None, "region": None, "severity": 3})

    rec["clean"] = len(rec["faults"]) == 0
    rec["status"] = "ok"
    rec["n_frames"], rec["fps"], rec["elapsed_s"] = n, fps, round(time.time() - t0, 1)
    log_key(rec)
    print(json.dumps({"status": "ok", "out": A.out, "cost_usd": COST}))


if __name__ == "__main__":
    main()
