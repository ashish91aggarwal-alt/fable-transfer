"""Synthesised fireplace (soft crackle, warm rumble) with optional winter wind outside. No recordings used.
Usage: python3 make_fire.py <out.wav> <seconds> [wind 0/1] [seed]"""
import sys, numpy as np, soundfile as sf, pyloudnorm as pyln
from scipy import signal
SR = 44100; out, secs = sys.argv[1], float(sys.argv[2])
wind_on = len(sys.argv) > 3 and sys.argv[3] == "1"; rng = np.random.default_rng(int(sys.argv[4]) if len(sys.argv) > 4 else 3)
N = int(secs * SR)
lp = lambda x, f: signal.sosfilt(signal.butter(2, f, "low", fs=SR, output="sos"), x)
bp = lambda x, a, b: signal.sosfilt(signal.butter(2, [a, b], "band", fs=SR, output="sos"), x)
slow = lambda hz: lp(rng.standard_normal(N), hz) / (np.std(lp(rng.standard_normal(N), hz)) + 1e-9)

def channel():
    flicker = np.clip(1 + 0.35 * slow(0.6), 0.3, 1.8)                  # flames breathing
    rumble = lp(np.cumsum(rng.standard_normal(N)) * 0.02, 180)        # warm low roar
    rumble = (rumble - lp(rumble, 20)) * flicker
    hiss = bp(rng.standard_normal(N), 1500, 5000) * 0.05 * flicker    # gentle gas hiss
    crk = np.zeros(N)
    n_events = int(secs * 2.2)                                        # soft crackles, in small clusters
    for c in rng.integers(0, N - SR, n_events):
        for k in range(rng.integers(1, 4)):
            p = c + int(rng.uniform(0, 0.25) * SR); L = int(rng.uniform(0.004, 0.02) * SR)
            burst = rng.standard_normal(L) * np.exp(-np.arange(L) / (L / 4)) * rng.uniform(0.2, 1.0)
            crk[p:p + L] += burst
    crk = bp(crk, 900, 6000) * 0.6
    pops = np.zeros(N)                                                # rare soft wood pops, low and rounded
    for p in rng.integers(0, N - SR, max(1, int(secs / 25))):
        L = int(0.08 * SR); pops[p:p + L] += np.sin(2 * np.pi * 90 * np.arange(L) / SR) * np.exp(-np.arange(L) / (0.02 * SR))
    x = rumble / (np.std(rumble) + 1e-9) * 0.5 + hiss / (np.std(hiss) + 1e-9) * 0.06 + crk / (np.std(crk) + 1e-9) * 0.12 + pops * 0.15
    if wind_on:                                                       # muffled wind outside the window
        w = lp(rng.standard_normal(N), 500) * np.clip(1 + 0.6 * slow(0.08), 0.2, 2)
        x += w / (np.std(w) + 1e-9) * 0.25
    return x
x = np.stack([channel(), channel()], 1)
e = np.ones(N); f = int(4 * SR); e[:f] = np.linspace(0, 1, f) ** 2; e[-f:] = np.linspace(1, 0, f) ** 2
x = (x.T * e).T
x *= 10 ** ((-30 - pyln.Meter(SR).integrated_loudness(x)) / 20)       # quiet bedtime level
x /= max(1, np.abs(x).max() / 0.8)
sf.write(out, x.astype(np.float32), SR); print(out, "ok")
