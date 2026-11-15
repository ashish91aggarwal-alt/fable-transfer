"""Endless, non-repeating fire bed from a short CC0 fire recording (random grains + crossfades) plus a soft warm roar.
Usage: python3 fire_bed.py <source.wav> <out.wav> <seconds> [seed]"""
import sys, numpy as np, soundfile as sf
from scipy import signal
src, out, secs = sys.argv[1], sys.argv[2], float(sys.argv[3]); rng = np.random.default_rng(int(sys.argv[4]) if len(sys.argv) > 4 else 5)
x, SR = sf.read(src, dtype="float32"); x = x if x.ndim == 1 else x.mean(1)
N = int(secs * SR)
def channel():
    y = np.zeros(N + SR * 3); p = 0; xf = int(0.35 * SR)
    while p < N:
        L = int(rng.uniform(0.9, min(2.2, len(x) / SR - 0.1)) * SR); s = int(rng.uniform(0, len(x) - L))
        g = x[s:s + L].copy() * rng.uniform(0.7, 1.0)
        if rng.random() < 0.5: g = g[::-1] * 0 + g          # keep forward (reversed crackles sound wrong)
        w = np.ones(L); w[:xf] = np.sin(np.linspace(0, np.pi / 2, xf)) ** 2; w[-xf:] = np.cos(np.linspace(0, np.pi / 2, xf)) ** 2
        y[p:p + L] += g * w; p += L - xf
    y = y[:N]
    roar = signal.sosfilt(signal.butter(2, 160, "low", fs=SR, output="sos"), np.cumsum(rng.standard_normal(N)) * 0.01)
    roar -= signal.sosfilt(signal.butter(1, 25, "low", fs=SR, output="sos"), roar)
    roar *= np.clip(1 + 0.3 * signal.sosfilt(signal.butter(1, 0.5, "low", fs=SR, output="sos"), rng.standard_normal(N)) * 20, 0.4, 1.6)
    return y / (np.std(y) + 1e-9) + 0.35 * roar / (np.std(roar) + 1e-9)
y = np.stack([channel(), channel()], 1); y /= np.abs(y).max() / 0.8
sf.write(out, y.astype(np.float32), SR); print(out, "ok")
