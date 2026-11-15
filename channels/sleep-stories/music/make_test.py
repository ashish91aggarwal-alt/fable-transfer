"""Music test A/B/C under real narration. Everything is synthesised here (no samples, soundfonts or recordings).
Usage: python3 make_test.py <voice.wav> <out_dir> [seed]
A = voice + rain, B = A + soft drone pad, C = B + sparse plucked notes in the voice gaps.
"""
import sys, json, numpy as np, soundfile as sf, pyloudnorm as pyln
from scipy import signal

SR = 44100
voice_path, out = sys.argv[1], sys.argv[2]
rng = np.random.default_rng(int(sys.argv[3]) if len(sys.argv) > 3 else 7)

v, vsr = sf.read(voice_path, dtype="float32")
if v.ndim > 1: v = v.mean(1)
v = signal.resample_poly(v, SR, vsr).astype(np.float32)
N = len(v) + 8 * SR                      # 8 s of tail after the last word
v = np.pad(v, (2 * SR, N - len(v) - 2 * SR))   # 2 s lead-in before the voice
t = np.arange(N) / SR
meter = pyln.Meter(SR)

def fade(x, a=4.0, b=6.0):
    e = np.ones(len(x)); na, nb = int(a * SR), int(b * SR)
    e[:na] = np.linspace(0, 1, na) ** 2; e[-nb:] = np.linspace(1, 0, nb) ** 2
    return (x.T * e).T

def stereo(l, r): return np.stack([l, r], 1)

def reverb(x, secs=3.5, mix=0.35):          # synthetic decaying-noise impulse response
    n = int(secs * SR); ir = rng.standard_normal(n) * np.exp(-np.arange(n) / SR * 6.0 / secs)
    ir = signal.sosfilt(signal.butter(2, 5000, "low", fs=SR, output="sos"), ir); ir /= np.abs(ir).sum() ** 0.5 * 8
    wet = signal.fftconvolve(x, ir)[: len(x)]
    return (1 - mix) * x + mix * wet / (np.abs(wet).max() + 1e-9) * np.abs(x).max()

# --- rain: soft filtered noise with slow swells + sparse muted drops, decorrelated L/R
def rain_ch():
    n = rng.standard_normal(N)
    n = signal.sosfilt(signal.butter(2, [300, 5500], "band", fs=SR, output="sos"), n)
    swell = 1 + 0.25 * signal.sosfilt(signal.butter(1, 0.05, "low", fs=SR, output="sos"), rng.standard_normal(N)) * 40
    n *= np.clip(swell, 0.6, 1.4)
    drops = np.zeros(N); idx = rng.integers(0, N, size=int(N / SR * 6))
    drops[idx] = rng.uniform(0.2, 1, len(idx)) * rng.choice([-1, 1], len(idx))
    drops = signal.sosfilt(signal.butter(2, [800, 3000], "band", fs=SR, output="sos"), drops) * 4
    return n + drops
rain = stereo(rain_ch(), rain_ch())

# --- drone: tanpura-inspired soft pad on D (tonic + fifth + octave), slow shimmer, no buzz
def drone_ch(detune):
    x = np.zeros(N)
    for f, a in [(73.42, 1.0), (110.0, 0.55), (146.83, 0.45), (220.0, 0.12), (293.66, 0.08)]:
        lfo = 1 + 0.25 * np.sin(2 * np.pi * (0.05 + 0.03 * rng.random()) * t + rng.uniform(0, 6.28))
        x += a * lfo * np.sin(2 * np.pi * f * (1 + detune) * t + rng.uniform(0, 6.28))
    return signal.sosfilt(signal.butter(2, 900, "low", fs=SR, output="sos"), x)
drone = stereo(drone_ch(-0.0015), drone_ch(0.0015))

# --- sparse plucked notes (Karplus-Strong harp-like) only in gaps where the voice is silent > 1.5 s
def pluck(f, dur=4.0):
    n = int(dur * SR); p = int(SR / f); buf = rng.uniform(-1, 1, p); y = np.zeros(n)
    for i in range(n):
        y[i] = buf[i % p]; buf[i % p] = 0.5 * (buf[i % p] + buf[(i + 1) % p]) * 0.996
    return signal.sosfilt(signal.butter(2, 2500, "low", fs=SR, output="sos"), y) * np.exp(-np.arange(n) / SR * 0.9)
env = np.convolve(np.abs(v), np.ones(int(0.05 * SR)) / (0.05 * SR), "same")
quiet = env < 0.004
gaps, i = [], 0
while i < N:
    if quiet[i]:
        j = i
        while j < N and quiet[j]: j += 1
        if (j - i) / SR > 1.5: gaps.append((i, j))
        i = j
    else: i += 1
scale = [293.66, 329.63, 369.99, 440.0, 493.88, 587.33]          # D major pentatonic, mid register
notes = np.zeros((N, 2)); last = -99; placed = 0
for a, b in gaps:
    if a / SR - last < 25: continue                                # at most one phrase every ~25 s
    f = rng.choice(scale); s = pluck(f); s2 = pluck(rng.choice(scale)) * 0.6
    st = a + int(0.3 * SR)
    for k, (snd, off, pan) in enumerate([(s, 0, 0.4), (s2, int(0.9 * SR), 0.6)]):
        e = min(N, st + off + len(snd)); seg = snd[: e - st - off]
        notes[st + off: e, 0] += seg * (1 - pan); notes[st + off: e, 1] += seg * pan
    last = a / SR; placed += 1
notes = stereo(reverb(notes[:, 0]), reverb(notes[:, 1]))

def at_lufs(x, target):
    x = x if x.ndim > 1 else stereo(x, x)
    return x * 10 ** ((target - meter.integrated_loudness(x)) / 20)

VOICE = -23.0                       # integrated loudness of the voice alone
voice = at_lufs(v, VOICE)
layers = {"rain": at_lufs(fade(rain), VOICE - 20), "drone": at_lufs(fade(drone), VOICE - 24),
          "notes": at_lufs(notes, VOICE - 24) if placed else np.zeros((N, 2))}
mixes = {"A_voice_rain": ["rain"], "B_plus_drone": ["rain", "drone"], "C_plus_notes": ["rain", "drone", "notes"]}
report = {"seed": int(sys.argv[3]) if len(sys.argv) > 3 else 7, "voice_lufs": VOICE, "note_phrases": placed}
for name, keys in mixes.items():
    m = voice + sum(layers[k] for k in keys)
    m = at_lufs(m, VOICE)                                          # loudness-matched across A/B/C
    peak = float(np.abs(m).max()); m = m / max(1.0, peak / 0.89)
    sf.write(f"{out}/{name}.wav", m.astype(np.float32), SR)
    report[name] = {"lufs": round(meter.integrated_loudness(m), 1), "peak": round(peak, 3)}
json.dump(report, open(f"{out}/mix_report.json", "w"), indent=1); print(report)
