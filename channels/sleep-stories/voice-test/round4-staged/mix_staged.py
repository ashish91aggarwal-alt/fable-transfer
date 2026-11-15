"""Mix staged-scene test: voice lines + soft synthesised SFX + rain (outside) -> fire (inside) bed.
Usage: python3 mix_staged.py <A|B> <lines_dir> <fire_bed.wav> <out.wav>
SFX are synthesised here (no recordings). The fire bed is a CC0 recording (see LEDGER.md)."""
import sys, json, glob, os, subprocess, numpy as np, soundfile as sf, pyloudnorm as pyln
from scipy import signal
SR = 44100; name, ldir, fire_path, out = sys.argv[1:5]
here = os.path.dirname(os.path.abspath(__file__)); rng = np.random.default_rng(11)
sc = json.load(open(f"{here}/{name}_scene.json"))
lp = lambda x, f: signal.sosfilt(signal.butter(2, f, "low", fs=SR, output="sos"), x)
bp = lambda x, a, b: signal.sosfilt(signal.butter(2, [a, b], "band", fs=SR, output="sos"), x)
env = lambda n, a, r: np.minimum(1, np.minimum(np.arange(n) / max(1, a * SR), (n - np.arange(n)) / max(1, r * SR)))

def load(path):
    d = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", path, "-f", "f32le", "-ac", "1", "-ar", str(SR), "-"], capture_output=True).stdout
    return np.frombuffer(d, np.float32).copy()

def steps(n, rate, lo, hi, dur):          # footsteps: soft bursts at a walking pace
    N = int(dur * SR); x = np.zeros(N); t = 0.3
    while t < dur - 0.4:
        L = int(rng.uniform(0.08, 0.14) * SR); p = int(t * SR)
        x[p:p + L] += rng.standard_normal(L) * np.exp(-np.arange(L) / (L / 3)) * rng.uniform(0.6, 1)
        t += 1 / rate * rng.uniform(0.9, 1.1)
    return bp(x, lo, hi) * env(N, 0.5, 0.8)

def creak(dur, f0):                        # soft wooden creak: slowly gliding, rough resonance
    N = int(dur * SR); t = np.arange(N) / SR
    f = f0 * (1 + 0.15 * np.sin(2 * np.pi * 0.7 * t)); ph = 2 * np.pi * np.cumsum(f) / SR
    x = signal.sawtooth(ph) * (1 + 0.5 * rng.standard_normal(N) * 0.3)
    return bp(x, 300, 1800) * env(N, 0.3, 0.4) * 0.5

def thud(dur=0.25, f=110):
    N = int(dur * SR); t = np.arange(N) / SR
    return (np.sin(2 * np.pi * f * t) + 0.3 * lp(rng.standard_normal(N), 600)) * np.exp(-t / 0.05)

def purr(dur=7.0):                         # cat purr: ~26 Hz pulses of low breathy noise, in and out breaths
    N = int(dur * SR); t = np.arange(N) / SR
    pulses = (0.5 + 0.5 * np.sin(2 * np.pi * 26 * t)) ** 3
    breath = 0.6 + 0.4 * np.sin(2 * np.pi * 0.45 * t)
    return lp(rng.standard_normal(N), 350) * pulses * breath * env(N, 1.0, 1.5)

def rustle(dur, lo=800, hi=4000):
    N = int(dur * SR); sw = np.clip(lp(rng.standard_normal(N), 3) * 30, 0, None)
    return bp(rng.standard_normal(N), lo, hi) * sw * env(N, 0.3, 0.5)

SFX = {  # name: (sound, gain dB relative to voice level, overlap the next line in seconds)
    "gravel": (lambda: steps(0, 1.7, 1500, 6000, 4.0), -22, 2.0),
    "gate": (lambda: np.concatenate([creak(1.2, 180), thud(0.3, 140) * 0.6]), -26, 0.5),
    "steps_inside": (lambda: steps(0, 1.5, 80, 400, 2.6), -20, 0.0),
    "door": (lambda: np.concatenate([thud(0.15, 220) * 0.5, creak(1.6, 140)]), -26, 0.6),
    "coat": (lambda: rustle(1.4), -30, 0.7),
    "room": (lambda: np.zeros(int(0.3 * SR)), -40, 0.0),
    "cat_stretch": (lambda: rustle(1.8, 300, 2000), -30, 1.2),
    "purr": (lambda: purr(9.0), -18, 7.5),
    "tray": (lambda: thud(0.3, 160) + 0.2 * np.pad(thud(0.2, 900), (0, int(0.1 * SR))), -26, 0.2),
    "mug": (lambda: thud(0.25, 240) * 0.8, -28, 0.1),
}

meter = pyln.Meter(SR)
files = {int(os.path.basename(f)[:3]): f for f in glob.glob(f"{ldir}/*.mp3")}
voice, sfx, cursor, door_at = [], [], 2.0 * SR, None
for i, (spk, text) in enumerate(sc["lines"]):
    if spk == "sfx":
        snd, g, overlap = SFX[text]; s = snd(); s = s / (np.abs(s).max() + 1e-9)
        sfx.append((int(cursor), s, g))
        if text == "door": door_at = int(cursor)
        cursor += len(s) - overlap * SR
        continue
    if name == "A" and door_at is None and "door opened" in text: door_at = int(cursor)
    v = load(files[i]); voice.append((int(cursor), v))
    gap = 1.1 if spk != "narrator" else 0.8
    nxt = sc["lines"][i + 1][0] if i + 1 < len(sc["lines"]) else None
    if nxt and nxt != "sfx" and nxt != spk and spk != "narrator" and nxt != "narrator": gap = 0.55   # quick back-and-forth in dialogue
    cursor += len(v) + gap * SR
N = int(cursor + 6 * SR)
V = np.zeros(N)
for p, v in voice: V[p:p + len(v)] += v
VL = meter.integrated_loudness(np.stack([V, V], 1))
V *= 10 ** ((-23 - VL) / 20)
S = np.zeros(N)
for p, s, g in sfx:
    e = min(N, p + len(s)); S[p:e] += s[: e - p] * 10 ** ((-23 + g + 3) / 20) * 0.5   # peak-normalised sfx scaled by gain
# beds: rain outside until the door, then fire inside with faint rain at the window
rain = bp(rng.standard_normal(N), 300, 5000)
f = load(fire_path); reps = int(np.ceil(N / len(f))) + 1; cf = int(2 * SR)
fire = np.zeros(len(f) * reps)
for k in range(reps):                      # loop the fire with 2 s crossfades
    seg = f.copy(); seg[:cf] *= np.linspace(0, 1, cf); seg[-cf:] *= np.linspace(1, 0, cf)
    st = k * (len(f) - cf); fire[st:st + len(f)] += seg
fire = fire[:N]
door_at = door_at or int(N * 0.15); x = np.arange(N); xf = int(3 * SR)
w_out = np.clip(1 - (x - door_at) / xf, 0, 1); w_in = 1 - w_out
def lvl(sig, db): return sig * 10 ** ((db - meter.integrated_loudness(np.stack([sig, sig], 1))) / 20)
bed = lvl(rain, -43) * (w_out + 0.25 * w_in) + lvl(fire, -42) * w_in
mix = V + S + bed
mix = np.stack([mix, mix], 1); mix *= 10 ** ((-23 - meter.integrated_loudness(mix)) / 20); mix /= max(1, np.abs(mix).max() / 0.89)
sf.write(out, mix.astype(np.float32), SR); print(out, round(N / SR, 1), "s")
