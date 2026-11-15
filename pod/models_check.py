"""Gate checks on every rendered wav: words vs script (faster-whisper), pace, median pitch, loudness spread."""
import sys, glob, re, json, difflib, numpy as np, librosa
from faster_whisper import WhisperModel
src, out = sys.argv[1], sys.argv[2]
ref = re.findall(r"[a-z']+", open(f"{src}/passage.txt").read().lower())
asr = WhisperModel("small.en", device="cuda", compute_type="float16"); rep = {}
for f in sorted(glob.glob(f"{out}/*.wav")):
    segs, _ = asr.transcribe(f, beam_size=5); hyp = re.findall(r"[a-z']+", " ".join(s.text for s in segs).lower())
    sm = difflib.SequenceMatcher(a=ref, b=hyp, autojunk=False)
    err = sum(max(a1 - a0, b1 - b0) for op, a0, a1, b0, b1 in sm.get_opcodes() if op != "equal")
    y, sr = librosa.load(f, sr=16000); f0, v, _ = librosa.pyin(y, fmin=70, fmax=400, sr=sr, frame_length=1024)
    rms = librosa.feature.rms(y=y)[0]; rms = rms[rms > 0.01]
    rep[f.split("/")[-1]] = dict(seconds=round(len(y) / sr, 1), wpm=round(len(ref) / (len(y) / sr) * 60), word_err_pct=round(100 * err / len(ref), 1),
        median_pitch_hz=round(float(np.nanmedian(f0[v]))), loudness_spread_db=round(float(np.std(20 * np.log10(rms))), 1),
        missing=[" ".join(ref[a0:a1]) for op, a0, a1, b0, b1 in sm.get_opcodes() if op in ("delete", "replace")][:10])
    print(f, rep[f.split("/")[-1]], flush=True)
json.dump(rep, open(f"{out}/checks.json", "w"), indent=1)
