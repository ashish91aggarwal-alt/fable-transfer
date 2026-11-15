"""Finish the voice test on the pod: neutral names, mp3s, ASR failure check, index.html.
Env MAP = json {"chatterbox_calm": "voice-1", ...} (kept out of the repo so the test stays blind)."""
import os, re, json, glob, subprocess, difflib, html
SH = "/w/share"; REPO = "/w/repo/channels/sleep-stories/voice-test"
MAP = json.loads(os.environ["MAP"])
os.makedirs(f"{SH}/s", exist_ok=True)
for src, dst in MAP.items():                       # pod renders -> neutral mp3
    for w in glob.glob(f"{SH}/{src}_*.wav"):
        part = os.path.basename(w)[len(src) + 1:-4]
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", w, "-af", os.environ.get("TEMPO", "atempo=0.85"), "-b:a", "96k", f"{SH}/s/{dst}_{part}.mp3"], check=True)  # slow to sleep pace, pitch kept
for m in glob.glob(f"{REPO}/samples/*.mp3"):       # samples made elsewhere (pushed to repo)
    subprocess.run(["cp", m, f"{SH}/s/"])
from faster_whisper import WhisperModel
asr = WhisperModel("small.en", device="cuda", compute_type="float16")
norm = lambda t: re.findall(r"[a-z']+", t.lower().replace("’", "'"))
report = {}
for f in sorted(glob.glob(f"{SH}/s/*.mp3")):
    name = os.path.basename(f)[:-4]; part = name.split("_", 1)[1]
    ref = norm(open(f"{REPO}/{part}.txt").read())
    segs, info = asr.transcribe(f, beam_size=5, vad_filter=False)
    hyp = norm(" ".join(s.text for s in segs))
    sm = difflib.SequenceMatcher(a=ref, b=hyp, autojunk=False); missing, extra = [], []
    for op, a0, a1, b0, b1 in sm.get_opcodes():
        if op in ("delete", "replace"): missing.append(" ".join(ref[a0:a1]))
        if op in ("insert", "replace"): extra.append(" ".join(hyp[b0:b1]))
    errs = sum(max(a1 - a0, b1 - b0) for op, a0, a1, b0, b1 in sm.get_opcodes() if op != "equal")
    dur = float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", f]))
    report[name] = dict(duration_s=round(dur, 1), words_per_min=round(len(ref) / dur * 60), word_error_pct=round(100 * errs / max(1, len(ref)), 1),
                        missing=missing[:25], extra=extra[:25])
    print(name, report[name]["word_error_pct"], flush=True)
json.dump(report, open(f"{SH}/asr_report.json", "w"), indent=1)
PART = {"A_opening": "1. Opening welcome", "B_final_fade": "2. Final fade to sleep", "C_stress": "3. Stress test (names, dialogue, odd words, repeats)", "D_long_drift": "4. Long passage, about 10 min (drift test)"}
voices = sorted({n.split("_", 1)[0] for n in report})
rows = []
for p, title in PART.items():
    cells = "".join(f'<div class="v"><b>{v}</b><audio controls preload="none" src="s/{v}_{p}.mp3"></audio></div>' for v in voices if f"{v}_{p}" in report)
    if cells: rows.append(f"<h2>{title}</h2>{cells}")
open(f"{SH}/index.html", "w").write("""<!doctype html><meta name=viewport content="width=device-width,initial-scale=1"><title>Voice test</title>
<style>body{font-family:system-ui;max-width:640px;margin:auto;padding:12px;background:#14161c;color:#e8e6e1}h2{font-size:1.05em;margin-top:1.6em}.v{margin:.6em 0}audio{width:100%}</style>
<h1>Sleep-story voice test (blind)</h1><p>Listen with eyes closed. Pick the voice you would fall asleep to, and note any voice that stumbles in part 3.</p>""" + "".join(rows))
print("PAGE DONE", flush=True)
