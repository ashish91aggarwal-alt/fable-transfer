"""Render voice-test passages with Chatterbox at sleep-story settings. Writes WAVs + timing log."""
import sys, os, re, time, json, torch, torchaudio as ta
from chatterbox.tts import ChatterboxTTS
src, out = sys.argv[1], sys.argv[2]
m = ChatterboxTTS.from_pretrained(device="cuda")
SETTINGS = {"calm": dict(exaggeration=0.3, cfg_weight=0.3), "default": dict(exaggeration=0.5, cfg_weight=0.5)}
def chunks(text):
    for para in [p.strip() for p in text.split("\n\n") if p.strip()]:
        sents = re.split(r'(?<=[.!?"])\s+', para); cur = ""
        for s in sents:
            if len(cur) + len(s) > 280 and cur: yield cur, False; cur = s
            else: cur = (cur + " " + s).strip()
        if cur: yield cur, True
log = {}
for name in ["A_opening", "B_final_fade", "C_stress", "D_long_drift"]:
    text = open(f"{src}/{name}.txt").read()
    for sname, kw in SETTINGS.items():
        if name == "D_long_drift" and sname != "calm": continue
        t0 = time.time(); parts = []
        for c, para_end in chunks(text):
            w = m.generate(c, **kw); parts.append(w)
            parts.append(torch.zeros(1, int(m.sr * (1.4 if para_end else 0.4))))
        wav = torch.cat(parts, dim=1)
        f = f"{out}/chatterbox_{sname}_{name}.wav"; ta.save(f, wav, m.sr)
        log[os.path.basename(f)] = dict(audio_s=round(wav.shape[1] / m.sr, 1), render_s=round(time.time() - t0, 1))
        print(log[os.path.basename(f)], f, flush=True); os.utime("/w/activity")
json.dump(log, open(f"{out}/timing.json", "w"), indent=1)
