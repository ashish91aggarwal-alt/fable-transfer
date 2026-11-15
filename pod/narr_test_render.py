"""Render narrator test jobs with Chatterbox. Each job: text file, reference clip, settings. Paragraph pauses 1.2 s."""
import sys, json, subprocess, torch, torchaudio as ta
from chatterbox.tts import ChatterboxTTS
src, out = sys.argv[1], sys.argv[2]
m = ChatterboxTTS.from_pretrained(device="cuda"); torch.manual_seed(7)
for j in json.load(open(f"{src}/jobs.json")):
    parts = []
    for para in [p.strip() for p in open(f"{src}/{j['text']}").read().split("\n\n") if p.strip()]:
        parts += [m.generate(para, audio_prompt_path=f"{src}/{j['ref']}", exaggeration=j["exaggeration"], cfg_weight=j["cfg"]), torch.zeros(1, int(m.sr * 1.2))]
    w = torch.cat(parts, 1); ta.save(f"{out}/{j['out']}.wav", w, m.sr)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", f"{out}/{j['out']}.wav", "-b:a", "128k", f"{out}/{j['out']}.mp3"], check=True)
    print(j["out"], round(w.shape[1] / m.sr, 1), flush=True)
