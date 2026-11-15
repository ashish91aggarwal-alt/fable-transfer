"""Render a cast scene with Chatterbox: each line uses its character's reference clip and expressiveness."""
import sys, json, subprocess, torch, torchaudio as ta
from chatterbox.tts import ChatterboxTTS
src, out = sys.argv[1], sys.argv[2]
sc = json.load(open(f"{src}/scene.json"))
m = ChatterboxTTS.from_pretrained(device="cuda"); torch.manual_seed(7)
for version, cfg in [("chatterbox_cast", 0.4), ("chatterbox_cast_softer", 0.3)]:
    parts = []
    for spk, text in sc["lines"]:
        c = sc["cast"][spk]
        ex = c["exaggeration"] if version == "chatterbox_cast" else max(0.25, c["exaggeration"] - 0.15)
        w = m.generate(text, audio_prompt_path=f"{src}/ref_{spk}.wav", exaggeration=ex, cfg_weight=cfg)
        parts += [w, torch.zeros(1, int(m.sr * (1.1 if spk != "narrator" else 0.9)))]
    wav = torch.cat(parts, dim=1); ta.save(f"{out}/{version}.wav", wav, m.sr)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", f"{out}/{version}.wav", "-af", "atempo=0.9", "-b:a", "96k", f"{out}/{version}.mp3"], check=True)
    print(version, round(wav.shape[1] / m.sr, 1), "s", flush=True)
