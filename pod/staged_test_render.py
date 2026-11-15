"""Render staged-scene lines with Chatterbox, one file per line, plus timeline. SFX markers are skipped (mixed later)."""
import sys, json, os, subprocess, torch, torchaudio as ta
from chatterbox.tts import ChatterboxTTS
src, out = sys.argv[1], sys.argv[2]
m = ChatterboxTTS.from_pretrained(device="cuda"); torch.manual_seed(7)
for name in ["A", "B"]:
    sc = json.load(open(f"{src}/{name}_scene.json")); d = f"{out}/{name}_lines"; os.makedirs(d, exist_ok=True)
    for i, (spk, text) in enumerate(sc["lines"]):
        if spk == "sfx": continue
        w = m.generate(text, audio_prompt_path=f"{src}/ref_{spk}.wav", exaggeration=sc["cast"][spk], cfg_weight=0.4)
        ta.save(f"{d}/{i:03d}_{spk}.wav", w, m.sr)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", f"{d}/{i:03d}_{spk}.wav", "-af", "atempo=0.9", "-b:a", "128k", f"{d}/{i:03d}_{spk}.mp3"], check=True)
        os.remove(f"{d}/{i:03d}_{spk}.wav"); print(name, i, spk, flush=True)
    subprocess.run(["tar", "-cf", f"{out}/{name}_lines.tar", "-C", out, f"{name}_lines"], check=True)
