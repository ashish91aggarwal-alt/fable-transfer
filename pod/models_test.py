"""Render the passage with one model (argv[1] = qwen | voxcpm) using a text voice description. One call for the whole passage."""
import sys, os, soundfile as sf, torch
which, src, out = sys.argv[1], sys.argv[2], sys.argv[3]
text = " ".join(p.strip() for p in open(f"{src}/passage.txt").read().split("\n\n") if p.strip())
desc = open(f"{src}/voice.txt").read().strip()
os.makedirs(out, exist_ok=True)
if which == "qwen":
    from qwen_tts import Qwen3TTSModel
    m = Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign", device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="sdpa")
    for seed in (1, 2):
        torch.manual_seed(seed)
        wavs, sr = m.generate_voice_design(text=text, language="English", instruct=desc)
        sf.write(f"{out}/qwen3_seed{seed}.wav", wavs[0], sr); print("qwen", seed, len(wavs[0]) / sr, flush=True)
else:
    from voxcpm import VoxCPM
    m = VoxCPM.from_pretrained("openbmb/VoxCPM2", load_denoiser=False)
    for seed in (1, 2):
        wav = m.generate(text=f"({desc}){text}", cfg_value=2.0, inference_timesteps=10, seed=seed)
        sf.write(f"{out}/voxcpm2_seed{seed}.wav", wav, m.tts_model.sample_rate); print("voxcpm", seed, len(wav) / m.tts_model.sample_rate, flush=True)
