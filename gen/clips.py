"""First-and-last-frame clips with Wan 2.2 I2V A14B: every clip starts and ends on approved keyframes.
python3 clips.py spread04 -> /w/share/out/clips/spread04_c<n>_s<seed>.mp4"""
import sys, os, json, torch
from PIL import Image
from diffusers import WanImageToVideoPipeline
from diffusers.utils import export_to_video
OUT = '/w/share/out/clips'; os.makedirs(OUT, exist_ok=True)
touch = lambda: os.utime('/w/activity') if os.path.exists('/w/activity') else None
NEG = ('talking, lip movement, open mouth, chewing, extra legs, missing legs, extra arms, fused limbs, merged legs, melting, morphing, '
       'extra antennae, extra eyes, face distortion, color shift, camera shake, zoom, pan, cut, text, watermark, flicker, blurry, '
       'wings, static, still picture, low quality, deformed')
pipe = WanImageToVideoPipeline.from_pretrained('Wan-AI/Wan2.2-I2V-A14B-Diffusers', torch_dtype=torch.bfloat16)
import glob
from huggingface_hub import hf_hub_download, list_repo_files
FAST = os.environ.get('FAST') == '1'
if FAST:
    repo = 'lightx2v/Wan2.2-Lightning'
    files = [f for f in list_repo_files(repo) if 'I2V-A14B-4steps' in f and f.endswith('.safetensors')]
    print('lightning files', files, flush=True)
    hi = [f for f in files if 'high' in f][0]; lo = [f for f in files if 'low' in f][0]
    pipe.load_lora_weights(repo, weight_name=hi, adapter_name='fast_hi')
    pipe.load_lora_weights(repo, weight_name=lo, adapter_name='fast_lo', load_into_transformer_2=True)
    pipe.set_adapters(['fast_hi', 'fast_lo'], adapter_weights=[1.0, 1.0])
pipe.to('cuda')
W, H = 832, 480
L = lambda f: Image.open(f).convert('RGB').resize((W, H), Image.LANCZOS)
KF = '/w/share/kf'
plan = json.load(open(f'/w/share/gen/{sys.argv[1]}.json'))
for c in plan:
    for sd in c.get('seeds', [1]):
        out = f"{OUT}/{sys.argv[1]}_{c['id']}_s{sd}{'_fast' if FAST else ''}.mp4"
        if os.path.exists(out): continue
        fr = pipe(image=L(f"{KF}/{c['start']}"), last_image=L(f"{KF}/{c['end']}"), prompt=c['prompt'], negative_prompt=NEG,
                  height=H, width=W, num_frames=81, guidance_scale=(1.0 if FAST else 3.5), num_inference_steps=(4 if FAST else 30),
                  generator=torch.Generator('cuda').manual_seed(sd)).frames[0]
        export_to_video(fr, out, fps=16); touch(); print('saved', out, flush=True)
print('DONE clips', flush=True)
