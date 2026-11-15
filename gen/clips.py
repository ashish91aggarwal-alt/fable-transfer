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
pipe.to('cuda')
W, H = 832, 480
L = lambda f: Image.open(f).convert('RGB').resize((W, H), Image.LANCZOS)
KF = '/w/share/kf'
plan = json.load(open(f'/w/share/gen/{sys.argv[1]}.json'))
for c in plan:
    for sd in c.get('seeds', [1]):
        out = f"{OUT}/{sys.argv[1]}_{c['id']}_s{sd}.mp4"
        if os.path.exists(out): continue
        fr = pipe(image=L(f"{KF}/{c['start']}"), last_image=L(f"{KF}/{c['end']}"), prompt=c['prompt'], negative_prompt=NEG,
                  height=H, width=W, num_frames=81, guidance_scale=3.5, num_inference_steps=30,
                  generator=torch.Generator('cuda').manual_seed(sd)).frames[0]
        export_to_video(fr, out, fps=16); touch(); print('saved', out, flush=True)
print('DONE clips', flush=True)
