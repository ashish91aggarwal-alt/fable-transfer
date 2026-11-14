"""Repaints (Qwen-Image + our LoRA) and keyframe edits (Qwen-Image-Edit-2509 from an approved painting).
python3 paint.py repaint  -> /w/share/out/repaint/*.png  (3 seeds each)
python3 paint.py keyframes -> /w/share/out/kf/spread04_k*.png (chained edits, 2 seeds each, best picked later)"""
import sys, os, json, torch, glob
from PIL import Image
T = json.load(open('/w/share/lora/triggers.json')); S, A, G = T['S'], T['A'], T['G']
OUT = '/w/share/out'; os.makedirs(OUT, exist_ok=True)
touch = lambda: os.utime('/w/activity')
NEG = 'blurry, text, watermark, extra legs, missing legs, extra arms, fused limbs, deformed, wings, hair, photo, 3d render'
def lora_path():
    c = sorted(glob.glob('/w/lora_out/fable_ant_grass_v1/*.safetensors')); return c[-1]
if sys.argv[1] == 'repaint':
    from diffusers import QwenImagePipeline
    p = QwenImagePipeline.from_pretrained('Qwen/Qwen-Image', torch_dtype=torch.bfloat16)
    p.enable_model_cpu_offload()
    p.load_lora_weights(lora_path())
    jobs = {
     'p07_gus_laugh': f'{S}. close-up of {G} laughing happily in the clover, eyes squeezed shut in joy, mouth closed in a wide smile, round friendly face, summer meadow behind, mustard scarf, no wings, no hair.',
     'p06_pantry': f'{S}. a warm underground pantry lit by a glowing lantern: wooden shelves stacked with baskets of seeds, nuts and berries; {A} with rust-red body sets down a small basket, proud smile, every leg separate.',
     'p07_dance': f'{S}. autumn meadow with falling orange leaves: {G} dances on his long hind legs holding a small guitar, happy face, mustard scarf flying, no wings, no hair, no crest, every leg clearly separate.',
     'p09_snow': f'{S}. winter: {G} stands alone in deep snow in the open meadow, hugging himself and shivering, sad eyes, mustard scarf, snow falling, farmhouse and picket fence far behind, strong clear contrast, navy and white palette.'}
    os.makedirs(f'{OUT}/repaint', exist_ok=True)
    for k, pr in jobs.items():
        for sd in (1, 2, 3):
            im = p(prompt=pr, negative_prompt=NEG, width=1344, height=768, num_inference_steps=40, true_cfg_scale=4.0, generator=torch.Generator('cuda').manual_seed(sd)).images[0]
            im.save(f'{OUT}/repaint/{k}_s{sd}.png'); touch(); print('saved', k, sd, flush=True)
elif sys.argv[1] == 'keyframes':
    from diffusers import QwenImageEditPlusPipeline
    p = QwenImageEditPlusPipeline.from_pretrained('Qwen/Qwen-Image-Edit-2509', torch_dtype=torch.bfloat16)
    p.enable_model_cpu_offload()
    keep = ' Keep everything else exactly the same: same painting style, same background, same lighting, same character designs and colors, same framing. Only change what is described.'
    steps = [('k2', 'Gus the green grasshopper lowers his waving arm and smiles; Ana the ant turns her head slightly to one side as if gently shaking her head no, still holding her seed, kind face, mouth closed.'),
             ('k3', 'Gus the green grasshopper laughs with his eyes squeezed shut and shoulders raised, mouth closed in a wide smile; Ana unchanged.'),
             ('k4', 'Ana the ant takes one step to the right along the path, still holding her seed, looking back over her shoulder with a small kind smile; Gus relaxed and smiling, mouth closed. Every leg clearly visible and separate.')]
    os.makedirs(f'{OUT}/kf', exist_ok=True)
    prev = Image.open('/w/share/inputs/start_04.png').convert('RGB'); prev.save(f'{OUT}/kf/spread04_k1.png')
    for name, instr in steps:
        best = None
        for sd in (1, 2):
            im = p(image=[prev], prompt=instr + keep, negative_prompt=NEG, num_inference_steps=40, true_cfg_scale=4.0, generator=torch.Generator('cuda').manual_seed(sd)).images[0]
            im = im.resize(prev.size); im.save(f'{OUT}/kf/spread04_{name}_s{sd}.png'); touch(); print('saved', name, sd, flush=True)
            if best is None: best = im
        prev = best   # chain from seed 1 by default; reviewer may swap
