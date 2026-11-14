"""Edit-based painting: every new picture is an edit of an approved painting, with the character sheet as
a second reference, so the character cannot drift (no wings, right legs). bf16 on an 80 GB card.
python3 edit.py jobs   -> /w/share/out/edit/<name>_s<seed>.png
python3 edit.py kf     -> /w/share/out/kf/spread04_k<n>_s<seed>.png (chained from the approved painting)"""
import sys, os, torch
from PIL import Image
from diffusers import QwenImageEditPlusPipeline
IN = '/w/share/inputs'; OUT = '/w/share/out'
touch = lambda: os.utime('/w/activity') if os.path.exists('/w/activity') else None
NEG = 'wings, hair, crest, extra legs, missing legs, extra arms, fused limbs, deformed, blurry, text, watermark, photo, 3d render, open mouth'
KEEP = ' Keep the same soft painted picture-book style, the same lighting and colors. The grasshopper must look exactly like the grasshopper in the reference sheet: olive green, mustard knitted scarf, round friendly face, no wings, no hair, no crest, two arms, two short middle legs and two long bent hind legs. The ant must look exactly like the ant reference: rust-red, acorn-cap hat, six thin legs. Every leg clearly visible and separate.'
p = QwenImageEditPlusPipeline.from_pretrained('Qwen/Qwen-Image-Edit-2509', torch_dtype=torch.bfloat16).to('cuda')
def run(images, instr, path, seed, size):
    im = p(image=images, prompt=instr + KEEP, negative_prompt=NEG, num_inference_steps=40, true_cfg_scale=4.0,
           generator=torch.Generator('cuda').manual_seed(seed)).images[0].resize(size, Image.LANCZOS)
    im.save(path); touch(); print('saved', path, flush=True); return im
L = lambda f: Image.open(f'{IN}/{f}').convert('RGB')
G, A = L('refs/gus_clean.png'), L('refs/ana_ref.png')
if sys.argv[1] == 'jobs':
    os.makedirs(f'{OUT}/edit', exist_ok=True)
    jobs = {
      'gus_laugh': ([L('antgrass_shot03.png'), G], 'Picture 1 is the scene, picture 2 is the reference sheet. In picture 1, make the grasshopper sit up and laugh happily: eyes squeezed shut, mouth closed in a wide smile, shoulders raised. Keep the clover meadow.'),
      'gus_dance': ([L('antgrass_shot09.png'), G], 'Picture 1 is the scene, picture 2 is the reference sheet. Redraw the grasshopper in picture 1 so he matches the reference sheet exactly (remove the wings, remove the orange crest, give him the round friendly face). He crouches on his long bent hind legs and sways happily holding the small guitar among the falling autumn leaves.'),
      'gus_snow': ([L('antgrass_shot11.png'), G], 'Picture 1 is the scene, picture 2 is the reference sheet. Make the grasshopper match the reference sheet exactly, standing in the snow hugging himself and shivering, sad eyes, mouth closed. Make the picture clearer with stronger contrast: deep navy evening sky, bright white snow, crisp shapes.'),
      'ana_pantry': ([L('pantry_s2.png'), A], 'Picture 1 is the scene, picture 2 is the reference sheet. Only change the ant\'s mouth: close it into a small proud smile. Keep everything else identical.'),
    }
    for name, (imgs, instr) in jobs.items():
        for sd in (1, 2): run(imgs, instr, f'{OUT}/edit/{name}_s{sd}.png', sd, imgs[0].size)
elif sys.argv[1] == 'kf':
    os.makedirs(f'{OUT}/kf', exist_ok=True)
    base = L('start_04.png'); base.save(f'{OUT}/kf/spread04_k1.png')
    steps = [('k2', 'Picture 1 is the scene, picture 2 and 3 are reference sheets. In picture 1, the grasshopper lowers his waving arm to his side and smiles with mouth closed; the ant turns her head slightly to one side as if gently shaking her head no, still holding her seed. Change nothing else: same positions, same background, same framing.'),
             ('k3', 'Picture 1 is the scene, picture 2 and 3 are reference sheets. In picture 1, the grasshopper laughs with eyes squeezed shut and mouth closed in a wide smile. Change nothing else.'),
             ('k4', 'Picture 1 is the scene, picture 2 and 3 are reference sheets. In picture 1, the ant takes one small step to the right along the path still holding her seed and looks back over her shoulder with a small kind smile; the grasshopper sits relaxed and smiling. Change nothing else.')]
    prev = base
    for name, instr in steps:
        outs = [run([prev, G, A], instr, f'{OUT}/kf/spread04_{name}_s{sd}.png', sd, base.size) for sd in (1, 2)]
        prev = outs[0]
print('DONE', sys.argv[1], flush=True)
