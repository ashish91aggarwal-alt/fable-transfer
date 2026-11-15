"""Paint with FLUX.2 Klein 4B (Apache-2.0). New pictures (generate) and small edits of an approved picture (edit).

    python3 klein.py generate jobs.json | edit jobs.json | dry jobs.json   (dry = print prompts only, no GPU)

jobs.json (paths relative to it):
{ "out":"out", "w":1344, "h":768, "steps":4, "guidance":1.0,
  "prefix":"children's picture-book illustration, soft gouache and watercolor texture, warm golden light, ",
  "suffix":", no text",
  "generate":[{"name":"s1","seeds":[1,2],"scene":"..."}],
  "edit":[{"name":"tilly_side","base":"refs/T1.jpg","seeds":[1,2,3],"change":"..."},
         {"name":"scene_tilly","base":"scene.jpg","refs":["cast/tilly.jpg"],"seeds":[1,2],"change":"...","raw":true}] }
An edit with "place": "hugo|hugo_asleep|tilly|wren" appends the fixed size, count and facing wording (never leave it out).
An edit with "raw": true sends its "change" text as the whole prompt (needed when image 2 is a character to place).
Edit prompt = the change, then "Keep everything else exactly the same." One image in, one change. Writes <name>_s<seed>.png,
timings.txt (load seconds, seconds per picture, peak VRAM) and touches /w/activity per picture (idle kill switch).
Offload is NEVER silent: set OFFLOAD=1 to enable it. Pictures that already exist are skipped (safe to rerun).
"""
import sys, os, json, time
mode, jp = sys.argv[1], sys.argv[2]
J = json.load(open(jp)); D = os.path.dirname(os.path.abspath(jp))
rel = lambda p: p if os.path.isabs(p) else os.path.join(D, p)
OUT = rel(J.get('out', 'out')); os.makedirs(OUT, exist_ok=True)
W, H = J.get('w', 1344), J.get('h', 768)
def gprompt(j): return f"{J.get('prefix','')}{j['scene']}{J.get('suffix','')}"
GUARDS = {  # mandatory wording for placing a character into a scene (G4 review, 15 Nov 2026); add "place": "<key>" to the job
 "hugo": " Hugo stands upright at the same distance from the viewer as the tortoise, so his feet are level with hers; his head is as high as three tortoise shells stacked and the tortoise only reaches his knees. Exactly two animals: one hare and one tortoise, no baby animals, no extra tail. His feet are clearly visible on bare ground or short grass, no flowers in front of him.",
 "hugo_asleep": " Copy the curled sleeping hare from image 2 as it is. Exactly one hare, no other animals.",
 "tilly": " The tortoise faces right. Add exactly one tortoise and keep every animal already in the picture exactly as it is.",
 "wren": " Add exactly one tiny wren and keep every animal already in the picture."}
def eprompt(j):
    if j.get('raw'): return j['change'] + GUARDS.get(j.get('place'), '')
    return f"{j['change']}. Keep everything else exactly the same: same character, colors, painting style and background."
if mode == 'dry':
    for j in J.get('edit', []): print('EDIT', j['name'], j['seeds'], '<-', j['base'], '|', eprompt(j))
    for j in J.get('generate', []): print('GEN ', j['name'], j['seeds'], '|', gprompt(j))
    n = sum(len(j['seeds']) for k in ('edit', 'generate') for j in J.get(k, []))
    print('TOTAL pictures:', n, ' size', W, 'x', H); sys.exit(0)
import torch
from PIL import Image
from diffusers import Flux2KleinPipeline
t0 = time.time()
pipe = Flux2KleinPipeline.from_pretrained('black-forest-labs/FLUX.2-klein-4B', torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload() if os.environ.get('OFFLOAD') == '1' else pipe.to('cuda')
load_s = time.time() - t0
touch = lambda: os.utime('/w/activity') if os.path.exists('/w/activity') else None
log = open(os.path.join(OUT, 'timings.txt'), 'a'); log.write(f'load_seconds {load_s:.0f}\n'); log.flush()
def run(name, seed, w=None, h=None, **kw):
    fp = os.path.join(OUT, f'{name}_s{seed}.png')
    if os.path.exists(fp): return
    t = time.time()
    img = pipe(height=h or H, width=w or W, num_inference_steps=J.get('steps', 4), guidance_scale=J.get('guidance', 1.0),
               generator=torch.Generator('cuda').manual_seed(seed), **kw).images[0]
    img.save(fp); touch()
    log.write(f'{name}_s{seed} {time.time()-t:.1f}s peak_vram_gb {torch.cuda.max_memory_allocated()/1e9:.1f}\n'); log.flush()
if mode in ('edit', 'both'):
    for j in J.get('edit', []):
        for s in j['seeds']:
            # base may be "out:<name>" = the same seed's output of an earlier job (chains: scene -> +Tilly -> +Hugo)
            bp = j['base']; bp = os.path.join(OUT, f"{bp[4:]}_s{s}.png") if bp.startswith('out:') else rel(bp)
            if not os.path.exists(bp): continue
            base = Image.open(bp).convert('RGB'); bw, bh = (base.width // 16) * 16, (base.height // 16) * 16  # keep the source shape: never stretch
            imgs = [base] + [Image.open(rel(r)).convert('RGB') for r in j.get('refs', [])]  # refs: ONE character picture at a time, never two
            run(j['name'], s, w=bw, h=bh, image=imgs if len(imgs) > 1 else base, prompt=eprompt(j))
if mode in ('generate', 'both'):
    for j in J.get('generate', []):
        for s in j['seeds']: run(j['name'], s, prompt=gprompt(j))
