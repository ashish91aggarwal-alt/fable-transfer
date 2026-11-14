"""Adds character close-up crops to the dataset: Qwen2.5-VL finds each character box on every scene image,
then a padded square-ish crop is saved with a character caption. Run on the pod after the model is downloaded."""
import json, os, sys, glob
sys.path.insert(0, '/w/repo/pipe'); import checks2
from PIL import Image
T = json.load(open('/w/repo/lora/triggers.json')); D = '/w/lora_data'
for p in sorted(glob.glob(f'{D}/s*.png')):
    boxes, (W, H) = checks2.char_boxes(p)
    im = Image.open(p).convert('RGB')
    for i, (lab, b) in enumerate(boxes):
        x1, y1, x2, y2 = b; w, h = x2 - x1, y2 - y1
        if w < 60 or h < 60: continue
        pad = 0.25; cx, cy = (x1 + x2) / 2, (y1 + y2) / 2; s = max(w, h) * (1 + pad)
        box = [int(max(0, cx - s / 2)), int(max(0, cy - s / 2)), int(min(W, cx + s / 2)), int(min(H, cy + s / 2))]
        c = im.crop(box); 
        if min(c.size) < 200: continue
        who = T['A'] if 'ant' in lab else T['G'] if 'grass' in lab else None
        if not who: continue
        name = f'{D}/{os.path.basename(p)[:-4]}_c{i}'
        c.save(name + '.png'); open(name + '.txt', 'w').write(f"{T['S']}. {who}, close-up.")
        print('crop', name, lab, box)
