"""GPU checker v2: cheap technical checks + SAM2 character tracking + Qwen2.5-VL video judge.
Library use: import checks2; checks2.run(clip, start, end) -> report dict.
Batch use:   python3 checks2.py jobs.jsonl out.jsonl   (each job: {"clip","start","end", ...extra kept})"""
import json, os, sys, re, tempfile, subprocess
import numpy as np, cv2
from skimage.metrics import structural_similarity as ssim
from skimage.morphology import skeletonize

C = {"fps_ok": [15, 31], "min_frames": 40, "min_dur": 2.5, "max_dur": 11.0,
     "start_ssim": 0.55, "end_ssim": 0.28, "black_luma": 12, "flicker_jump": 14.0,
     "area_tol": 0.30, "area_jump": 0.15, "frag_jump": 1, "ends_dev": 4}
for k, v in json.loads(os.environ.get("CHECK_CFG", "{}")).items(): C[k] = v
TOUCH = lambda: os.utime("/w/activity") if os.path.exists("/w/activity") else None

# ---------------- models (lazy)
_qwen = None
def qwen():
    global _qwen
    if _qwen is None:
        import torch
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        m = Qwen2_5_VLForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype=torch.bfloat16, device_map="cuda")
        p = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
        _qwen = (m, p)
    return _qwen

def ask(content, max_new=300):
    from qwen_vl_utils import process_vision_info
    m, p = qwen()
    msgs = [{"role": "user", "content": content}]
    text = p.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    imgs, vids = process_vision_info(msgs)
    inp = p(text=[text], images=imgs, videos=vids, padding=True, return_tensors="pt").to("cuda")
    out = m.generate(**inp, max_new_tokens=max_new, do_sample=False)
    return p.batch_decode(out[:, inp.input_ids.shape[1]:], skip_special_tokens=True)[0]

def jparse(s):
    m = re.search(r"\{.*\}|\[.*\]", s, re.S)
    try: return json.loads(m.group(0)) if m else None
    except Exception: return None

_box_cache = {}
def char_boxes(img_path):
    """Ask Qwen for the bounding boxes of Ana (ant) and Gus (grasshopper) on the painting."""
    if img_path in _box_cache: return _box_cache[img_path]
    im = cv2.imread(img_path); H, W = im.shape[:2]
    r = ask([{"type": "image", "image": "file://" + img_path},
             {"type": "text", "text": "Locate every cartoon insect character in this picture (an ant with an acorn hat, a green grasshopper with a scarf). "
              "Answer only JSON: [{\"label\":\"ant\"|\"grasshopper\",\"bbox_2d\":[x1,y1,x2,y2]}] in pixel coordinates of this image."}])
    js = jparse(r) or []
    boxes = []
    for b in js if isinstance(js, list) else []:
        bb = b.get("bbox_2d") if isinstance(b, dict) else None
        if bb and len(bb) == 4: boxes.append((b.get("label", "?"), [max(0, bb[0]), max(0, bb[1]), min(W, bb[2]), min(H, bb[3])]))
    _box_cache[img_path] = (boxes, (W, H)); return _box_cache[img_path]

def track(clip_small, boxes_px):
    from ultralytics.models.sam import SAM2VideoPredictor
    pr = SAM2VideoPredictor(overrides=dict(conf=0.25, task="segment", mode="predict", imgsz=640, model="/w/sam2.1_b.pt", verbose=False, save=False))
    masks = []
    for r in pr(source=clip_small, bboxes=boxes_px, stream=True):
        masks.append(r.masks.data.cpu().numpy().astype(np.uint8) if r.masks is not None else None)
    return masks

def skel_ends(m):
    if m.sum() < 30: return 0
    sk = skeletonize(m > 0).astype(np.uint8); nb = cv2.filter2D(sk, -1, np.ones((3, 3), np.float32)) * sk
    return int((nb == 2).sum())

def nparts(m):
    n, _, st, _ = cv2.connectedComponentsWithStats(m, 8); a = sorted([s[4] for s in st[1:]], reverse=True)
    return sum(1 for x in a if a and x > 0.03 * a[0])

JUDGE = ("You are a strict QA checker for AI-animated children's book clips. The first image is the approved painting; the video is the generated clip, "
 "which should show the SAME characters moving gently. Watch every frame carefully and answer only JSON with these keys (true means the problem IS present): "
 "{\"missing_limb\":bool, \"extra_limb\":bool, \"merged_or_melting_limbs\":bool, \"body_morphing\":bool, \"extra_eyes_or_face_distortion\":bool, "
 "\"mouth_talking_or_chewing\":bool, \"color_shift_of_character\":bool, \"extra_or_duplicated_character\":bool, \"camera_moves\":bool, "
 "\"flicker_or_texture_boiling\":bool, \"notes\":\"one short sentence\"}")
JKEYS = ["missing_limb", "extra_limb", "merged_or_melting_limbs", "body_morphing", "extra_eyes_or_face_distortion", "mouth_talking_or_chewing",
         "color_shift_of_character", "extra_or_duplicated_character", "camera_moves", "flicker_or_texture_boiling"]

def run(clip, start, end, use_vlm=True, use_sam=True):
    R = {"clip": clip, "checks": {}, "fail": []}
    def chk(name, ok, value):
        R["checks"][name] = {"pass": bool(ok), "value": value}
        if not ok: R["fail"].append(name)
    if not os.path.exists(clip) or os.path.getsize(clip) < 1000: chk("file", False, "missing/empty"); return R
    cap = cv2.VideoCapture(clip); fps = cap.get(cv2.CAP_PROP_FPS) or 0; fr = []
    while True:
        ok, f = cap.read()
        if not ok: break
        fr.append(f)
    if len(fr) < 8: chk("file", False, f"undecodable ({len(fr)} frames)"); return R
    n = len(fr); dur = n / fps if fps else 0
    chk("fps", C["fps_ok"][0] <= fps <= C["fps_ok"][1], round(fps, 2))
    chk("length", n >= C["min_frames"] and C["min_dur"] <= dur <= C["max_dur"], {"frames": n, "sec": round(dur, 2)})
    g = [cv2.cvtColor(cv2.resize(f, (344, 194)), cv2.COLOR_BGR2GRAY) for f in fr]
    S = cv2.cvtColor(cv2.resize(cv2.imread(start), (344, 194)), cv2.COLOR_BGR2GRAY); E = cv2.cvtColor(cv2.resize(cv2.imread(end), (344, 194)), cv2.COLOR_BGR2GRAY)
    chk("start_match", ssim(S, g[0], data_range=255) >= C["start_ssim"], round(float(ssim(S, g[0], data_range=255)), 3))
    chk("end_match", ssim(E, g[-1], data_range=255) >= C["end_ssim"], round(float(ssim(E, g[-1], data_range=255)), 3))
    luma = np.array([x.mean() for x in g]); chk("black_frames", int((luma < C["black_luma"]).sum()) == 0, int((luma < C["black_luma"]).sum()))
    jl = np.abs(np.diff(luma)); chk("flicker", jl.max() < C["flicker_jump"], round(float(jl.max()), 2))
    tmpd = tempfile.mkdtemp(); small = f"{tmpd}/s.mp4"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", clip, "-vf", "fps=8,scale=640:-2", "-an", small])
    TOUCH()
    if use_sam:
        boxes, (W, H) = char_boxes(start)
        if boxes:
            vh, vw = cv2.VideoCapture(small).read()[1].shape[:2]
            bpx = [[int(b[0] * vw / W), int(b[1] * vh / H), int(b[2] * vw / W), int(b[3] * vh / H)] for _, b in boxes]
            M = track(small, bpx)
            per = []
            for ci in range(len(bpx)):
                ser = [m[ci] if (m is not None and len(m) > ci) else np.zeros((vh, vw), np.uint8) for m in M]
                area = np.array([s.sum() for s in ser], float); a0 = max(area[0], 1)
                ra = area / a0; jump = np.abs(np.diff(area)) / np.maximum(area[:-1], 1)
                parts = np.array([nparts(s) for s in ser]); ends = np.array([skel_ends(s) for s in ser])
                per.append({"who": boxes[ci][0], "area_min": round(float(ra.min()), 3), "area_max": round(float(ra.max()), 3),
                            "jump_max": round(float(jump.max()), 3), "parts_dev": int(np.abs(parts - np.median(parts)).max()),
                            "ends_dev": int(np.abs(ends - np.median(ends)).max())})
            bad_area = [p for p in per if p["area_min"] < 1 - C["area_tol"] or p["area_max"] > 1 + C["area_tol"]]
            chk("sam_area", not bad_area, per)
            chk("sam_jump", all(p["jump_max"] < C["area_jump"] for p in per), [p["jump_max"] for p in per])
            chk("sam_fragments", all(p["parts_dev"] <= C["frag_jump"] for p in per), [p["parts_dev"] for p in per])
            chk("sam_limb_ends", all(p["ends_dev"] <= C["ends_dev"] for p in per), [p["ends_dev"] for p in per])
        else:
            chk("sam_area", True, "no characters located (skipped)")
    TOUCH()
    if use_vlm:
        fdir = f"{tmpd}/f"; os.makedirs(fdir, exist_ok=True)
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", clip, "-vf", "fps=4,scale=640:-2", f"{fdir}/%03d.jpg"])
        frames = sorted(os.listdir(fdir)); frames = frames[: 24] if len(frames) <= 24 else [frames[int(i * (len(frames) - 1) / 23)] for i in range(24)]
        if len(frames) % 2: frames = frames[:-1]
        r = ask([{"type": "image", "image": "file://" + start},
                 {"type": "video", "video": ["file://" + f"{fdir}/{x}" for x in frames], "max_pixels": 360 * 640},
                 {"type": "text", "text": JUDGE}], max_new=250)
        js = jparse(r) or {}
        flagged = [k for k in JKEYS if js.get(k) is True]
        chk("vlm_judge", not flagged, {"flagged": flagged, "notes": js.get("notes", r[:160] if not js else "")})
    TOUCH()
    return R

if __name__ == "__main__":
    jobs = [json.loads(l) for l in open(sys.argv[1])]
    done = set()
    if os.path.exists(sys.argv[2]): done = {json.loads(l)["clip"] for l in open(sys.argv[2])}
    for j in jobs:
        if j["clip"] in done: continue
        try: rep = run(j["clip"], j["start"], j["end"])
        except Exception as e: rep = {"clip": j["clip"], "error": repr(e)[:300], "fail": ["checker_error"], "checks": {}}
        rep.update({k: v for k, v in j.items() if k not in rep})
        open(sys.argv[2], "a").write(json.dumps(rep, default=lambda o: o.item() if hasattr(o, "item") else str(o)) + "\n")
        print(json.dumps({"clip": j["clip"], "fail": rep["fail"]}), flush=True)
