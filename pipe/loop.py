"""Regeneration controller (bible section 7). Runs a plan of clip slots through a model,
the frame checks, and the retry / calm / world-only fallback ladder, under budget caps.
Usage: python3 loop.py --plan plan.json --model "python3 /tmp/dry/model/fakemodel.py" --work DIR [--cfg thresholds.json]
Model contract: <model> --start A --end B --prompt P --seed S --out O.mp4 -> stdout JSON {status, out, cost_usd}; exit!=0 on error."""
import argparse, json, os, subprocess, time, hashlib, shlex, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import checks

SLOT_CAP = 1.20          # first + calm rungs
WORLD_RESERVE = 0.60     # kept back so the safe world-only fallback can always run
NEG = "talking, lip movement, open mouth, chewing, extra legs, missing legs, extra arms, fused limbs, merged legs, melting, morphing, extra antennae, extra eyes, face distortion, color shift, camera shake, zoom, pan, cut, text, watermark, flicker"

class Ledger:
    def __init__(s, path): s.path = path; s.spent = 0.0; s.hashes = {}
    def log(s, **kw):
        kw["t"] = round(time.time(), 1); open(s.path, "a").write(json.dumps(kw) + "\n")

def call_model(model, start, end, prompt, seed, out, ledger, slot, rung, max_tries=3):
    """One generation with transport retries. Returns (path or None, cost, error)."""
    cost = 0.0
    for k in range(max_tries):
        cmd = shlex.split(model) + ["--start", start, "--end", end, "--prompt", prompt + " | negative: " + NEG, "--seed", str(seed), "--out", out]
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        except subprocess.TimeoutExpired:
            ledger.log(slot=slot, rung=rung, seed=seed, event="transport_timeout", attempt=k); continue
        info = {}
        for line in (p.stdout.strip().splitlines() or [""])[::-1]:
            try: info = json.loads(line); break
            except Exception: pass
        c = float(info.get("cost_usd", 0) or 0)
        if not c and p.returncode != 0:
            try: c = float(json.loads(p.stderr.strip().splitlines()[-1]).get("cost_usd", 0))
            except Exception: c = 0
        cost += c; ledger.spent += c
        if p.returncode == 0 and info.get("status") == "ok" and os.path.exists(out):
            return out, cost, None
        err = (p.stderr.strip().splitlines() or ["?"])[-1][:200]
        ledger.log(slot=slot, rung=rung, seed=seed, event="model_error", attempt=k, err=err, cost=c)
        if "rate" in err.lower(): time.sleep(2 * (k + 1))
    return None, cost, "model_failed"

def run_slot(slot, model, work, cfg, ledger, global_cap):
    sid = slot["id"]; os.makedirs(f"{work}/{sid}", exist_ok=True)
    legs = slot.get("legs_move", False)
    ladder = [
        ("first", slot["prompt"], slot["end"], [11, 22, 33][: 3 if legs else 2]),
        ("calm", slot.get("calm_prompt") or ("slower and smaller: " + slot["prompt"]), slot["end"], [44, 55]),
        ("world_only", slot.get("world_prompt") or "character stays still, breathing gently; " + slot.get("world", "grass sways softly"), slot["start"], [66, 77]),
    ]
    spent = 0.0
    for rung, prompt, end, seeds in ladder:
        passing = []
        for sd in seeds:
            cap = SLOT_CAP + (WORLD_RESERVE if rung == 'world_only' else 0)
            if spent + 0.30 > cap + 1e-9 or ledger.spent + 0.30 > global_cap + 1e-9:
                ledger.log(slot=sid, rung=rung, event="budget_stop", slot_spent=spent, total=ledger.spent)
                break
            out = f"{work}/{sid}/{rung}_{sd}.mp4"
            path, c, err = call_model(model, slot["start"], end, prompt, sd, out, ledger, sid, rung)
            spent += c
            if not path: continue
            h = hashlib.sha1(open(path, "rb").read()).hexdigest()
            if h in ledger.hashes:
                ledger.log(slot=sid, rung=rung, seed=sd, event="duplicate_output", same_as=ledger.hashes[h], cost=c); continue
            ledger.hashes[h] = out
            rep = checks.run(path, slot["start"], end, cfg)
            ok = not rep["fail"]
            ledger.log(slot=sid, rung=rung, seed=sd, event="checked", out=out, ok=ok, fail=rep["fail"], worst=rep.get("worst"),
                       values={k: v["value"] for k, v in rep["checks"].items()}, score=rep.get("score"), cost=c)
            if ok: passing.append((rep["score"], out))
        if passing:
            passing.sort(); best = passing[0][1]
            return {"slot": sid, "result": "clip", "rung": rung, "out": best, "spent": round(spent, 2), "candidates": len(passing)}
    return {"slot": sid, "result": "hold_still", "reason": "all_rungs_failed", "spent": round(spent, 2)}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--plan"); ap.add_argument("--model"); ap.add_argument("--work"); ap.add_argument("--cfg"); ap.add_argument("--cap", type=float, default=18.0)
    a = ap.parse_args(); os.makedirs(a.work, exist_ok=True)
    cfg = json.load(open(a.cfg)) if a.cfg and os.path.exists(a.cfg) else {}
    plan = json.load(open(a.plan)); L = Ledger(f"{a.work}/ledger.jsonl")
    res = []
    for s in plan:
        r = run_slot(s, a.model, a.work, cfg, L, a.cap); res.append(r); print(json.dumps(r), flush=True)
    json.dump({"results": res, "spent": round(L.spent, 2)}, open(f"{a.work}/results.json", "w"), indent=1)
    print("SPENT", round(L.spent, 2))
