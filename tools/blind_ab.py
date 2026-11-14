#!/usr/bin/env python3
"""Blind side-by-side comparison of two runs' videos, judged by a person in a browser (local, no GPU).

make:  copies each pair's mp4s under random names into OUT/v/, writes OUT/index.html (shows one
       pair at a time, left/right randomized, run names hidden) and OUT/key.json (the mapping; the
       page never loads it). Answers are kept in the browser's localStorage, and the page shows them
       as JSON at the end to copy into OUT/answers.json.
score: decodes answers.json with key.json and prints, per comparison, how often the base run was
       preferred.

usage:
  python tools/blind_ab.py make --base results/q_steps50 --tests results/q_steps20 results/q_fbc020 --out results/blind_ab
  python tools/blind_ab.py score --out results/blind_ab
"""
import argparse
import json
import os
import random
import secrets
import shutil
from collections import Counter

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Blind video comparison</title>
<style>
 body{font-family:system-ui,sans-serif;margin:16px;background:#111;color:#eee}
 .pair{display:flex;gap:12px;flex-wrap:wrap}.pair div{flex:1 1 400px}
 video{width:100%;background:#000}.lab{text-align:center;font-size:20px;margin:4px}
 button{font-size:18px;padding:10px 18px;margin:8px 6px;cursor:pointer}
 #bar{margin:8px 0;color:#aaa}textarea{width:100%;height:240px}
</style></head><body>
<div id="bar"></div>
<div id="main"></div>
<script>
const ITEMS = __ITEMS__;
const STORE = "blind_ab_" + __TAG__;
let ans = {};
try { ans = JSON.parse(localStorage.getItem(STORE) || "{}"); } catch (e) {}
function save(){ try { localStorage.setItem(STORE, JSON.stringify(ans)); } catch (e) {} }
function next(){ return ITEMS.findIndex(it => !(it.id in ans)); }
function show(){
  const i = next(), done = Object.keys(ans).length;
  document.getElementById("bar").textContent = `${done} of ${ITEMS.length} judged. Keys: 1 = left better, 2 = same, 3 = right better, u = undo.`;
  const m = document.getElementById("main");
  if (i < 0) {
    m.innerHTML = "<p>Done. Copy this into answers.json:</p><textarea readonly></textarea>";
    m.querySelector("textarea").value = JSON.stringify(ans, null, 1);
    return;
  }
  const it = ITEMS[i];
  m.innerHTML = `<p>${it.prompt}</p><div class="pair">
   <div><div class="lab">Left</div><video src="v/${it.left}" autoplay loop muted playsinline></video></div>
   <div><div class="lab">Right</div><video src="v/${it.right}" autoplay loop muted playsinline></video></div></div>
   <button data-a="left">Left better (1)</button><button data-a="same">Same (2)</button><button data-a="right">Right better (3)</button>`;
  const vs = m.querySelectorAll("video");   // keep the two videos in step
  vs[0].addEventListener("play", () => { vs[1].currentTime = vs[0].currentTime; });
  m.querySelectorAll("button").forEach(b => b.onclick = () => pick(it.id, b.dataset.a));
}
function pick(id, a){ ans[id] = a; save(); show(); }
document.addEventListener("keydown", e => {
  const i = next();
  if (e.key === "u") { const ks = Object.keys(ans); if (ks.length) { delete ans[ks[ks.length - 1]]; save(); show(); } return; }
  if (i < 0) return;
  const a = {"1": "left", "2": "same", "3": "right"}[e.key];
  if (a) pick(ITEMS[i].id, a);
});
show();
</script></body></html>
"""


def load_runs(run):
    with open(os.path.join(run, "report.json")) as f:
        return {str(r.get("key", r.get("prompt_id", i))): r["prompt"] for i, r in enumerate(json.load(f)["runs"])}


def make(args):
    rng = random.Random(args.seed)
    vdir = os.path.join(args.out, "v")
    os.makedirs(vdir, exist_ok=True)
    base = load_runs(args.base)
    items, key = [], {}
    for test in args.tests:
        t = load_runs(test)
        for k in sorted(set(base) & set(t)):
            names = {}
            for run in (args.base, test):
                n = secrets.token_hex(6) + ".mp4"
                shutil.copy(os.path.join(run, f"video_{k}.mp4"), os.path.join(vdir, n))
                names[run] = n
            base_left = rng.random() < 0.5
            iid = secrets.token_hex(4)
            left, right = (args.base, test) if base_left else (test, args.base)
            items.append(dict(id=iid, prompt=base[k], left=names[left], right=names[right]))
            key[iid] = dict(base=args.base, test=test, video=k, base_side="left" if base_left else "right")
    rng.shuffle(items)
    tag = json.dumps(secrets.token_hex(3))
    with open(os.path.join(args.out, "index.html"), "w", encoding="utf-8") as f:
        f.write(PAGE.replace("__ITEMS__", json.dumps(items)).replace("__TAG__", tag))
    with open(os.path.join(args.out, "key.json"), "w") as f:
        json.dump(key, f, indent=1)
    print(f"{len(items)} pairs -> {os.path.join(args.out, 'index.html')}")


def score(args):
    key = json.load(open(os.path.join(args.out, "key.json")))
    ans = json.load(open(os.path.join(args.out, "answers.json")))
    by = {}
    for iid, a in ans.items():
        k = key[iid]
        res = "same" if a == "same" else ("base" if a == k["base_side"] else "test")
        by.setdefault(k["test"], Counter())[res] += 1
    out = {}
    for test, c in by.items():
        n = sum(c.values())
        out[test] = dict(n=n, base=c["base"], same=c["same"], test=c["test"],
                         base_preferred=(c["base"] + 0.5 * c["same"]) / n)
        print(f"{os.path.basename(args_base(key, test))} vs {os.path.basename(test)}: base better {c['base']}, "
              f"same {c['same']}, test better {c['test']} of {n} -> base preferred {out[test]['base_preferred']:.0%} "
              f"(ties count half)")
    with open(os.path.join(args.out, "scores.json"), "w") as f:
        json.dump(out, f, indent=1)


def args_base(key, test):
    return next(v["base"] for v in key.values() if v["test"] == test)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make")
    m.add_argument("--base", required=True)
    m.add_argument("--tests", nargs="+", required=True)
    m.add_argument("--out", required=True)
    m.add_argument("--seed", type=int, default=None, help="for the left/right and order shuffle; default random")
    s = sub.add_parser("score")
    s.add_argument("--out", required=True)
    args = ap.parse_args()
    make(args) if args.cmd == "make" else score(args)


if __name__ == "__main__":
    main()
