"""Capability evaluation runner.

Principles baked in:
- Fixed prompts, fixed budgets, fixed scorers -> only the deployment config varies.
- Raw generations are always saved (generations.jsonl); scoring is separable.
- Thinking policy is explicit per benchmark: budget via max_tokens (portable),
  no-think via a Qwen3 '/no_think' suffix where the engine lacks a switch.

Usage:
  python runner/eval.py --config configs/mtplx-qwen38-27b-opt-quality.json --suite pilot
"""
import argparse
import datetime
import json
import os
import platform
import re
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
import common  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "data", "cache")

# ---------------------------------------------------------------- datasets

def load_math(n):
    path = os.path.join(CACHE, "math500.jsonl")
    if not os.path.exists(path):
        url = "https://huggingface.co/datasets/HuggingFaceH4/MATH-500/resolve/main/test.jsonl"
        import urllib.request
        urllib.request.urlretrieve(url, path)
    rows = [json.loads(l) for l in open(path)]
    return rows[:n]


# ---------------------------------------------------------------- scorers

def _norm_math(s):
    if s is None:
        return None
    s = s.strip().strip("$").replace("\\!", "").replace("\\,", "").replace("\\ ", "").replace(" ", "")
    s = s.replace("\\left", "").replace("\\right", "").replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    s = s.replace("\\text{", "{").rstrip(".").lower()
    return s


def extract_boxed(text):
    idx = text.rfind("\\boxed")
    if idx < 0:
        return None
    i = text.find("{", idx)
    if i < 0:
        return None
    depth = 0
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1:j]
    return None


def score_math(record, gold_answer):
    pred = extract_boxed(record["content"])
    ok = pred is not None and _norm_math(pred) == _norm_math(gold_answer)
    exhausted = record["finish_reason"] == "length" or pred is None
    return {"score": int(ok), "extracted": pred, "budget_exhausted": int(exhausted)}


# IF-Lite: hand-rolled verifiable instruction following (deterministic, portable).
# TODO(tier1): swap in official google/IFEval dataset + its verifier for the real thing.
IF_LITE = [
    ("Write exactly 3 bullet points about apples. Start each line with '- '. Nothing else.",
     lambda t: len([l for l in t.splitlines() if l.strip().startswith("- ")]) == 3),
    ("Write a paragraph about the sea that is exactly 40 words. Output nothing else.",
     lambda t: len(t.split()) == 40),
    ("Reply with a valid JSON object having exactly the keys 'name' and 'age'. No other text, no markdown.",
     lambda t: _json_has_keys(t, {"name", "age"})),
    ("Write between 30 and 50 words about photosynthesis, entirely in lowercase letters. No capitals anywhere.",
     lambda t: t == t.lower() and 30 <= len(t.split()) <= 50),
    ("Mention the word 'quokka' at least twice and end your reply with the exact phrase 'End of Report'.",
     lambda t: t.lower().count("quokka") >= 2 and t.strip().endswith("End of Report")),
    ("Write exactly 5 sentences about Rome. No bullet points, no lists, no headings.",
     lambda t: len(re.findall(r"[.!?](\s|$)", t.strip())) == 5),
    ("Give me a comma-separated list of exactly 6 colors, all in lowercase. Nothing else, no newline.",
     lambda t: "\n" not in t.strip() and len([c for c in t.strip().split(",") if c.strip()]) == 6
               and t.strip() == t.strip().lower()),
    ("Your first line must be 'START' and your last line must be 'END'. Between them write one sentence about trains.",
     lambda t: t.strip().splitlines()[0].strip() == "START" and t.strip().splitlines()[-1].strip() == "END"
               and len(t.strip().splitlines()) >= 3),
]


def _json_has_keys(t, keys):
    t = t.strip()
    t = re.sub(r"^```(json)?|```$", "", t, flags=re.MULTILINE).strip()
    try:
        return keys.issubset(set(json.loads(t).keys()))
    except Exception:
        return False


# Tool-calling micro-suite: scores the *stack* (template + parser), not just the model.
WEATHER = {"type": "function", "function": {
    "name": "get_current_weather", "description": "Get current weather for a city",
    "parameters": {"type": "object",
                   "properties": {"location": {"type": "string"},
                                  "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}},
                   "required": ["location"]}}}
CALC = {"type": "function", "function": {
    "name": "calculator", "description": "Evaluate an arithmetic expression",
    "parameters": {"type": "object", "properties": {"expression": {"type": "string"}},
                   "required": ["expression"]}}}


def _args(raw):
    try:
        return json.loads(raw or "{}")
    except Exception:
        return {}


TOOL_CASES = [
    {"id": "weather-tokyo", "tools": [WEATHER], "prompt": "What's the current weather in Tokyo? Give it in celsius.",
     "check": lambda r: r["tool_calls"] and r["tool_calls"][0]["name"] == "get_current_weather"
                        and _args(r["tool_calls"][0]["arguments"]).get("location", "").lower().find("tokyo") >= 0},
    {"id": "calc-17x23", "tools": [CALC], "prompt": "What is 17*23? Use the calculator.",
     "check": lambda r: r["tool_calls"] and r["tool_calls"][0]["name"] == "calculator"
                        and "17" in _args(r["tool_calls"][0]["arguments"]).get("expression", "")
                        and "23" in _args(r["tool_calls"][0]["arguments"]).get("expression", "")},
    {"id": "pick-right-tool", "tools": [WEATHER, CALC], "prompt": "Compute 144/12 for me.",
     "check": lambda r: r["tool_calls"] and r["tool_calls"][0]["name"] == "calculator"},
    {"id": "no-tool", "tools": [WEATHER], "prompt": "Do not call any tools. Just answer: what is 2+2?",
     "check": lambda r: not r["tool_calls"] and "4" in (r["content"] or "")},
]


# ---------------------------------------------------------------- suites

SUITES = {
    "pilot": [
        {"bench": "math10-think", "kind": "math", "n": 10, "max_tokens": 3072, "think": True},
        {"bench": "iflite8-nothink", "kind": "iflite", "n": 8, "max_tokens": 768, "think": False},
        {"bench": "tools4-nothink", "kind": "tools", "n": 4, "max_tokens": 512, "think": False},
    ],
    "tier1": [
        # Screen suite (~2.5-4h/config at ~35 tok/s with thinking). Sizes chosen
        # so a 3-point accuracy delta is roughly at the noise floor (Wilson CI).
        {"bench": "math100-think", "kind": "math", "n": 100, "max_tokens": 3072, "think": True},
        {"bench": "iflite8-nothink", "kind": "iflite", "n": 8, "max_tokens": 768, "think": False},
        {"bench": "tools4-nothink", "kind": "tools", "n": 4, "max_tokens": 512, "think": False},
        # TODO(tier1): official IFEval(150) + verifier, HumanEval+/EvalPlus(164)
        # with sandboxed exec, BFCL v4 subset, GPQA-D or MMLU-Pro subset.
    ],
}


def run_item(cfg, item, idx):
    """Dispatch one benchmark item; return (record, score_dict)."""
    kind = item["kind"]
    nothink = not item["think"]
    suffix = " " + cfg["nothink_suffix"] if (nothink and cfg.get("nothink_suffix")) else ""
    extra = cfg.get("nothink_extra") if nothink else None  # engine-specific body override

    if kind == "math":
        row = item["rows"][idx]
        prompt = row["problem"] + "\n\nSolve the problem. End with the final answer as \\boxed{...}."
        res = common.chat(cfg, [{"role": "user", "content": prompt}],
                          max_tokens=item["max_tokens"], seed=item.get("seed"), extra=extra)
        rec = {"prompt": prompt, "gold": row["answer"], "res": res}
        return rec, score_math(res, row["answer"])

    if kind == "iflite":
        prompt, check = item["rows"][idx]
        res = common.chat(cfg, [{"role": "user", "content": prompt + suffix}],
                          max_tokens=item["max_tokens"], seed=item.get("seed"), extra=extra)
        return ({"prompt": prompt, "res": res},
                {"score": int(bool(check(res["content"] or ""))), "extracted": None, "budget_exhausted": int(res["finish_reason"] == "length")})

    if kind == "tools":
        case = item["rows"][idx]
        res = common.chat(cfg, [{"role": "user", "content": case["prompt"] + suffix}],
                          max_tokens=item["max_tokens"], tools=case["tools"], stream=False,
                          seed=item.get("seed"), extra=extra)
        return ({"prompt": case["prompt"], "res": res},
                {"score": int(bool(case["check"](res))), "extracted": None, "budget_exhausted": 0})

    raise ValueError(kind)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--suite", default="pilot")
    ap.add_argument("--out", default=os.path.join(ROOT, "results"))
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--run-dir", default=None,
                    help="resume an interrupted run: append to its generations.jsonl, skip done items")
    args = ap.parse_args()

    cfg = json.load(open(args.config))
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if args.run_dir:
        run_dir = args.run_dir
    else:
        run_dir = os.path.join(args.out, cfg["name"], f"{args.suite}-{ts}")
    os.makedirs(run_dir, exist_ok=True)
    gen_path = os.path.join(run_dir, "generations.jsonl")

    done = set()
    stored = {}
    if os.path.exists(gen_path):
        for line in open(gen_path):
            try:
                r = json.loads(line)
                done.add((r["bench"], r["i"]))
                stored[(r["bench"], r["i"])] = r
            except json.JSONDecodeError:
                pass  # truncated tail line from a crash
        mode = "a"
    else:
        mode = "w"
    if done:
        print(f"resuming: {len(done)} items already done")

    specs = []
    for spec in SUITES[args.suite]:
        spec = dict(spec)
        if spec["kind"] == "math":
            spec["rows"] = load_math(spec["n"])
        elif spec["kind"] == "iflite":
            spec["rows"] = IF_LITE[:spec["n"]]
        elif spec["kind"] == "tools":
            spec["rows"] = TOOL_CASES[:spec["n"]]
        spec["seed"] = args.seed
        specs.append(spec)

    print(f"config={cfg['name']} suite={args.suite} -> {run_dir}")
    metrics = {}
    with open(gen_path, mode) as out:
        for spec in specs:
            scores, rows_meta = [], []
            for i in range(spec["n"]):
                if (spec["bench"], i) in done:
                    r = stored[(spec["bench"], i)]
                    scores.append(r["score"])
                    rows_meta.append({"ttft": r.get("ttft_s"), "wall": r.get("total_s"),
                                      "ctok": (r.get("usage") or {}).get("completion_tokens"),
                                      "rtok": ((r.get("usage") or {}).get("completion_tokens_details") or {}).get("reasoning_tokens")})
                    print(f"  [{spec['bench']}] {i+1}/{spec['n']} skip (done)")
                    continue
                for attempt in range(3):
                    try:
                        rec, sc = run_item(cfg, spec, i)
                        break
                    except Exception as e:
                        if attempt == 2:
                            raise
                        print(f"  retry {i} after error: {e}", file=sys.stderr)
                        time.sleep(3)
                res = rec["res"]
                rec_out = {
                    "bench": spec["bench"], "i": i, "prompt": rec["prompt"],
                    "gold": rec.get("gold"),
                    "content": res["content"], "reasoning_content": res["reasoning_content"],
                    "tool_calls": res["tool_calls"], "finish_reason": res["finish_reason"],
                    "usage": res["usage"], "ttft_s": res["ttft_s"], "total_s": res["total_s"],
                    "score": sc,
                }
                out.write(json.dumps(rec_out) + "\n")
                out.flush()
                scores.append(sc)
                rows_meta.append({"ttft": res["ttft_s"], "wall": res["total_s"],
                                  "ctok": res["usage"].get("completion_tokens"),
                                  "rtok": (res["usage"].get("completion_tokens_details") or {}).get("reasoning_tokens")})
                acc = sum(s["score"] for s in scores) / len(scores)
                print(f"  [{spec['bench']}] {i+1}/{spec['n']} score={sc['score']} run_acc={acc:.2f} "
                      f"ctok={rec_out['usage'].get('completion_tokens')} wall={res['total_s']}s")
            n = len(scores)
            metrics[spec["bench"]] = {
                "n": n,
                "accuracy": sum(s["score"] for s in scores) / n,
                "budget_exhaustion_rate": sum(s["budget_exhausted"] for s in scores) / n,
                "avg_completion_tokens": _mean([m["ctok"] for m in rows_meta if m["ctok"] is not None]),
                "avg_reasoning_tokens": _mean([m["rtok"] for m in rows_meta if m["rtok"] is not None]),
                "avg_wall_s": _mean([m["wall"] for m in rows_meta]),
                "avg_ttft_s": _mean([m["ttft"] for m in rows_meta if m["ttft"] is not None]),
            }

    manifest = {
        "utc": ts,
        "config": cfg,
        "suite": args.suite,
        "specs": [{k: v for k, v in s.items() if k != "rows"} for s in specs],
        "models_endpoint": common.list_models(cfg),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "note": "Compare numbers only within this harness, never vs published leaderboards.",
    }
    json.dump({"manifest": manifest, "metrics": metrics},
              open(os.path.join(run_dir, "metrics.json"), "w"), indent=2)
    print(json.dumps(metrics, indent=2))


def _mean(xs):
    return round(sum(xs) / len(xs), 2) if xs else None


if __name__ == "__main__":
    main()
