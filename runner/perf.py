"""Performance suite: latency, throughput, cache behavior, concurrency.

Profiles (all against the served stack, OpenAI-compatible):
  decode     - single stream: TTFT, tok/s, p50/p95 inter-token latency
  ttft       - cold vs warm TTFT at 2 prompt sizes (prefill + session cache)
  replay     - multi-turn agent session replay: per-turn TTFT + cached_tokens
  conc       - concurrency sweep: aggregate vs per-stream tok/s

Usage:
  python runner/perf.py --config configs/mtplx-qwen38-27b-opt-quality.json --quick
"""
import argparse
import concurrent.futures
import datetime
import json
import os
import platform
import subprocess
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(__file__))
import common  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Word-salad filler: deliberately NON-repetitive so n-gram/prefix caches cannot
# serve a "cold" prompt. (Repetitive filler games MTPLX's SSD n-gram cache.)
import random
WORDS = ("harbor lantern meridian keel cargo ballast tide rudder sextant anchor buoy mooring "
         "chart ferry mast hull compass dock signal beacon vessel trade silk amber iron copper "
         "wheat timber resin pitch salt hemp flax clerk ledger quill ink parchment stamp seal "
         "gull tern osprey dolphin krill kelp reef shoal strait cape cove fjord islet atoll "
         "lagoon marsh estuary delta spit bar current eddy swell breaker surf foam spray mist").split()


def filler_prompt(approx_tokens, secret):
    rng = random.Random(secret)
    n = max(64, int(approx_tokens * 0.75))
    body = " ".join(rng.choice(WORDS) for n_ in range(n))
    return (f"Logbook excerpt follows. The secret codeword is '{secret}'.\n\n" +
            body +
            "\n\nReply with only the secret codeword.")


def bench_decode(cfg, max_tokens=256):
    res = common.chat(cfg, [{"role": "user", "content":
                      "Write a detailed technical essay about river deltas. Do not think, just write."}],
                      max_tokens=max_tokens)
    itl = sorted(res["itl_s"])
    ctok = res["usage"].get("completion_tokens") or res["n_stream_tokens"]
    return {"ttft_s": res["ttft_s"], "tokens": ctok, "total_s": res["total_s"],
            "tok_s": round(ctok / max(res["total_s"] - (res["ttft_s"] or 0), 1e-3), 1),
            "itl_p50_ms": round(1000 * common.pct(itl, 50), 1) if itl else None,
            "itl_p95_ms": round(1000 * common.pct(itl, 95), 1) if itl else None}


def bench_ttft(cfg, sizes=(2048, 16384)):
    out = {}
    for size in sizes:
        secret = f"delta-{size}-x7q"
        msgs = [{"role": "user", "content": filler_prompt(size, secret)}]
        cold = common.chat(cfg, msgs, max_tokens=6)
        warm = common.chat(cfg, msgs, max_tokens=6)
        def cached(r):
            return ((r["usage"].get("prompt_tokens_details") or {}).get("cached_tokens"))
        out[f"{size}tok"] = {
            "cold_ttft_s": cold["ttft_s"], "cold_cached_tokens": cached(cold),
            "warm_ttft_s": warm["ttft_s"], "warm_cached_tokens": cached(warm),
            "prompt_tokens": cold["usage"].get("prompt_tokens"),
        }
    return out


def bench_replay(cfg, turns=5):
    """Simulates an agent session (pi/nanoclaw style): one big system context,
    growing conversation, short replies each turn. Prefix cache should kick in."""
    msgs = [{"role": "system", "content": filler_prompt(2048, "kappa-9")}]
    per_turn = []
    for t in range(turns):
        msgs.append({"role": "user", "content": f"Turn {t+1}: append one sentence to the logbook about day {t+1}. Reply with just that sentence."})
        r = common.chat(cfg, msgs, max_tokens=64)
        det = (r["usage"].get("prompt_tokens_details") or {})
        per_turn.append({
            "turn": t + 1, "prompt_tokens": r["usage"].get("prompt_tokens"),
            "cached_tokens": det.get("cached_tokens"), "ttft_s": r["ttft_s"], "total_s": r["total_s"],
        })
        msgs.append({"role": "assistant", "content": r["content"][:200]})
    return per_turn


def bench_conc(cfg, levels=(1, 4), max_tokens=192):
    out = {}
    for n in levels:
        def job(i):
            return common.chat(cfg, [{"role": "user", "content":
                              f"Write a detailed essay number {i} about ocean currents. Do not think, just write."}],
                              max_tokens=max_tokens)
        t0 = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=n) as ex:
            rs = list(ex.map(job, range(n)))
        wall = time.perf_counter() - t0
        toks = [r["usage"].get("completion_tokens") or r["n_stream_tokens"] for r in rs]
        out[f"c{n}"] = {"wall_s": round(wall, 2), "tokens_per_stream": toks,
                        "aggregate_tok_s": round(sum(toks) / wall, 1),
                        "slowest_stream_tok_s": round(min(toks) / max(max(r["total_s"] for r in rs), 1e-3), 1)}
    return out


def sample_rss_soon(pid, stop, sink):
    while not stop.is_set():
        try:
            kb = subprocess.check_output(["ps", "-o", "rss=", "-p", str(pid)], text=True).strip()
            sink.append(int(kb))
        except Exception:
            pass
        time.sleep(2.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default=os.path.join(ROOT, "results"))
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    cfg = json.load(open(args.config))
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = os.path.join(args.out, cfg["name"], f"perf-{ts}")
    os.makedirs(run_dir, exist_ok=True)

    # best-effort server RSS sampling (GPU wired memory is not fully visible via RSS;
    # for watts run: sudo powermetrics --samplers gpu_power -i 1000 during the suite)
    server_pid = None
    try:
        server_pid = int(subprocess.check_output(["pgrep", "-f", "mtplx.server"], text=True).strip().splitlines()[0])
    except Exception:
        pass
    rss, stop = [], threading.Event()
    sampler = threading.Thread(target=sample_rss_soon, args=(server_pid, stop, rss), daemon=True) if server_pid else None
    if sampler:
        sampler.start()

    sizes = (2048,) if args.quick else (2048, 16384)
    levels = (1, 4) if args.quick else (1, 2, 4, 8)
    results = {
        "decode": bench_decode(cfg),
        "ttft": bench_ttft(cfg, sizes),
        "replay": bench_replay(cfg, turns=3 if args.quick else 5),
        "conc": bench_conc(cfg, levels),
    }
    stop.set()
    results["server_rss_mb"] = {"peak": max(rss) if rss else None, "samples": len(rss)}
    results["platform"] = {"machine": subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip(),
                           "ram_gb": round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30)}
    results["config"] = {k: cfg[k] for k in ("name", "engine", "model", "weights") if k in cfg}
    results["utc"] = ts
    json.dump(results, open(os.path.join(run_dir, "perf.json"), "w"), indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
