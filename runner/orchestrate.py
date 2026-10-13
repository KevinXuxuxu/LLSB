#!/usr/bin/env python
"""Unattended orchestrator for the benchmark matrix.

- Queue = configs/*.json with enabled:true, sorted by "priority" (or explicit --configs).
- Per config: (stop all servers) -> launch server -> health check -> warmup ->
  stages [pilot -> perf -> tier1] -> teardown.
- Crash-safe: state in orchestrator_state.json; rerun with --resume (default on)
  to skip finished stages and resume interrupted eval runs item-by-item.
- One model resident at a time: before any managed launch, all known server
  patterns are killed.
- managed: true  = always (re)launch
          "auto" = attach if healthy, else launch (baseline uses this)
          false  = attach only; never kill (external server)

Usage:
  .venv/bin/python runner/orchestrate.py --dry-run          # show the plan
  .venv/bin/python runner/orchestrate.py                    # overnight: all enabled configs
  .venv/bin/python runner/orchestrate.py --configs 'configs/mtplx-qwen38-flash-next*.json' --stages pilot
  .venv/bin/python runner/orchestrate.py --resume           # continue after a crash
"""
import argparse
import datetime
import glob
import json
import os
import re
import signal
import subprocess
import sys
import time

import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "runner"))
import common  # noqa: E402

STATE_PATH = os.path.join(ROOT, "orchestrator_state.json")
LOG_DIR = os.path.join(ROOT, "logs")

STAGES = ["pilot", "perf", "tier1"]           # canonical execution order
STAGE_TIMEOUT_S = {"pilot": 2700, "perf": 7200, "tier1": 8 * 3600}
STAGE_RETRIES = {"pilot": 1, "perf": 1, "tier1": 0}
LOAD_TIMEOUT_S = 1200                          # big models take a while to load

# How to stop each engine when it's resident (patterns via pkill -f; commands via shell).
# Only REAL model servers belong here: launching an engine in this map triggers
# teardown of all others (one-resident rule). Test engines (e.g. mock) must NOT
# appear here, and launching an engine not in this map never tears anything down.
# ds4/lms process names are best-effort; override per-config with "stop_command".
ENGINE_TEARDOWN = {
    "mtplx":    {"patterns": ["mtplx.server"]},
    "llama.cpp": {"patterns": ["llama-server"]},
    "ds4":      {"patterns": ["dwarfstar", "ds4-server"]},
    "lmstudio": {"commands": ["lms server stop || true"], "patterns": ["lms-server"]},
}


def utc():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(msg):
    line = f"[{utc()}] {msg}"
    print(line, flush=True)


# ---------------------------------------------------------------- state

def load_state():
    if os.path.exists(STATE_PATH):
        return json.load(open(STATE_PATH))
    return {"updated": None, "runs": {}}


def save_state(state):
    state["updated"] = utc()
    tmp = STATE_PATH + ".tmp"
    json.dump(state, open(tmp, "w"), indent=2)
    os.replace(tmp, STATE_PATH)


# ---------------------------------------------------------------- server lifecycle

def resolve_api_key(cfg):
    if cfg.get("api_key"):
        return cfg["api_key"]
    env = cfg.get("api_key_env")
    return os.environ.get(env, "") if env else ""


def healthy(cfg, timeout=5):
    try:
        h = {"Content-Type": "application/json"}
        key = resolve_api_key(cfg)
        if key:
            h["Authorization"] = f"Bearer {key}"
        r = requests.get(cfg["base_url"].rstrip("/") + "/models", headers=h, timeout=timeout)
        return r.status_code == 200
    except Exception:
        return False


def teardown_all(reason=""):
    killed = []
    for engine, spec in ENGINE_TEARDOWN.items():
        for cmd in spec.get("commands", []):
            subprocess.run(cmd, shell=True, capture_output=True)
        for pat in spec.get("patterns", []):
            p = subprocess.run(["pkill", "-f", pat], capture_output=True)
            if p.returncode == 0:
                killed.append(pat)
    if killed:
        log(f"teardown_all({reason}): killed {killed}")
        time.sleep(3)


def kill_proc(proc):
    if proc is None or proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        pass
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        proc.wait(timeout=10)


def launch_server(cfg):
    os.makedirs(os.path.join(LOG_DIR, cfg["name"]), exist_ok=True)
    srv_log = os.path.join(LOG_DIR, cfg["name"], "server.log")
    cmd = cfg.get("launch", "")
    if not cmd or "TODO" in cmd:
        return None, "launch command missing/TODO in config"
    with open(srv_log, "a") as f:
        f.write(f"\n==== launch {utc()} ====\n{cmd}\n")
        proc = subprocess.Popen(cmd, shell=True, stdout=f, stderr=subprocess.STDOUT,
                                start_new_session=True)
    log(f"launched server pid={proc.pid} (log: {srv_log})")
    return proc, None


def wait_healthy(cfg, timeout_s=LOAD_TIMEOUT_S, proc=None):
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if proc is not None and proc.poll() is not None:
            log(f"server process died during load (exit={proc.returncode}); see logs/{cfg['name']}/server.log")
            return False
        if healthy(cfg):
            log(f"server healthy after {time.time()-t0:.0f}s")
            return True
        time.sleep(3)
    return False


def warmup(cfg):
    """Cheap calls so first measurements aren't paying load/JIT costs."""
    for max_tokens, tag in ((16, "tiny"), (64, "decode")):
        for attempt in range(3):
            try:
                common.chat(cfg, [{"role": "user", "content": f"Warmup {tag}. Reply briefly."}],
                            max_tokens=max_tokens, extra=cfg.get("nothink_extra"), timeout=120)
                break
            except Exception as e:
                if attempt == 2:
                    log(f"warmup {tag} failed: {e}")
                    return False
                time.sleep(2)
    return True


# ---------------------------------------------------------------- stages

def build_cmd(stage, cfg_path, resume_dir):
    script = os.path.join(ROOT, "runner", "eval.py" if stage != "perf" else "perf.py")
    cmd = [sys.executable, script, "--config", cfg_path]
    if stage == "perf":
        return cmd
    cmd += ["--suite", stage]
    if resume_dir and os.path.isdir(resume_dir):
        cmd += ["--run-dir", resume_dir]
    return cmd


def run_stage(stage, cfg, cfg_path, entry, server_proc):
    os.makedirs(os.path.join(LOG_DIR, cfg["name"]), exist_ok=True)
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = os.path.join(LOG_DIR, cfg["name"], f"{ts}-{stage}.log")
    timeout = STAGE_TIMEOUT_S[stage]
    for attempt in range(STAGE_RETRIES[stage] + 1):
        cmd = build_cmd(stage, cfg_path, entry.get("run_dir"))
        entry.update({"status": "running", "attempt": attempt, "log": log_path,
                      "started": utc()})
        save_state(state)
        shell = f"set -o pipefail; {' '.join(repr(c) for c in cmd)} 2>&1 | tee -a {log_path}"
        log(f"stage {stage} attempt {attempt} -> {log_path}")
        t0 = time.time()
        try:
            p = subprocess.run(["bash", "-c", shell], timeout=timeout)
            ok = p.returncode == 0
            err = f"exit={p.returncode}"
        except subprocess.TimeoutExpired:
            ok, err = False, f"timeout after {timeout}s"
        except Exception as e:
            ok, err = False, str(e)
        # eval prints 'config=... -> <run_dir>'; capture for --run-dir resume
        if stage != "perf" and os.path.exists(log_path):
            m = re.search(r"->\s+(\S+)", open(log_path).read())
            if m and os.path.isdir(m.group(1)):
                entry["run_dir"] = m.group(1)
        entry.update({"status": "done" if ok else "failed", "ended": utc(),
                      "elapsed_s": round(time.time() - t0), "error": None if ok else err})
        save_state(state)
        log(f"stage {stage} {'DONE' if ok else 'FAILED ('+err+')'} in {entry['elapsed_s']}s")
        if ok:
            return True
    return False


# ---------------------------------------------------------------- main

state = load_state()  # module-level so run_stage can save


def plan_configs(args):
    if args.configs:
        paths = []
        for pat in args.configs:
            paths += sorted(glob.glob(pat))
    else:
        paths = sorted(glob.glob(os.path.join(ROOT, "configs", "*.json")))
    cfgs = []
    for p in paths:
        c = json.load(open(p))
        c["_path"] = p
        cfgs.append(c)
    cfgs = [c for c in cfgs if c.get("enabled", False)]
    if not args.configs:  # priority order only for the default queue
        cfgs.sort(key=lambda c: c.get("priority", 99))
    return cfgs


def notify(msg):
    try:
        subprocess.run(["osascript", "-e",
                        f'display notification "{msg}" with title "bench orchestrator"'],
                       capture_output=True, timeout=10)
    except Exception:
        pass


def process(cfg, stages):
    entry = state["runs"].setdefault(cfg["name"], {"config_path": cfg["_path"],
                                                   "stages": {}})

    if not cfg.get("enabled", False):
        entry["status"] = "skipped (disabled)"
        save_state(state)
        log(f"[{cfg['name']}] skipped (enabled=false)")
        return True
    if cfg.get("managed") in (True, "auto") and "TODO" in cfg.get("launch", ""):
        entry["status"] = "blocked (launch is TODO)"
        save_state(state)
        log(f"[{cfg['name']}] BLOCKED: fill 'launch' in {cfg['_path']}")
        return False

    managed = cfg.get("managed", False)
    server_proc = None
    we_launched = False
    try:
        if managed in (True, "auto"):
            if managed == "auto" and healthy(cfg):
                log(f"[{cfg['name']}] attaching to healthy server")
            else:
                if cfg.get("engine") in ENGINE_TEARDOWN:
                    teardown_all(f"before {cfg['name']}")
                else:
                    log(f"[{cfg['name']}] engine not a real model server; skipping teardown of others")
                server_proc, err = launch_server(cfg)
                if err:
                    entry["status"] = f"blocked ({err})"
                    save_state(state)
                    log(f"[{cfg['name']}] BLOCKED: {err}")
                    return False
                we_launched = True
                if not wait_healthy(cfg, proc=server_proc):
                    entry["status"] = "failed (server never became healthy)"
                    save_state(state)
                    log(f"[{cfg['name']}] FAILED: server never healthy; see logs/{cfg['name']}/server.log")
                    return False
        else:
            if not healthy(cfg, timeout=10):
                entry["status"] = "failed (external server not healthy)"
                save_state(state)
                log(f"[{cfg['name']}] FAILED: external server not running at {cfg['base_url']}")
                return False

        if not warmup(cfg):
            entry["status"] = "failed (warmup)"
            save_state(state)
            return False

        all_ok = True
        for stage in stages:
            se = entry["stages"].setdefault(stage, {})
            if args_resume_skip and se.get("status") == "done":
                log(f"[{cfg['name']}] stage {stage}: already done, skipping")
                continue
            ok = run_stage(stage, cfg, cfg["_path"], se, server_proc)
            if not ok and stage == "pilot":
                log(f"[{cfg['name']}] pilot failed; skipping later stages for this config")
                all_ok = False
                break
            all_ok = all_ok and ok
        entry["status"] = "ok" if all_ok else "completed with failures"
        save_state(state)
        return all_ok
    finally:
        if we_launched and server_proc is not None:
            kill_proc(server_proc)
            if cfg.get("engine") in ENGINE_TEARDOWN:
                teardown_all(f"after {cfg['name']}")
            log(f"[{cfg['name']}] server stopped")


args_resume_skip = True


def main():
    global args_resume_skip
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="*", help="explicit config paths/globs (default: all enabled, by priority)")
    ap.add_argument("--stages", default="pilot,perf,tier1", help="comma-separated, executed in this order")
    ap.add_argument("--no-resume", action="store_true", help="re-run stages even if marked done")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    args_resume_skip = not args.no_resume

    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    cfgs = plan_configs(args)

    log(f"queue: {[c['name'] for c in cfgs]}")
    log(f"stages: {stages}  resume={not args.no_resume}")
    if args.dry_run:
        for c in cfgs:
            prev = state["runs"].get(c["name"], {}).get("stages", {})
            done = [s for s in stages if prev.get(s, {}).get("status") == "done"]
            log(f"  would run {c['name']}: stages={stages} (done already: {done or 'none'})")
        return

    results = {}
    interrupted = False
    try:
        for c in cfgs:
            results[c["name"]] = process(c, stages)
    except KeyboardInterrupt:
        interrupted = True
        log("interrupted; state saved; rerun with --resume to continue")
    finally:
        log("summary: " + json.dumps(results))
        notify("queue finished" + (" (interrupted)" if interrupted else "")
               + ("" if all(results.values()) else " - some configs failed"))


if __name__ == "__main__":
    main()
