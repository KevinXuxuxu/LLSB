"""OpenAI-compatible chat client with per-token timing.

Stdlib + requests only, so the harness runs anywhere (macOS/Linux, any engine
that speaks the OpenAI chat API). All timing is wall-clock from request start.
"""
import json
import time

import requests


def _headers(cfg):
    h = {"Content-Type": "application/json"}
    if cfg.get("api_key"):
        h["Authorization"] = f"Bearer {cfg['api_key']}"
    return h


def _url(cfg, path):
    return cfg["base_url"].rstrip("/") + path


def list_models(cfg, timeout=10):
    r = requests.get(_url(cfg, "/models"), headers=_headers(cfg), timeout=timeout)
    r.raise_for_status()
    return r.json()


def chat(cfg, messages, max_tokens=1024, temperature=None, top_p=None,
         tools=None, stream=True, timeout=1800, extra=None, seed=None):
    """One chat completion. Returns a dict with content, reasoning_content,
    tool_calls, usage, ttft_s, total_s, itl_s (per-token gaps), finish_reason,
    and mtplx_stats keys if the server exposes them."""
    sampling = cfg.get("sampling", {})
    body = {
        "model": cfg["model"],
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": sampling.get("temperature", 0.6) if temperature is None else temperature,
        "top_p": sampling.get("top_p", 0.95) if top_p is None else top_p,
        "stream": stream,
    }
    if seed is not None:
        body["seed"] = seed
    if tools:
        body["tools"] = tools
    if extra:
        body.update(extra)

    t0 = time.perf_counter()
    r = requests.post(_url(cfg, "/chat/completions"), headers=_headers(cfg),
                      json=body, stream=stream, timeout=timeout)
    r.raise_for_status()

    if not stream:
        t1 = time.perf_counter()
        d = r.json()
        msg = d["choices"][0]["message"]
        return {
            "content": msg.get("content") or "",
            "reasoning_content": msg.get("reasoning_content") or "",
            "tool_calls": [
                {"name": tc["function"]["name"], "arguments": tc["function"].get("arguments", "")}
                for tc in (msg.get("tool_calls") or [])
            ],
            "finish_reason": d["choices"][0].get("finish_reason"),
            "usage": d.get("usage", {}),
            "mtplx_keys": sorted((d.get("mtplx_stats") or {}).keys()),
            "ttft_s": round(t1 - t0, 4), "total_s": round(t1 - t0, 4),
            "itl_s": [], "n_stream_tokens": 0,
        }

    content, reasoning, stamps = [], [], []
    tool_parts = {}          # index -> {"name":..., "args": str}
    ttft = None
    usage, finish, mtplx_keys = {}, None, []
    for raw in r.iter_lines():
        if not raw:
            continue
        line = raw.decode("utf-8", "replace")
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            break
        try:
            d = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if d.get("usage"):
            usage = d["usage"]
        if d.get("mtplx_stats"):
            mtplx_keys = sorted(d["mtplx_stats"].keys())
        ch = (d.get("choices") or [{}])[0]
        if ch.get("finish_reason"):
            finish = ch["finish_reason"]
        delta = ch.get("delta") or {}
        piece_c, piece_r = delta.get("content"), delta.get("reasoning_content")
        if piece_c or piece_r:
            now = time.perf_counter()
            if ttft is None:
                ttft = now - t0
            stamps.append(now)
            if piece_c:
                content.append(piece_c)
            if piece_r:
                reasoning.append(piece_r)
        for tc in delta.get("tool_calls") or []:
            slot = tool_parts.setdefault(tc.get("index", 0), {"name": "", "args": ""})
            fn = tc.get("function") or {}
            if fn.get("name"):
                slot["name"] += fn["name"]
            if fn.get("arguments"):
                slot["args"] += fn["arguments"]

    t_end = stamps[-1] if stamps else time.perf_counter()
    itls = [round(b - a, 4) for a, b in zip(stamps, stamps[1:])]
    return {
        "content": "".join(content),
        "reasoning_content": "".join(reasoning),
        "tool_calls": [{"name": v["name"], "arguments": v["args"]}
                       for _, v in sorted(tool_parts.items())],
        "finish_reason": finish,
        "usage": usage,
        "mtplx_keys": mtplx_keys,
        "ttft_s": round(ttft, 4) if ttft is not None else None,
        "total_s": round(t_end - t0, 4),
        "itl_s": itls,
        "n_stream_tokens": len(stamps),
    }


def pct(sorted_vals, p):
    if not sorted_vals:
        return None
    k = min(int(round(p / 100.0 * (len(sorted_vals) - 1))), len(sorted_vals) - 1)
    return sorted_vals[k]
