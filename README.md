# Local LLM Serving Bench

Fixed-harness evaluation of local serving options (engine × model × quant) for
**agentic coding + general agent** use on a 128 GB M5 Max, portable to other setups.

## Principles

1. **Compare only within this harness.** Same prompts, budgets, sampling, scorers
   across configs. Never compare against published leaderboard numbers.
2. **Generation is separable from scoring.** Raw generations land in
   `generations.jsonl`; scorers can be upgraded and re-run without re-paying inference.
3. **Everything is recorded.** Each run dir has a `manifest.json` (config, models
   endpoint dump, suite spec, platform, timestamp) so results stay interpretable later.
4. **Crash-safe.** `--run-dir` resumes an interrupted run item-by-item.

## Thinking policy (decided, validated in pilot)

| Mode | Mechanism | Used for | Rationale |
|---|---|---|---|
| think | engine's native default effort (MTPLX: medium), budget = `max_tokens` cap (3072–8192) | MATH, coding, hard reasoning | User guideline: some thinking is necessary at this model scale. Budget is portable across engines; exhaustion rate is tracked. |
| think-light | `reasoning_effort: "minimal"` where supported | optional middle mode | Cuts latency ~2–4×, still helps precision tasks (BUT stochastic: can still burn a low cap — use caps ≥1536). |
| no-think | engine's hard switch (`chat_template_kwargs.enable_thinking=false`) | IF-Lite, tool calling | Measures crisp agent-loop mode. Note: Qwen3.8 **ignores `/no_think`** on MTPLX. |

Pilot evidence (MATH/IF-Lite/tools, this machine): IF-Lite 62.5% hard-off vs
87.5% think-light — instruction precision benefits from thinking; tool calling
is 4/4 either way (and ~2× faster without thinking).

## Config matrix (priority order)

| # | Config | Purpose |
|---|---|---|
| 1 | `mtplx-qwen38-27b-opt-quality` (live baseline) | reference point, Q8 |
| 2 | `mtplx-qwen38-flash-next-opt-speed` | capability ceiling (125B MoE; author's daily driver on identical HW) |
| 3 | `ds4-qwen38-flash-next` | engine A/B on same model: ds4 (no MTP) vs MTPLX (MTP) |
| 4 | `mtplx-qwen38-27b-opt-speed` | quant cost curve: 4-bit vs 8-bit on same weights |
| 5 | `llamacpp-qwen36-35b-a3b` | continuous-batching engine on cheap-fast MoE |
| 6 | `lmstudio-qwen36-35b-a3b` | 3-way engine comparison (with #7) on same model |
| 7 | `mtplx-qwen36-35b-a3b-opt-speed` | 3-way engine comparison, MTPLX leg |
| 8+ | stretch: ds4 + DeepSeek V4/4.1 Flash or GLM 5.3 Flash; Bonsai 2 27B (11.4GB speed/memory extreme); Gemma4 | family diversity if time allows |

Engine pairs share the model wherever possible so engine deltas are clean
(note: ds4 GGUF quant ≠ MTPLX dynamic 4-bit exactly — record both in `weights`).

## Suites

- **pilot** (~10 min): math10-think + iflite8-nothink + tools4-nothink. Validates any new config quickly.
- **tier1** (~2.5–4 h/config at ~40 tok/s): math100-think (budget 3072), iflite, tools.
  TODO add: official IFEval(150)+verifier, HumanEval+/EvalPlus(164) with sandboxed exec,
  BFCL v4 subset, GPQA-D(198) or MMLU-Pro subset, WikiText-2 PPL / KL-vs-fp16 for quant sanity.
- **tier2** (finalists only): Aider Polyglot, RULER 16k/32k, full-thinking MATH-500,
  nanoclaw capstone (~12 real agent tasks, success + wall-clock + tokens — the
  joint intelligence×speed metric).
- **perf** (~15 min quick / ~40 min full): decode TTFT/tok-s/ITL p50+p95, cold-vs-warm
  TTFT at 2k/16k, agent session replay (per-turn TTFT + cached_tokens), concurrency
  sweep 1/2/4/8 (aggregate vs per-stream), server RSS sampling
  (GPU watts: run `sudo powermetrics --samplers gpu_power -i 1000` alongside).

Perf validity notes: filler is seeded word-salad (repetitive text games MTPLX's
SSD n-gram cache); "cold" numbers assume no prior identical prefix.

## Runbook

```bash
cd ~/Documents/bench
.venv/bin/python runner/eval.py --config configs/<cfg>.json --suite pilot      # validate a new server
.venv/bin/python runner/eval.py --config configs/<cfg>.json --suite tier1      # overnight screen
.venv/bin/python runner/perf.py  --config configs/<cfg>.json --quick           # or full (no --quick)
# resume after a crash:
.venv/bin/python runner/eval.py --config configs/<cfg>.json --suite tier1 --run-dir results/<cfg>/<dir>
```

Results: `results/<config>/<suite>-<ts>/{generations.jsonl, metrics.json}` and
`results/<config>/perf-<ts>/perf.json`. One model resident at a time (stop the
previous server before starting the next config — memory fairness).

## Baseline pilot numbers (2026-10-13, MTPLX 27B Q8 quality)

- math10-think: 80% (avg 582 reasoning tok, 1/10 budget exhaustion)
- iflite8: 62.5% no-think / 87.5% think-light · tools4: 100% both modes
- decode 39.5 tok/s (ITL p50 66 ms), cold TTFT@2k 2.57 s → warm 0.16 s (16×, SSD cache)
- replay turns 2+: ~0.4 s TTFT (prefix cache 2048 tok hit)
- concurrency: aggregate flat ~38 tok/s at c1→c4 (serial scheduler) — key axis vs llama.cpp/LM Studio
- server RSS peak ~41 GB

## TODO

- [ ] orchestrator: queue of managed configs with server launch/health/teardown (enables full unattended matrix)
- [ ] official IFEval dataset+verifier; HumanEval+ exec sandbox; BFCL; GPQA/MMLU-Pro; PPL/KL quant check
- [ ] RULER long-context; nanoclaw capstone task pack
- [ ] stats: Wilson CIs on all accuracy numbers in metrics.json
- [ ] report generator: one markdown table across all completed runs
