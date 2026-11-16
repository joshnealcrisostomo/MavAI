"""Run the full 3-layer eval suite + RAGAS in one command.

Chains, in order:
  0. check_keys.py        - fast pre-flight check that required API keys are set (skip with
                             --skip-key-check; a broken key wastes everyone's time an hour into
                             a RAGAS run instead of failing in 1s)
  1. build_dataset.py     - builds forecast_dataset.json (skipped if it already exists, unless
                             --rebuild-dataset is passed - both eval_forecast_vetting.py and
                             eval_ragas.py read from this file)
  2. eval_suite.py         - reliability suite: tool accuracy, guardrails, disambiguation,
                             edge cases, routing, PH history, cross-market (unit/trajectory/e2e
                             checks on the stock-query pipeline)
  3. eval_forecast_vetting.py - unit / trajectory / end-to-end / LLM-judge tiers on the
                             forecast-vetting agent
  4. eval_ragas.py         - RAGAS Faithfulness + Answer Relevance on the same vetting answers,
                             the standard-framework view of step 3's generation checks

Deliberately NOT included by default (opt in with flags - see below):
  --with-curated      eval_curated.py needs a hand-filled curated_events.json first and isn't
                       meaningful to run blind on every invocation
  --with-capability    the capability/ scripts (eval_edt*.py, eval_faang_8k.py, eval_finbert*.py)
                       benchmark the underlying skill against external datasets rather than this
                       pipeline's own tiers - see the eval_capability_* naming discussion
  --with-why-feature   eval_why_feature.py (purpose not yet confirmed - included as a bare call)
  --with-frontier      frontier_benchmark.py - writes results/frontier_template.md (MavAI vs.
                       ChatGPT/Claude/Gemini groundedness); the chatbot-pasting half is manual,
                       this only generates the template, so it's opt-in rather than default
  --with-bakeoff       forecast_bakeoff.py - walk-forward FAANG backtest (RRL evidence for
                       Component #14); slow (fits every backend at every sampled as-of date)
  --with-ablation      run_ablation.py - prompt-variant ablation (12 queries x N prompt variants,
                       with a 10s sleep between each variant for rate limits); slow and loads
                       .env itself
  --with-scan-moves    scan_moves.py - pure price-data candidate scan for curated_events.csv, no
                       AI/agent calls; cheap, but only useful when re-curating events
  --with-blindsided    blindsided.py - "consistent-then-blindsided" showcase scan; calls the
                       vetting agent per candidate event, so cost scales with how many big moves
                       events.auto_events() finds
  --with-extras        shorthand for all six of the above at once

Each step runs even if an earlier one fails (a broken guardrail check shouldn't hide a broken
RAGAS run) - failures are collected and reported in the final summary, and the script's own exit
code is non-zero if anything failed, so it's CI-friendly.

Run from the project root:
    ./.venv/Scripts/python.exe experiments/run_all.py
    ./.venv/Scripts/python.exe experiments/run_all.py --rebuild-dataset
    ./.venv/Scripts/python.exe experiments/run_all.py --ragas-limit 12
    ./.venv/Scripts/python.exe experiments/run_all.py --skip-ragas
    ./.venv/Scripts/python.exe experiments/run_all.py --with-curated --with-capability
    ./.venv/Scripts/python.exe experiments/run_all.py --with-extras   # truly everything
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS = ROOT / "experiments"
CAPABILITY = EXPERIMENTS / "capability"
RESULTS = EXPERIMENTS / "results"
FORECAST_DATA = RESULTS / "forecast_dataset.json"
SUMMARY = RESULTS / "all_evals_summary.txt"




def _run(label, script_path, args=None):
    """Run one eval script as a subprocess, streaming its output live (so a long RAGAS run
    doesn't look hung), and return (label, returncode, elapsed_seconds)."""
    
    
    if not script_path.exists():
        print(f"\n{'=' * 70}\n[SKIP] {label} - {script_path} not found\n{'=' * 70}")
        return label, None, 0.0

    print(f"\n{'=' * 70}\n[RUN]  {label}  ({script_path.relative_to(ROOT)})\n{'=' * 70}")
    cmd = [sys.executable, str(script_path)] + (args or [])
    start = time.time()
    proc = subprocess.run(cmd, cwd=str(ROOT))
    elapsed = time.time() - start
    status = "PASS" if proc.returncode == 0 else f"FAIL (exit {proc.returncode})"
    print(f"[{status}] {label}  ({elapsed:.1f}s)")
    return label, proc.returncode, elapsed


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--skip-key-check", action="store_true", help="skip check_keys.py pre-flight")
    p.add_argument("--rebuild-dataset", action="store_true",
                    help="force-rebuild forecast_dataset.json even if it already exists")
    p.add_argument("--ragas-limit", type=int, default=None,
                    help="cap RAGAS to the first N events (passed through to eval_ragas.py)")
    p.add_argument("--ragas-rebuild", action="store_true",
                    help="regenerate RAGAS's cached vetting answers instead of reusing them")
    p.add_argument("--skip-suite", action="store_true", help="skip eval_suite.py")
    p.add_argument("--skip-vetting", action="store_true", help="skip eval_forecast_vetting.py")
    p.add_argument("--skip-ragas", action="store_true", help="skip eval_ragas.py")
    p.add_argument("--with-curated", action="store_true",
                    help="also run eval_curated.py (needs curated_events.json filled in first)")
    p.add_argument("--with-capability", action="store_true",
                    help="also run the capability/ benchmarks (EDT, FAANG 8-K, FinBERT x2)")
    p.add_argument("--with-why-feature", action="store_true",
                    help="also run eval_why_feature.py (purpose unconfirmed)")
    p.add_argument("--with-frontier", action="store_true",
                    help="also run frontier_benchmark.py (writes frontier_template.md; "
                         "chatbot-pasting step afterward is manual)")
    p.add_argument("--with-bakeoff", action="store_true",
                    help="also run forecast_bakeoff.py (slow walk-forward FAANG backtest)")
    p.add_argument("--with-ablation", action="store_true",
                    help="also run run_ablation.py (slow prompt-variant ablation, sleeps between "
                         "variants for rate limits)")
    p.add_argument("--with-scan-moves", action="store_true",
                    help="also run scan_moves.py (price-only candidate scan, no AI calls)")
    p.add_argument("--with-blindsided", action="store_true",
                    help="also run blindsided.py (consistent-then-blindsided showcase scan)")
    p.add_argument("--with-extras", action="store_true",
                    help="shorthand for all of --with-why-feature/--with-frontier/"
                         "--with-bakeoff/--with-ablation/--with-scan-moves/--with-blindsided")
    args = p.parse_args()

    if args.with_extras:
        args.with_why_feature = args.with_frontier = args.with_bakeoff = True
        args.with_ablation = args.with_scan_moves = args.with_blindsided = True

    results = []

    # step 0: pre-flight key check
    if not args.skip_key_check:
        results.append(_run("check_keys (pre-flight)", EXPERIMENTS / "check_keys.py"))

    # step 1: dataset (prerequisite for both eval_forecast_vetting.py and eval_ragas.py)
    need_vetting_or_ragas = not (args.skip_vetting and args.skip_ragas)
    if need_vetting_or_ragas and (args.rebuild_dataset or not FORECAST_DATA.exists()):
        results.append(_run("build_dataset", EXPERIMENTS / "build_dataset.py"))
    elif need_vetting_or_ragas:
        print(f"\n[SKIP] build_dataset - {FORECAST_DATA.name} already exists "
              f"(use --rebuild-dataset to force)")

    # step 2: reliability suite
    if not args.skip_suite:
        results.append(_run("eval_suite (reliability)", EXPERIMENTS / "eval_suite.py"))

    # step 3: forecast-vetting 3-tier suite
    if not args.skip_vetting:
        results.append(_run("eval_forecast_vetting (unit/trajectory/e2e/judge)",
                             EXPERIMENTS / "eval_forecast_vetting.py"))

    # step 4: RAGAS
    if not args.skip_ragas:
        ragas_args = []
        if args.ragas_rebuild:
            ragas_args.append("--rebuild")
        if args.ragas_limit:
            ragas_args.append(str(args.ragas_limit))
        results.append(_run("eval_ragas (Faithfulness/Answer Relevance)",
                             EXPERIMENTS / "eval_ragas.py", ragas_args))

    # optional extras
    if args.with_curated:
        results.append(_run("eval_curated (hand-labeled benchmark)",
                             EXPERIMENTS / "eval_curated.py"))
    if args.with_capability:
        results.append(_run("eval_edt", CAPABILITY / "eval_edt.py"))
        results.append(_run("eval_edt_events", CAPABILITY / "eval_edt_events.py"))
        results.append(_run("eval_faang_8k", CAPABILITY / "eval_faang_8k.py"))
        results.append(_run("eval_finbert (in-domain PhraseBank)", CAPABILITY / "eval_finbert.py"))
        results.append(_run("eval_finbert_heldout (FiQA)", EXPERIMENTS / "eval_finbert_heldout.py"))
    if args.with_why_feature:
        results.append(_run("eval_why_feature", EXPERIMENTS / "eval_why_feature.py"))
    if args.with_frontier:
        results.append(_run("frontier_benchmark (writes frontier_template.md)",
                             EXPERIMENTS / "frontier_benchmark.py"))
    if args.with_bakeoff:
        results.append(_run("forecast_bakeoff (walk-forward FAANG backtest, RRL #14)",
                             EXPERIMENTS / "forecast_bakeoff.py"))
    if args.with_ablation:
        results.append(_run("run_ablation (prompt-variant ablation)",
                             EXPERIMENTS / "run_ablation.py"))
    if args.with_scan_moves:
        results.append(_run("scan_moves (curated-events candidate scan)",
                             EXPERIMENTS / "scan_moves.py"))
    if args.with_blindsided:
        results.append(_run("blindsided (consistent-then-blindsided showcase)",
                             EXPERIMENTS / "blindsided.py"))

    # summary
    lines = ["=== ALL-EVALS SUMMARY ==="]
    any_fail = False
    for label, code, elapsed in results:
        if code is None:
            lines.append(f"  [SKIP] {label}")
        elif code == 0:
            lines.append(f"  [PASS] {label}  ({elapsed:.1f}s)")
        else:
            lines.append(f"  [FAIL] {label}  exit={code}  ({elapsed:.1f}s)")
            any_fail = True
    report = "\n".join(lines)
    print(f"\n{report}\n")

    RESULTS.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text(report + "\n", encoding="utf-8")
    print(f"wrote {SUMMARY}")
    print("Individual reports: eval_results.txt, forecast_vetting.txt, ragas_eval.txt, "
          "curated_eval.txt (in experiments/results/ unless noted otherwise)")

    sys.exit(1 if any_fail else 0)


if __name__ == "__main__":
    main()