#!/usr/bin/env python
"""
Run the golden-set eval harness (Advisor Roadmap Theme 2.4).

The manual pre-change gate: run this before any prompt, model, or provider
change to confirm the risk layer still catches its dominant failure modes.

    python scripts/run_eval_harness.py          # deterministic (offline, free)
    python scripts/run_eval_harness.py --live    # + LLM judge (real model calls)

Deterministic mode runs the pure-Python grounding + IPS pre-checks — offline,
free, safe for CI. --live runs every scenario through the ACTUAL Risk Judge
(one model call each, ~12 total) via the pure judge_advice seam, which does NOT
persist verdicts or send status — so a harness run never pollutes the per-profile
risk_verdicts.jsonl audit trail. Run it before any provider/model/prompt change.

Exit code is non-zero if any scenario FAILs, so it can wire into a pre-change
checklist or CI step.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.eval.golden_harness import format_report, run_all, run_all_live  # noqa: E402


def main() -> int:
    live = "--live" in sys.argv[1:]
    if live:
        print("Running LIVE judge mode — real model calls, one per scenario.\n")
        results = run_all_live()
        print(format_report(results, mode="live-judge"))
    else:
        results = run_all()
        print(format_report(results))
    return 1 if any(r.status in ("FAIL", "ERROR") for r in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
