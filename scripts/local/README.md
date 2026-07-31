# `scripts/local/` — local-only, never-committed scripts

**Everything in this directory is gitignored by default.** Only this `README.md` is
tracked. This is a *default-deny* zone: any script you drop here is automatically
excluded from the (public) repository, so a one-off fix can never leak into git
history by accident. It replaces the old, brittle "add each script to `.gitignore`
by name" approach.

## What goes here
- Ad-hoc fix-up scripts and one-off data migrations.
- Anything that reads or writes **real personal data** — holdings, account numbers,
  balances, dividend/fee figures, feedback/interaction history, portfolio CSVs.
- Maintainer-only tooling not needed by end users or contributors: DSPy prompt
  optimization (`optimize_prompts.py`, `run_optimization.sh`,
  `generate_training_examples.py`), metric/portfolio validation harnesses
  (`validate_metrics.py`, `test_historical_baseline.py`), RAG/log diagnostics
  (`analyze_tool_retrieval.py`, `monitor_health_check.py`), dependency hygiene
  (`scan_imports.py`), and deploy scripts that pull infra secrets from
  `user_data/.env` (`deploy_landing.sh`).
- Keep the DSPy set together here — `run_optimization.sh` and
  `generate_training_examples.py` call/reference `optimize_prompts.py`.

## What does NOT go here
Shared, contributor- or end-user-facing tooling belongs in `scripts/` proper
(`install/`, `package/`, `docker/`, the launchers). Those are tracked and shipped.

## The rule: never commit personal data — anywhere
This is a **public** repository, and git history is forever.
- Never `git add -f` anything under `scripts/local/`.
- Never hardcode account numbers, balances, holdings, API keys, infra hostnames, or
  personal identifiers in *any* file — not even a local script. The deleted
  `graph_cleanup.py` leaked a real brokerage account number this way, which then
  required a full git-history rewrite and force-push to scrub.
- If personal data or a secret ever lands in git, deleting the file is **not**
  enough — it must be erased from history (`git filter-repo`) and force-pushed; on a
  public repo, also have GitHub purge cached commits.

Put personal data only under the gitignored `user_data/`, or pass it via env/args.
