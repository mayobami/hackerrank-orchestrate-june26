"""Production entry point (AGENTS.md section 6.1 designated entry point).

Reads dataset/claims.csv (the unlabeled, real submission set) and produces
code/output.csv with structured predictions, using Strategy A (the
monolithic single-call-per-claim strategy chosen after comparison against
Strategy B on dataset/sample_claims.csv -- see evaluation/evaluation_report.md).

This is a thin wrapper around pipeline/run_full_eval.py, which does the
actual work: loading claims/user_history/evidence_requirements, resolving
and encoding images, calling Claude per claim (with on-disk response
caching so re-running after a prior full or partial run costs zero new API
spend for unchanged rows), validating each output row against the schema,
and writing predictions to code/output.csv in the exact column order
dataset/output.csv uses.

Requires ANTHROPIC_API_KEY in the environment (see README.md for setup).

Run directly:
    python3 code/main.py
"""

from __future__ import annotations

import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from pipeline.run_full_eval import main as run_full_eval_main  # noqa: E402


def main() -> int:
    return run_full_eval_main()


if __name__ == "__main__":
    raise SystemExit(main())
