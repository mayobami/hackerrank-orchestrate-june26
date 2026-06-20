"""Evaluation entry point (AGENTS.md section 6.1 designated entry point).

Ties together the Strategy A sample runner and the scoring harness:

  1. Loads dataset/sample_claims.csv ground truth.
  2. Loads existing predictions from pipeline/outputs/strategy_a_sample_predictions.csv
     if present; otherwise generates them by running pipeline/run_sample_eval.py
     (this makes live Anthropic API calls -- requires ANTHROPIC_API_KEY in the
     environment).
  3. Scores the predictions against ground truth and prints/writes the
     summary produced by evaluation/scorer.py.

Run directly:
    python3 code/evaluation/main.py

    # Force regeneration of predictions even if a predictions file already
    # exists:
    python3 code/evaluation/main.py --regenerate
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = REPO_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from evaluation.scorer import (  # noqa: E402
    DEFAULT_METRICS_PATH,
    SAMPLE_CLAIMS_PATH,
    run_scoring,
)
from pipeline.run_sample_eval import (  # noqa: E402
    DEFAULT_PREDICTIONS_PATH,
    run_sample_eval,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Strategy A sample evaluation + scoring.")
    parser.add_argument(
        "--regenerate",
        action="store_true",
        help="Re-run Strategy A over sample_claims.csv even if a predictions file already exists.",
    )
    args = parser.parse_args()

    predictions_path = DEFAULT_PREDICTIONS_PATH

    if args.regenerate or not predictions_path.exists():
        if not predictions_path.exists():
            print(f"No predictions file found at {predictions_path}; generating via Strategy A...")
        else:
            print(f"--regenerate passed; re-running Strategy A over sample_claims.csv...")
        run_sample_eval(predictions_path=predictions_path, claims_path=SAMPLE_CLAIMS_PATH)
        print()
    else:
        print(f"Using existing predictions file: {predictions_path}")
        print("(pass --regenerate to force a fresh Strategy A run)")
        print()

    run_scoring(predictions_path, claims_path=SAMPLE_CLAIMS_PATH, metrics_path=DEFAULT_METRICS_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
