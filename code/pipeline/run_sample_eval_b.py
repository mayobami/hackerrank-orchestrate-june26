"""Full sample-set runner for Strategy B (decomposed two-call).

Loads all rows of dataset/sample_claims.csv (plus user_history.csv and
evidence_requirements.csv), runs Strategy B (vision-analysis stage 1 +
text-only aggregation/decision stage 2, two Claude calls per claim) on every
row, validates each resulting output row against the schema, and writes
predictions to a CSV -- mirroring run_sample_eval.py exactly except for the
strategy called and the output path, so the two strategies' predictions are
directly comparable on the same 20 sample rows.

Per-row failures (API errors, validation failures, image load errors) are
caught and logged so a single bad row never aborts the run for the other
rows. Progress, total token usage (summed across BOTH stages per claim), and
wall-clock runtime are printed at the end since this feeds the
operational-analysis section of the evaluation report.

This module must stay fully general: it must never branch on case_id,
user_id, or any other case-specific identifier.

Run directly:
    source "$HOME/.hackerrank_orchestrate_env" && python3 code/pipeline/run_sample_eval_b.py
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = REPO_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from pipeline import schema  # noqa: E402
from pipeline.client import get_client  # noqa: E402
from pipeline.data import (  # noqa: E402
    Claim,
    load_claims,
    load_evidence_requirements,
    load_user_history,
    lookup_evidence_requirements,
    lookup_user_history,
)
from pipeline.images import load_images_for_claim  # noqa: E402
from pipeline.strategies.strategy_b_decomposed import run_strategy_b  # noqa: E402

DATASET_ROOT = REPO_ROOT / "dataset"
SAMPLE_CLAIMS_PATH = DATASET_ROOT / "sample_claims.csv"
USER_HISTORY_PATH = DATASET_ROOT / "user_history.csv"
EVIDENCE_REQUIREMENTS_PATH = DATASET_ROOT / "evidence_requirements.csv"

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
DEFAULT_PREDICTIONS_PATH = OUTPUT_DIR / "strategy_b_sample_predictions.csv"


class RowFailure(Exception):
    """Raised internally to record why a single row could not be scored."""


def _process_claim(
    claim: Claim,
    user_history: dict,
    evidence_requirements: list,
) -> tuple[dict[str, str], Optional[dict]]:
    """Run the full per-row pipeline for one claim.

    Returns (output_row, total_usage_dict). total_usage_dict sums
    input_tokens/output_tokens across BOTH of Strategy B's stages (image
    load problems are tolerated individually by load_images_for_claim/
    ImageRecord.error and surfaced to the model rather than raised here;
    this only raises for API/validation failures that should mark the
    whole row as failed).
    """
    image_records = load_images_for_claim(claim.image_ids, claim.image_paths_list, DATASET_ROOT)

    user_hist = lookup_user_history(user_history, claim.user_id)
    relevant_requirements = lookup_evidence_requirements(evidence_requirements, claim.claim_object)

    try:
        output_row, decomposed_result = run_strategy_b(
            claim,
            image_records,
            user_hist,
            relevant_requirements,
        )
    except Exception as exc:  # noqa: BLE001 - any API/SDK failure should fail just this row
        raise RowFailure(f"Strategy B call failed: {exc}") from exc

    valid_image_ids = set(claim.image_ids)
    is_valid, errors = schema.validate_output_row(output_row, valid_image_ids=valid_image_ids)
    if not is_valid:
        raise RowFailure(f"output row failed validation: {errors}")

    return output_row, decomposed_result.total_usage


def run_sample_eval_b(
    *,
    predictions_path: Path = DEFAULT_PREDICTIONS_PATH,
    claims_path: Path = SAMPLE_CLAIMS_PATH,
) -> dict:
    """Run Strategy B over every row of claims_path and write predictions.

    Returns a summary dict with rows processed/failed, total usage (summed
    across both stages per claim), and elapsed wall-clock time, for callers
    (e.g. the evaluation harness) that want to report on the run without
    re-parsing stdout.
    """
    start = time.time()

    claims = load_claims(claims_path)
    user_history = load_user_history(USER_HISTORY_PATH)
    evidence_requirements = load_evidence_requirements(EVIDENCE_REQUIREMENTS_PATH)

    client = get_client()

    total = len(claims)
    rows_written: list[dict[str, str]] = []
    errors: list[dict[str, str]] = []
    total_input_tokens = 0
    total_output_tokens = 0

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loaded {total} claim(s) from {claims_path}")
    print(f"Loaded {len(user_history)} user_history row(s)")
    print(f"Loaded {len(evidence_requirements)} evidence_requirements row(s)")
    print()

    for i, claim in enumerate(claims, start=1):
        try:
            output_row, usage = _process_claim(claim, user_history, evidence_requirements)
        except RowFailure as exc:
            print(f"[{i}/{total}] user_id={claim.user_id!r} row={claim.row_index} FAILED: {exc}")
            errors.append({"row_index": str(claim.row_index), "user_id": claim.user_id, "error": str(exc)})
            continue
        except Exception as exc:  # noqa: BLE001 - last-resort guard so one row never aborts the run
            print(f"[{i}/{total}] user_id={claim.user_id!r} row={claim.row_index} FAILED (unexpected): {exc}")
            errors.append({"row_index": str(claim.row_index), "user_id": claim.user_id, "error": f"unexpected: {exc}"})
            continue

        rows_written.append(output_row)
        if usage:
            total_input_tokens += usage.get("input_tokens", 0) or 0
            total_output_tokens += usage.get("output_tokens", 0) or 0

        print(f"[{i}/{total}] user_id={claim.user_id!r} row={claim.row_index} OK "
              f"(claim_status={output_row['claim_status']}, processed {len(rows_written)}/{total})")

    with predictions_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=schema.OUTPUT_COLUMNS)
        writer.writeheader()
        for row in rows_written:
            writer.writerow(row)

    elapsed = time.time() - start

    print()
    print("=" * 80)
    print("RUN SUMMARY (Strategy B: decomposed two-call)")
    print("=" * 80)
    print(f"rows processed successfully: {len(rows_written)}/{total}")
    print(f"rows failed: {len(errors)}/{total}")
    for err in errors:
        print(f"  - row {err['row_index']} (user_id={err['user_id']}): {err['error']}")
    print(f"total input_tokens (both stages): {total_input_tokens}")
    print(f"total output_tokens (both stages): {total_output_tokens}")
    print(f"total tokens (both stages): {total_input_tokens + total_output_tokens}")
    print(f"wall-clock runtime: {elapsed:.2f}s")
    print(f"predictions written to: {predictions_path}")

    return {
        "total": total,
        "succeeded": len(rows_written),
        "failed": len(errors),
        "errors": errors,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "elapsed_seconds": elapsed,
        "predictions_path": str(predictions_path),
        "rows": rows_written,
    }


def main() -> int:
    summary = run_sample_eval_b()
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
