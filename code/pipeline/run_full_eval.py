"""Full production runner for Strategy A over dataset/claims.csv.

Loads all rows of dataset/claims.csv (the unlabeled, real submission set;
44 rows as of this writing), plus user_history.csv and
evidence_requirements.csv, runs Strategy A (one monolithic Claude call per
claim) on every row, validates each resulting output row against the
schema, and writes predictions to code/output.csv in schema.OUTPUT_COLUMNS
column order -- the exact layout dataset/output.csv uses.

Like run_sample_eval.py, this checks the on-disk response cache
(pipeline/cache_store.py) before calling the API for each row: a cache hit
reuses the previously stored row with zero new API spend, and only a cache
miss calls run_strategy_a. This means re-running this script after a prior
full (or partial) run completes in seconds for rows that already have a
cached result.

Per-row failures (API errors, validation failures, image load errors) are
caught and logged so a single bad row never aborts the run for the other
rows. Progress, total token usage, cache-hit counts, and wall-clock runtime
are printed at the end.

This module must stay fully general: it must never branch on case_id,
user_id, or any other case-specific identifier.

Run directly:
    source "$HOME/.hackerrank_orchestrate_env" && python3 code/pipeline/run_full_eval.py
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
from pipeline.cache_store import get_cached, store_cached  # noqa: E402
from pipeline.client import DEFAULT_MODEL, get_client  # noqa: E402
from pipeline.data import (  # noqa: E402
    Claim,
    load_claims,
    load_evidence_requirements,
    load_user_history,
    lookup_evidence_requirements,
    lookup_user_history,
)
from pipeline.images import load_images_for_claim  # noqa: E402
from pipeline.strategies.strategy_a_monolithic import run_strategy_a  # noqa: E402

DATASET_ROOT = REPO_ROOT / "dataset"
CLAIMS_PATH = DATASET_ROOT / "claims.csv"
USER_HISTORY_PATH = DATASET_ROOT / "user_history.csv"
EVIDENCE_REQUIREMENTS_PATH = DATASET_ROOT / "evidence_requirements.csv"

DEFAULT_OUTPUT_PATH = CODE_ROOT / "output.csv"

# Strategy/model identifier folded into the response-cache key (see
# cache_store.py) so a future change to the strategy or model never
# silently reuses a stale prediction from a different configuration. Must
# match run_sample_eval.py's STRATEGY_NAME so cache entries populated by
# the sample runner are also reusable here for any overlapping rows (the
# key is content-addressed on the claim fields, not the source file).
STRATEGY_NAME = "strategy_a_monolithic"


class RowFailure(Exception):
    """Raised internally to record why a single row could not be scored."""


def _process_claim(
    claim: Claim,
    user_history: dict,
    evidence_requirements: list,
    *,
    model: str = DEFAULT_MODEL,
) -> tuple[dict[str, str], Optional[dict], bool]:
    """Run the full per-row pipeline for one claim, using the response cache
    to avoid spending new API budget on unchanged inputs.

    Returns (output_row, usage_dict, cache_hit). Raises RowFailure with a
    human readable message on any failure (image load problems are
    tolerated individually by load_images_for_claim/ImageRecord.error and
    surfaced to the model rather than raised here; this only raises for
    API/validation failures that should mark the whole row as failed).
    """
    cached = get_cached(claim, STRATEGY_NAME, model)
    if cached is not None:
        return cached["output_row"], cached.get("usage"), True

    image_records = load_images_for_claim(claim.image_ids, claim.image_paths_list, DATASET_ROOT)

    user_hist = lookup_user_history(user_history, claim.user_id)
    relevant_requirements = lookup_evidence_requirements(evidence_requirements, claim.claim_object)

    try:
        output_row, judgment = run_strategy_a(
            claim,
            image_records,
            user_hist,
            relevant_requirements,
            model=model,
        )
    except Exception as exc:  # noqa: BLE001 - any API/SDK failure should fail just this row
        raise RowFailure(f"Strategy A call failed: {exc}") from exc

    valid_image_ids = set(claim.image_ids)
    is_valid, errors = schema.validate_output_row(output_row, valid_image_ids=valid_image_ids)
    if not is_valid:
        raise RowFailure(f"output row failed validation: {errors}")

    store_cached(claim, output_row, judgment.usage, STRATEGY_NAME, model)

    return output_row, judgment.usage, False


def run_full_eval(
    *,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    claims_path: Path = CLAIMS_PATH,
) -> dict:
    """Run Strategy A over every row of claims_path and write predictions to
    output_path in schema.OUTPUT_COLUMNS order.

    Returns a summary dict with rows processed/failed, cache hits, total
    usage, and elapsed wall-clock time.
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
    cache_hits = 0

    print(f"Loaded {total} claim(s) from {claims_path}")
    print(f"Loaded {len(user_history)} user_history row(s)")
    print(f"Loaded {len(evidence_requirements)} evidence_requirements row(s)")
    print()

    for i, claim in enumerate(claims, start=1):
        try:
            output_row, usage, cache_hit = _process_claim(claim, user_history, evidence_requirements)
        except RowFailure as exc:
            print(f"[{i}/{total}] user_id={claim.user_id!r} row={claim.row_index} FAILED: {exc}")
            errors.append({"row_index": str(claim.row_index), "user_id": claim.user_id, "error": str(exc)})
            continue
        except Exception as exc:  # noqa: BLE001 - last-resort guard so one row never aborts the run
            print(f"[{i}/{total}] user_id={claim.user_id!r} row={claim.row_index} FAILED (unexpected): {exc}")
            errors.append({"row_index": str(claim.row_index), "user_id": claim.user_id, "error": f"unexpected: {exc}"})
            continue

        rows_written.append(output_row)
        if cache_hit:
            cache_hits += 1
            print(f"[{i}/{total}] user_id={claim.user_id!r} row={claim.row_index} CACHE HIT (no API call) "
                  f"(claim_status={output_row['claim_status']}, processed {len(rows_written)}/{total})")
        else:
            if usage:
                total_input_tokens += usage.get("input_tokens", 0) or 0
                total_output_tokens += usage.get("output_tokens", 0) or 0
            print(f"[{i}/{total}] user_id={claim.user_id!r} row={claim.row_index} OK "
                  f"(claim_status={output_row['claim_status']}, processed {len(rows_written)}/{total})")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=schema.OUTPUT_COLUMNS)
        writer.writeheader()
        for row in rows_written:
            writer.writerow(row)

    elapsed = time.time() - start

    print()
    print("=" * 80)
    print("RUN SUMMARY")
    print("=" * 80)
    print(f"rows processed successfully: {len(rows_written)}/{total}")
    print(f"rows failed: {len(errors)}/{total}")
    for err in errors:
        print(f"  - row {err['row_index']} (user_id={err['user_id']}): {err['error']}")
    print(f"cache hits: {cache_hits}/{total} (zero API spend on these rows)")
    print(f"total input_tokens: {total_input_tokens}")
    print(f"total output_tokens: {total_output_tokens}")
    print(f"total tokens: {total_input_tokens + total_output_tokens}")
    print(f"wall-clock runtime: {elapsed:.2f}s")
    print(f"predictions written to: {output_path}")

    return {
        "total": total,
        "succeeded": len(rows_written),
        "failed": len(errors),
        "errors": errors,
        "cache_hits": cache_hits,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "elapsed_seconds": elapsed,
        "output_path": str(output_path),
        "rows": rows_written,
    }


def main() -> int:
    summary = run_full_eval()
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
