"""Scoring harness for evidence-review predictions against ground truth.

Compares a predictions CSV (e.g. produced by
pipeline/run_sample_eval.py, written with schema.OUTPUT_COLUMNS as the
header) against the ground-truth columns carried by
dataset/sample_claims.csv (via data.Claim.ground_truth), and prints/writes
a column-by-column metrics summary.

Scoring rules:
  - Single-valued enum fields (evidence_standard_met, valid_image,
    claim_status, issue_type, object_part, severity): exact-match accuracy.
  - Multi-valued semicolon-delimited fields (risk_flags,
    supporting_image_ids): set-based precision/recall/F1 (split on ";",
    compare as sets) so flag/id order and whitespace don't matter.
  - Free-text fields (evidence_standard_met_reason,
    claim_status_justification) are intentionally NOT scored.

Rows are matched between predictions and ground truth by user_id +
image_paths (the deterministic echo of the input row), not by row order,
so a predictions file with skipped/failed rows still scores correctly
against whatever ground-truth rows it actually covers.

This module must stay fully general: it must never branch on case_id,
user_id, or any other case-specific identifier.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = REPO_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from pipeline.data import Claim, load_claims  # noqa: E402

DATASET_ROOT = REPO_ROOT / "dataset"
SAMPLE_CLAIMS_PATH = DATASET_ROOT / "sample_claims.csv"

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "pipeline" / "outputs"
DEFAULT_METRICS_PATH = OUTPUT_DIR / "strategy_a_sample_metrics.json"

# Single-valued fields scored by exact match.
EXACT_MATCH_FIELDS: tuple[str, ...] = (
    "evidence_standard_met",
    "valid_image",
    "claim_status",
    "issue_type",
    "object_part",
    "severity",
)

# Multi-valued, semicolon-delimited fields scored by set precision/recall/F1.
SET_MATCH_FIELDS: tuple[str, ...] = (
    "risk_flags",
    "supporting_image_ids",
)


def _row_key(user_id: str, image_paths: str) -> tuple[str, str]:
    return (user_id, image_paths)


def _split_set_field(value: str) -> set[str]:
    if value is None:
        return set()
    return {part.strip() for part in value.split(";") if part.strip()}


def load_predictions(predictions_path: Path) -> dict[tuple[str, str], dict[str, str]]:
    """Load a predictions CSV into a dict keyed by (user_id, image_paths)."""
    with predictions_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = [dict(row) for row in reader]
    return {_row_key(row.get("user_id", ""), row.get("image_paths", "")): row for row in rows}


def load_ground_truth(claims_path: Path = SAMPLE_CLAIMS_PATH) -> dict[tuple[str, str], Claim]:
    """Load ground-truth claims keyed by (user_id, image_paths). Only rows
    that actually carry a ClaimGroundTruth (i.e. sample_claims.csv) are
    included."""
    claims = load_claims(claims_path)
    gt: dict[tuple[str, str], Claim] = {}
    for claim in claims:
        if claim.ground_truth is None:
            continue
        gt[_row_key(claim.user_id, claim.image_paths)] = claim
    return gt


def _precision_recall_f1(predicted: set[str], expected: set[str]) -> tuple[float, float, float]:
    if not predicted and not expected:
        return 1.0, 1.0, 1.0
    if not predicted:
        return 0.0, 0.0, 0.0
    if not expected:
        return 0.0, 0.0, 0.0
    tp = len(predicted & expected)
    precision = tp / len(predicted)
    recall = tp / len(expected)
    f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def score(
    predictions: dict[tuple[str, str], dict[str, str]],
    ground_truth: dict[tuple[str, str], Claim],
) -> dict:
    """Compute column-by-column metrics.

    Only keys present in BOTH predictions and ground_truth are scored
    (rows that failed/errored out of the prediction run are simply absent
    from `predictions` and are excluded from the denominator, with a count
    reported separately as `missing_predictions`).
    """
    matched_keys = [k for k in ground_truth if k in predictions]
    missing_keys = [k for k in ground_truth if k not in predictions]

    n = len(matched_keys)

    exact_match_results: dict[str, dict] = {}
    for field in EXACT_MATCH_FIELDS:
        correct = 0
        for key in matched_keys:
            pred_val = (predictions[key].get(field) or "").strip()
            gt_val = (getattr(ground_truth[key].ground_truth, field) or "").strip()
            if pred_val == gt_val:
                correct += 1
        accuracy = (correct / n) if n else 0.0
        exact_match_results[field] = {
            "accuracy": accuracy,
            "correct": correct,
            "total": n,
        }

    set_match_results: dict[str, dict] = {}
    for field in SET_MATCH_FIELDS:
        precisions: list[float] = []
        recalls: list[float] = []
        f1s: list[float] = []
        for key in matched_keys:
            pred_val = predictions[key].get(field) or ""
            gt_val = getattr(ground_truth[key].ground_truth, field) or ""
            pred_set = _split_set_field(pred_val)
            gt_set = _split_set_field(gt_val)
            p, r, f1 = _precision_recall_f1(pred_set, gt_set)
            precisions.append(p)
            recalls.append(r)
            f1s.append(f1)
        set_match_results[field] = {
            "precision": (sum(precisions) / n) if n else 0.0,
            "recall": (sum(recalls) / n) if n else 0.0,
            "f1": (sum(f1s) / n) if n else 0.0,
            "total": n,
        }

    overall_accuracy = (
        sum(r["accuracy"] for r in exact_match_results.values()) / len(exact_match_results)
        if exact_match_results
        else 0.0
    )
    overall_f1 = (
        sum(r["f1"] for r in set_match_results.values()) / len(set_match_results)
        if set_match_results
        else 0.0
    )

    return {
        "n_scored": n,
        "n_ground_truth": len(ground_truth),
        "n_missing_predictions": len(missing_keys),
        "missing_keys": [{"user_id": k[0], "image_paths": k[1]} for k in missing_keys],
        "exact_match_fields": exact_match_results,
        "set_match_fields": set_match_results,
        "overall_accuracy": overall_accuracy,
        "overall_f1": overall_f1,
    }


def format_summary_table(results: dict) -> str:
    lines: list[str] = []
    lines.append("=" * 80)
    lines.append("EVALUATION METRICS (Strategy A vs. sample_claims.csv ground truth)")
    lines.append("=" * 80)
    lines.append(
        f"scored rows: {results['n_scored']}/{results['n_ground_truth']} "
        f"(missing predictions: {results['n_missing_predictions']})"
    )
    if results["missing_keys"]:
        for mk in results["missing_keys"]:
            lines.append(f"  - missing prediction for user_id={mk['user_id']} image_paths={mk['image_paths']}")
    lines.append("")
    lines.append("-- Exact-match fields --")
    lines.append(f"{'field':30s} {'accuracy':>10s} {'correct/total':>15s}")
    for field, r in results["exact_match_fields"].items():
        lines.append(f"{field:30s} {r['accuracy']:>10.1%} {r['correct']:>6d}/{r['total']:<8d}")
    lines.append("")
    lines.append("-- Set-based fields (semicolon-delimited) --")
    lines.append(f"{'field':30s} {'precision':>10s} {'recall':>10s} {'f1':>10s}")
    for field, r in results["set_match_fields"].items():
        lines.append(f"{field:30s} {r['precision']:>10.1%} {r['recall']:>10.1%} {r['f1']:>10.1%}")
    lines.append("")
    lines.append(f"OVERALL exact-match accuracy (avg of {len(results['exact_match_fields'])} fields): {results['overall_accuracy']:.1%}")
    lines.append(f"OVERALL set-match F1 (avg of {len(results['set_match_fields'])} fields): {results['overall_f1']:.1%}")
    lines.append("=" * 80)
    return "\n".join(lines)


def run_scoring(
    predictions_path: Path,
    claims_path: Path = SAMPLE_CLAIMS_PATH,
    metrics_path: Optional[Path] = DEFAULT_METRICS_PATH,
) -> dict:
    """Load predictions + ground truth, score, print summary, write JSON."""
    predictions = load_predictions(predictions_path)
    ground_truth = load_ground_truth(claims_path)
    results = score(predictions, ground_truth)

    print(format_summary_table(results))

    if metrics_path is not None:
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with metrics_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\nmetrics written to: {metrics_path}")

    return results


def main() -> int:
    predictions_path = OUTPUT_DIR / "strategy_a_sample_predictions.csv"
    if not predictions_path.exists():
        print(f"predictions file not found: {predictions_path}", file=sys.stderr)
        print("run pipeline/run_sample_eval.py first to generate predictions.", file=sys.stderr)
        return 1
    run_scoring(predictions_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
