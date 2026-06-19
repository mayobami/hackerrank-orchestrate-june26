"""Throwaway single-case prototype runner for Strategy A.

Loads dataset/sample_claims.csv, dataset/user_history.csv, and
dataset/evidence_requirements.csv via the existing loaders, picks the row
for user_id == "user_001" (case_001 -- rear bumper dent, single image),
loads its images, runs Strategy A, and prints both the raw model tool-call
output and the final assembled+validated output row.

Row selection here is intentionally hardcoded to user_001 for this
prototype script ONLY -- this is just "pick a row to demo", not test-case
logic baked into the pipeline itself (client.py and strategy_a_monolithic.py
contain no case-specific branching).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = REPO_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from pipeline import schema  # noqa: E402
from pipeline.data import (  # noqa: E402
    load_claims,
    load_evidence_requirements,
    load_user_history,
    lookup_evidence_requirements,
    lookup_user_history,
)
from pipeline.images import load_images_for_claim  # noqa: E402
from pipeline.strategies.strategy_a_monolithic import run_strategy_a  # noqa: E402

DEMO_USER_ID = "user_001"


def main() -> int:
    dataset_root = REPO_ROOT / "dataset"

    claims = load_claims(dataset_root / "sample_claims.csv")
    user_history = load_user_history(dataset_root / "user_history.csv")
    evidence_requirements = load_evidence_requirements(dataset_root / "evidence_requirements.csv")

    claim = next((c for c in claims if c.user_id == DEMO_USER_ID), None)
    if claim is None:
        print(f"No row found for user_id={DEMO_USER_ID!r} in sample_claims.csv", file=sys.stderr)
        return 1

    print("=" * 80)
    print("INPUT CLAIM")
    print("=" * 80)
    print(f"user_id: {claim.user_id}")
    print(f"claim_object: {claim.claim_object}")
    print(f"image_paths: {claim.image_paths}")
    print(f"image_ids: {claim.image_ids}")
    print(f"user_claim: {claim.user_claim}")
    if claim.ground_truth is not None:
        print("-" * 80)
        print("GROUND TRUTH (sample_claims.csv)")
        print("-" * 80)
        for field_name in (
            "evidence_standard_met",
            "evidence_standard_met_reason",
            "risk_flags",
            "issue_type",
            "object_part",
            "claim_status",
            "claim_status_justification",
            "supporting_image_ids",
            "valid_image",
            "severity",
        ):
            print(f"{field_name}: {getattr(claim.ground_truth, field_name)}")

    image_records = load_images_for_claim(claim.image_ids, claim.image_paths_list, dataset_root)

    print()
    print("=" * 80)
    print("LOADED IMAGES")
    print("=" * 80)
    for rec in image_records:
        if rec.ok:
            print(f"{rec.image_id}: OK format={rec.real_format} media_type={rec.media_type} transcoded={rec.transcoded}")
        else:
            print(f"{rec.image_id}: ERROR {rec.error}")

    user_hist = lookup_user_history(user_history, claim.user_id)
    relevant_requirements = lookup_evidence_requirements(evidence_requirements, claim.claim_object)

    print()
    print("=" * 80)
    print("CALLING STRATEGY A (single monolithic Claude call)")
    print("=" * 80)

    output_row, judgment = run_strategy_a(
        claim,
        image_records,
        user_hist,
        relevant_requirements,
    )

    print()
    print("=" * 80)
    print("RAW MODEL TOOL-CALL OUTPUT (unprocessed)")
    print("=" * 80)
    print(json.dumps(judgment.tool_input, indent=2))
    print()
    print(f"model: {judgment.model}")
    print(f"stop_reason: {judgment.stop_reason}")
    print(f"usage: {judgment.usage}")

    print()
    print("=" * 80)
    print("FINAL ASSEMBLED OUTPUT ROW")
    print("=" * 80)
    for col in schema.OUTPUT_COLUMNS:
        print(f"{col}: {output_row[col]}")

    valid_image_ids = set(claim.image_ids)
    is_valid, errors = schema.validate_output_row(output_row, valid_image_ids=valid_image_ids)

    print()
    print("=" * 80)
    print("VALIDATION")
    print("=" * 80)
    print(f"valid: {is_valid}")
    if errors:
        for err in errors:
            print(f"  - {err}")

    return 0 if is_valid else 2


if __name__ == "__main__":
    raise SystemExit(main())
