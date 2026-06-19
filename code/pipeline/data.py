"""CSV loading for the evidence-review pipeline.

Loads the three input CSVs (claims/sample_claims, user_history,
evidence_requirements) into small typed dataclasses instead of passing
raw csv.DictReader rows around the codebase.

All loading goes through Python's csv module with proper quoting, since
several fields (user_claim conversations, history_summary) contain
embedded commas and pipe-delimited turns.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Claim:
    """One row of claims.csv / sample_claims.csv (input fields only).

    image_ids is the ordered, parsed form of image_paths (filename stem,
    e.g. "img_1"), kept alongside the raw paths so downstream code does not
    need to re-parse the semicolon-separated string.
    """

    user_id: str
    image_paths: str
    image_paths_list: tuple[str, ...]
    image_ids: tuple[str, ...]
    user_claim: str
    claim_object: str
    row_index: int

    # Ground-truth columns, present only in sample_claims.csv. None for
    # claims.csv rows.
    ground_truth: Optional["ClaimGroundTruth"] = None


@dataclass(frozen=True)
class ClaimGroundTruth:
    """Expected-output columns present in sample_claims.csv only."""

    evidence_standard_met: str
    evidence_standard_met_reason: str
    risk_flags: str
    issue_type: str
    object_part: str
    claim_status: str
    claim_status_justification: str
    supporting_image_ids: str
    valid_image: str
    severity: str


@dataclass(frozen=True)
class UserHistory:
    """One row of user_history.csv."""

    user_id: str
    past_claim_count: str
    accept_claim: str
    manual_review_claim: str
    rejected_claim: str
    last_90_days_claim_count: str
    history_flags: str
    history_summary: str


@dataclass(frozen=True)
class EvidenceRequirement:
    """One row of evidence_requirements.csv."""

    requirement_id: str
    claim_object: str
    applies_to: str
    minimum_image_evidence: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    """Read a CSV file into a list of dicts, using the csv module so quoted
    fields with embedded commas / pipes are handled correctly."""
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return [dict(row) for row in reader]


def parse_image_paths(image_paths: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Parse a semicolon-separated image_paths field into:
      - ordered tuple of raw relative paths (whitespace-trimmed)
      - ordered tuple of image IDs (filename without extension)

    Empty/blank segments are dropped. Order is preserved since downstream
    logic (supporting_image_ids, per-image review) is order-sensitive.
    """
    if not image_paths:
        return (), ()

    paths: list[str] = []
    ids: list[str] = []
    for segment in image_paths.split(";"):
        cleaned = segment.strip()
        if not cleaned:
            continue
        paths.append(cleaned)
        ids.append(Path(cleaned).stem)

    return tuple(paths), tuple(ids)


_SAMPLE_GROUND_TRUTH_COLUMNS = (
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
)


def _build_claim(row: dict[str, str], row_index: int) -> Claim:
    raw_paths = row.get("image_paths", "") or ""
    paths_list, ids_list = parse_image_paths(raw_paths)

    ground_truth = None
    # sample_claims.csv has extra ground-truth columns; claims.csv does not.
    # Detect by presence of the columns in the parsed row (DictReader keys
    # come from the header), not by filename, so this works for any CSV
    # that happens to carry (or omit) these columns.
    if all(col in row for col in _SAMPLE_GROUND_TRUTH_COLUMNS):
        ground_truth = ClaimGroundTruth(
            **{col: row.get(col, "") or "" for col in _SAMPLE_GROUND_TRUTH_COLUMNS}
        )

    return Claim(
        user_id=row.get("user_id", "") or "",
        image_paths=raw_paths,
        image_paths_list=paths_list,
        image_ids=ids_list,
        user_claim=row.get("user_claim", "") or "",
        claim_object=(row.get("claim_object", "") or "").strip(),
        row_index=row_index,
        ground_truth=ground_truth,
    )


# ---------------------------------------------------------------------------
# Public loaders
# ---------------------------------------------------------------------------


def load_claims(csv_path: Path) -> list[Claim]:
    """Load claims.csv or sample_claims.csv into a list of Claim records.

    Works for both files: sample_claims.csv carries extra ground-truth
    columns which are captured in Claim.ground_truth when present.
    """
    rows = _read_csv_rows(csv_path)
    return [_build_claim(row, idx) for idx, row in enumerate(rows)]


def load_user_history(csv_path: Path) -> dict[str, UserHistory]:
    """Load user_history.csv into a dict keyed by user_id."""
    rows = _read_csv_rows(csv_path)
    history: dict[str, UserHistory] = {}
    for row in rows:
        uh = UserHistory(
            user_id=row.get("user_id", "") or "",
            past_claim_count=row.get("past_claim_count", "") or "",
            accept_claim=row.get("accept_claim", "") or "",
            manual_review_claim=row.get("manual_review_claim", "") or "",
            rejected_claim=row.get("rejected_claim", "") or "",
            last_90_days_claim_count=row.get("last_90_days_claim_count", "") or "",
            history_flags=row.get("history_flags", "") or "",
            history_summary=row.get("history_summary", "") or "",
        )
        history[uh.user_id] = uh
    return history


def lookup_user_history(
    history: dict[str, UserHistory], user_id: str
) -> Optional[UserHistory]:
    """Look up a single user's history record. Returns None if unknown."""
    return history.get(user_id)


def load_evidence_requirements(csv_path: Path) -> list[EvidenceRequirement]:
    """Load evidence_requirements.csv into a flat list of requirement rows."""
    rows = _read_csv_rows(csv_path)
    return [
        EvidenceRequirement(
            requirement_id=row.get("requirement_id", "") or "",
            claim_object=(row.get("claim_object", "") or "").strip(),
            applies_to=row.get("applies_to", "") or "",
            minimum_image_evidence=row.get("minimum_image_evidence", "") or "",
        )
        for row in rows
    ]


def lookup_evidence_requirements(
    requirements: list[EvidenceRequirement], claim_object: str
) -> list[EvidenceRequirement]:
    """Return all requirement rows relevant to a given claim_object: rows
    whose claim_object matches exactly, plus rows with claim_object == "all".

    Object-specific rows are returned before the "all" rows so callers that
    care about specificity (e.g. for prompting) see the more targeted rules
    first.
    """
    specific = [r for r in requirements if r.claim_object == claim_object]
    general = [r for r in requirements if r.claim_object == "all"]
    return specific + general
