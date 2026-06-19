"""Output schema constants, row assembly, validation, and a deterministic
prompt-injection detector for the evidence-review pipeline.

All enum values and the output column order are transcribed verbatim from
problem_statement.md ("Allowed values" / "Required output" sections).
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Output column order (problem_statement.md "Required output")
# ---------------------------------------------------------------------------

OUTPUT_COLUMNS: tuple[str, ...] = (
    "user_id",
    "image_paths",
    "user_claim",
    "claim_object",
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

# ---------------------------------------------------------------------------
# Enums (problem_statement.md "Allowed values")
# ---------------------------------------------------------------------------

CLAIM_OBJECTS: tuple[str, ...] = ("car", "laptop", "package")

CLAIM_STATUSES: tuple[str, ...] = (
    "supported",
    "contradicted",
    "not_enough_information",
)

ISSUE_TYPES: tuple[str, ...] = (
    "dent",
    "scratch",
    "crack",
    "glass_shatter",
    "broken_part",
    "missing_part",
    "torn_packaging",
    "crushed_packaging",
    "water_damage",
    "stain",
    "none",
    "unknown",
)

CAR_OBJECT_PARTS: tuple[str, ...] = (
    "front_bumper",
    "rear_bumper",
    "door",
    "hood",
    "windshield",
    "side_mirror",
    "headlight",
    "taillight",
    "fender",
    "quarter_panel",
    "body",
    "unknown",
)

LAPTOP_OBJECT_PARTS: tuple[str, ...] = (
    "screen",
    "keyboard",
    "trackpad",
    "hinge",
    "lid",
    "corner",
    "port",
    "base",
    "body",
    "unknown",
)

PACKAGE_OBJECT_PARTS: tuple[str, ...] = (
    "box",
    "package_corner",
    "package_side",
    "seal",
    "label",
    "contents",
    "item",
    "unknown",
)

# claim_object -> allowed object_part enum. object_part has a DIFFERENT
# enum list per claim_object.
OBJECT_PART_BY_CLAIM_OBJECT: dict[str, tuple[str, ...]] = {
    "car": CAR_OBJECT_PARTS,
    "laptop": LAPTOP_OBJECT_PARTS,
    "package": PACKAGE_OBJECT_PARTS,
}

RISK_FLAGS: tuple[str, ...] = (
    "none",
    "blurry_image",
    "cropped_or_obstructed",
    "low_light_or_glare",
    "wrong_angle",
    "wrong_object",
    "wrong_object_part",
    "damage_not_visible",
    "claim_mismatch",
    "possible_manipulation",
    "non_original_image",
    "text_instruction_present",
    "user_history_risk",
    "manual_review_required",
)

SEVERITIES: tuple[str, ...] = ("none", "low", "medium", "high", "unknown")

BOOLEAN_STRINGS: tuple[str, ...] = ("true", "false")

# Columns that are free-text (not validated against an enum) but must be
# present (non-None) in every output row.
_FREE_TEXT_COLUMNS: tuple[str, ...] = (
    "user_id",
    "image_paths",
    "user_claim",
    "evidence_standard_met_reason",
    "claim_status_justification",
    "supporting_image_ids",
)


# ---------------------------------------------------------------------------
# Row assembly
# ---------------------------------------------------------------------------


def build_output_row(
    *,
    user_id: str,
    image_paths: str,
    user_claim: str,
    claim_object: str,
    evidence_standard_met: str,
    evidence_standard_met_reason: str,
    risk_flags: str,
    issue_type: str,
    object_part: str,
    claim_status: str,
    claim_status_justification: str,
    supporting_image_ids: str,
    valid_image: str,
    severity: str,
) -> dict[str, str]:
    """Assemble a final output row dict, keyed by the exact 14 output
    columns in OUTPUT_COLUMNS order (dict insertion order matches, so this
    can be passed straight to csv.DictWriter(fieldnames=OUTPUT_COLUMNS)).
    """
    row = {
        "user_id": user_id,
        "image_paths": image_paths,
        "user_claim": user_claim,
        "claim_object": claim_object,
        "evidence_standard_met": evidence_standard_met,
        "evidence_standard_met_reason": evidence_standard_met_reason,
        "risk_flags": risk_flags,
        "issue_type": issue_type,
        "object_part": object_part,
        "claim_status": claim_status,
        "claim_status_justification": claim_status_justification,
        "supporting_image_ids": supporting_image_ids,
        "valid_image": valid_image,
        "severity": severity,
    }
    return {col: row[col] for col in OUTPUT_COLUMNS}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate_risk_flags(value: str, errors: list[str]) -> None:
    if value is None or value == "":
        errors.append("risk_flags: must not be empty (use 'none' if no flags apply)")
        return
    flags = [f.strip() for f in value.split(";") if f.strip()]
    if not flags:
        errors.append("risk_flags: must not be empty (use 'none' if no flags apply)")
        return
    if "none" in flags and len(flags) > 1:
        errors.append("risk_flags: 'none' must not be combined with other flags")
    invalid = [f for f in flags if f not in RISK_FLAGS]
    if invalid:
        errors.append(f"risk_flags: invalid value(s) {invalid}, allowed: {RISK_FLAGS}")


def _validate_supporting_image_ids(value: str, valid_image_ids: set[str] | None, errors: list[str]) -> None:
    if value is None or value == "":
        errors.append(
            "supporting_image_ids: must not be empty (use 'none' if no image is sufficient)"
        )
        return
    if value == "none":
        return
    ids = [v.strip() for v in value.split(";") if v.strip()]
    if not ids:
        errors.append(
            "supporting_image_ids: must not be empty (use 'none' if no image is sufficient)"
        )
        return
    if valid_image_ids is not None:
        unknown = [i for i in ids if i not in valid_image_ids]
        if unknown:
            errors.append(
                f"supporting_image_ids: id(s) {unknown} not found among submitted image_ids {sorted(valid_image_ids)}"
            )


def validate_output_row(
    row: dict[str, str], valid_image_ids: set[str] | None = None
) -> tuple[bool, list[str]]:
    """Validate a candidate output row dict against the required columns
    and enum constraints.

    valid_image_ids, if provided, is the set of image IDs actually
    submitted for this claim (parsed from image_paths); when given,
    supporting_image_ids entries are checked for membership in that set.

    Returns (is_valid, errors). errors is empty iff is_valid is True.
    """
    errors: list[str] = []

    missing_columns = [c for c in OUTPUT_COLUMNS if c not in row]
    if missing_columns:
        errors.append(f"missing required column(s): {missing_columns}")
        return False, errors

    for col in _FREE_TEXT_COLUMNS:
        if row.get(col) is None or row.get(col) == "":
            # image_paths/user_claim/user_id are expected to be non-empty
            # echoes of the input; reasons/justifications must be filled in.
            errors.append(f"{col}: must not be empty")

    claim_object = row.get("claim_object", "")
    if claim_object not in CLAIM_OBJECTS:
        errors.append(f"claim_object: invalid value {claim_object!r}, allowed: {CLAIM_OBJECTS}")

    evidence_standard_met = row.get("evidence_standard_met", "")
    if evidence_standard_met not in BOOLEAN_STRINGS:
        errors.append(
            f"evidence_standard_met: invalid value {evidence_standard_met!r}, allowed: {BOOLEAN_STRINGS}"
        )

    valid_image = row.get("valid_image", "")
    if valid_image not in BOOLEAN_STRINGS:
        errors.append(f"valid_image: invalid value {valid_image!r}, allowed: {BOOLEAN_STRINGS}")

    issue_type = row.get("issue_type", "")
    if issue_type not in ISSUE_TYPES:
        errors.append(f"issue_type: invalid value {issue_type!r}, allowed: {ISSUE_TYPES}")

    claim_status = row.get("claim_status", "")
    if claim_status not in CLAIM_STATUSES:
        errors.append(f"claim_status: invalid value {claim_status!r}, allowed: {CLAIM_STATUSES}")

    severity = row.get("severity", "")
    if severity not in SEVERITIES:
        errors.append(f"severity: invalid value {severity!r}, allowed: {SEVERITIES}")

    object_part = row.get("object_part", "")
    allowed_parts = OBJECT_PART_BY_CLAIM_OBJECT.get(claim_object)
    if allowed_parts is None:
        # claim_object itself already invalid and reported above; still
        # check object_part against the union of all known parts so we
        # surface a useful error rather than skipping the field entirely.
        all_parts = set(CAR_OBJECT_PARTS) | set(LAPTOP_OBJECT_PARTS) | set(PACKAGE_OBJECT_PARTS)
        if object_part not in all_parts:
            errors.append(f"object_part: invalid value {object_part!r}")
    elif object_part not in allowed_parts:
        errors.append(
            f"object_part: invalid value {object_part!r} for claim_object={claim_object!r}, "
            f"allowed: {allowed_parts}"
        )

    _validate_risk_flags(row.get("risk_flags", ""), errors)
    _validate_supporting_image_ids(row.get("supporting_image_ids", ""), valid_image_ids, errors)

    return len(errors) == 0, errors


# ---------------------------------------------------------------------------
# Deterministic prompt-injection / instruction-override detector
# ---------------------------------------------------------------------------
#
# This is a safety NET, not the primary defense: a generic, non-case-specific
# regex scan over conversation/user_claim text (and could be applied to OCR'd
# image text later) that flags common attempts to manipulate an LLM-based
# reviewer into ignoring instructions, auto-approving a claim, or skipping
# review. It must never branch on a specific case ID, user ID, or expected
# label - only on the text content itself.

_INJECTION_PATTERNS: tuple[re.Pattern, ...] = (
    re.compile(r"ignore\s+(all\s+)?(previous|prior)\s+instructions", re.IGNORECASE),
    re.compile(r"disregard\s+(all\s+)?(previous|prior)\s+instructions", re.IGNORECASE),
    re.compile(r"approve\s+(this|the)\s+claim", re.IGNORECASE),
    re.compile(r"skip\s+(the\s+)?(manual\s+)?review", re.IGNORECASE),
    re.compile(r"bypass\s+(the\s+)?(manual\s+)?review", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+(in\s+)?(developer|admin|debug)\s+mode", re.IGNORECASE),
    re.compile(r"system\s*:\s*override", re.IGNORECASE),
    re.compile(r"do\s+not\s+(flag|reject|deny)\s+this\s+claim", re.IGNORECASE),
    re.compile(r"mark\s+(this\s+)?(claim\s+)?as\s+(supported|approved|valid)", re.IGNORECASE),
)


def detect_injection_attempt(text: str) -> bool:
    """Return True if text contains a generic prompt-injection / instruction
    override pattern (e.g. "ignore previous instructions", "approve this
    claim", "skip the manual review"). Case-insensitive. This is a
    deterministic safety net, not the primary defense against injected
    instructions in claim text or image content.
    """
    if not text:
        return False
    return any(pattern.search(text) for pattern in _INJECTION_PATTERNS)
