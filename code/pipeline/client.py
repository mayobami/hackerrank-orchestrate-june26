"""Thin Anthropic Messages API wrapper for the evidence-review pipeline.

Responsibilities (and only these):
  - read ANTHROPIC_API_KEY from the environment (never hardcoded, never
    read from a .env file in this repo)
  - build a vision + text Messages API call
  - force structured output via a single `strict: true` tool definition
    (tool_choice forces that exact tool) so the response is schema
    guaranteed to match the judgment fields in schema.py
  - return the parsed tool-call input dict (plus the raw SDK response, for
    callers that want usage/stop_reason/etc.)

Retries for 429/5xx are left to the SDK's built-in retry behavior
(Anthropic() defaults to retrying transient errors) rather than hand
rolled here; that is a deliberate scope decision for this stage.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional

import anthropic

from . import schema

# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

# Default model used by every strategy unless explicitly overridden. Kept as
# a single named constant (not buried inline) so cost/quality comparisons
# across models (e.g. a cheaper Haiku/Sonnet pass) only require passing a
# different `model=` argument into call_judgment(), not editing call sites.
DEFAULT_MODEL = "claude-opus-4-8"

# Generous enough for a multi-image, multi-field structured judgment without
# being unbounded.
DEFAULT_MAX_TOKENS = 2048

JUDGMENT_TOOL_NAME = "submit_claim_judgment"


# ---------------------------------------------------------------------------
# Tool schema (mirrors OUTPUT_COLUMNS minus the pure input echoes)
# ---------------------------------------------------------------------------
#
# user_id / image_paths / user_claim / claim_object are NOT asked of the
# model: they are deterministic echoes of the input Claim and are filled in
# by the calling strategy code via schema.build_output_row(...).
#
# object_part's *real* enum depends on claim_object (car/laptop/package each
# have a different allowed list). JSON Schema does not cleanly express a
# single tool-call-wide conditional enum across sibling properties, so this
# tool schema uses the UNION of all three object-part enums; the stricter,
# claim_object-aware check happens downstream in schema.validate_output_row.

_OBJECT_PART_UNION: tuple[str, ...] = tuple(
    sorted(
        set(schema.CAR_OBJECT_PARTS)
        | set(schema.LAPTOP_OBJECT_PARTS)
        | set(schema.PACKAGE_OBJECT_PARTS)
    )
)


def _build_judgment_tool() -> dict[str, Any]:
    """Build the strict tool definition the model must call.

    Property order intentionally matches the order fields are listed in the
    task description (evidence -> risk -> issue/part -> status -> support ->
    image validity -> severity) so the model reasons through them in a
    sensible sequence.
    """
    return {
        "name": JUDGMENT_TOOL_NAME,
        "description": (
            "Submit the structured evidence-review judgment for one damage "
            "claim. Every field is required. Use only the enum values listed "
            "for fields that have an enum; never invent new values."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "evidence_standard_met": {
                    "type": "boolean",
                    "description": (
                        "true if the submitted image set is sufficient to "
                        "evaluate the claim against the evidence "
                        "requirements provided; otherwise false."
                    ),
                },
                "evidence_standard_met_reason": {
                    "type": "string",
                    "description": "Short reason for the evidence_standard_met decision.",
                },
                "risk_flags": {
                    "type": "array",
                    "description": (
                        "Risk flags that apply to this claim, chosen only "
                        "from the allowed list. Use an empty array if (and "
                        "only if) no risk flags apply other than 'none'; "
                        "otherwise list every flag that applies. Do not mix "
                        "'none' with other flags."
                    ),
                    "items": {"type": "string", "enum": list(schema.RISK_FLAGS)},
                },
                "issue_type": {
                    "type": "string",
                    "enum": list(schema.ISSUE_TYPES),
                    "description": "The visible issue type, or 'none'/'unknown'.",
                },
                "object_part": {
                    "type": "string",
                    "enum": list(_OBJECT_PART_UNION),
                    "description": (
                        "The relevant object part. Must be a part that is "
                        "valid for the claim's claim_object (car/laptop/"
                        "package parts differ); use 'unknown' if it cannot "
                        "be determined."
                    ),
                },
                "claim_status": {
                    "type": "string",
                    "enum": list(schema.CLAIM_STATUSES),
                    "description": "Final decision on whether the claim is supported by the images.",
                },
                "claim_status_justification": {
                    "type": "string",
                    "description": (
                        "Concise, image-grounded explanation for claim_status. "
                        "Mention relevant image IDs when helpful."
                    ),
                },
                "supporting_image_ids": {
                    "type": "array",
                    "description": (
                        "Image IDs (e.g. 'img_1') that support the decision, "
                        "evaluated per-image. Empty array if no image is "
                        "sufficient."
                    ),
                    "items": {"type": "string"},
                },
                "valid_image": {
                    "type": "boolean",
                    "description": (
                        "true if the submitted image set is usable for "
                        "automated review at all (right subject, "
                        "decodable, not purely irrelevant/garbage); "
                        "otherwise false."
                    ),
                },
                "severity": {
                    "type": "string",
                    "enum": list(schema.SEVERITIES),
                    "description": "Estimated severity of the visible issue.",
                },
            },
            "required": [
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
            ],
        },
    }


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


@dataclass
class JudgmentResult:
    """Result of a single structured judgment call."""

    tool_input: dict[str, Any]
    model: str
    stop_reason: Optional[str]
    usage: Optional[dict[str, Any]]
    raw_response: Any


def _get_api_key() -> str:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set in the environment. This pipeline "
            "reads credentials from the environment only (no .env file is "
            "read or created); export ANTHROPIC_API_KEY before running."
        )
    return api_key


def get_client() -> anthropic.Anthropic:
    """Build an Anthropic SDK client. Relies on the SDK's built-in retry
    behavior for transient 429/5xx errors (default max_retries)."""
    return anthropic.Anthropic(api_key=_get_api_key())


def call_judgment(
    *,
    system_prompt: str,
    content_blocks: list[dict[str, Any]],
    model: str = DEFAULT_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    client: Optional[anthropic.Anthropic] = None,
) -> JudgmentResult:
    """Call the Messages API with a single forced strict tool call.

    content_blocks is the full ordered list of user-turn content blocks
    (text + image blocks) the caller has already assembled; this function
    does not interpret or reorder them.

    Returns the parsed tool_input dict from the (sole, forced) tool_use
    block, plus basic response metadata for logging/cost tracking.
    """
    active_client = client or get_client()
    tool = _build_judgment_tool()

    response = active_client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system_prompt,
        tools=[tool],
        tool_choice={"type": "tool", "name": JUDGMENT_TOOL_NAME},
        messages=[{"role": "user", "content": content_blocks}],
    )

    tool_use_block = None
    for block in response.content:
        if getattr(block, "type", None) == "tool_use" and block.name == JUDGMENT_TOOL_NAME:
            tool_use_block = block
            break

    if tool_use_block is None:
        raise RuntimeError(
            f"Model response did not contain a '{JUDGMENT_TOOL_NAME}' tool_use "
            f"block (stop_reason={response.stop_reason!r}); cannot extract "
            "structured judgment."
        )

    usage = None
    if response.usage is not None:
        usage = {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        }

    return JudgmentResult(
        tool_input=dict(tool_use_block.input),
        model=response.model,
        stop_reason=response.stop_reason,
        usage=usage,
        raw_response=response,
    )
