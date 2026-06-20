"""Deterministic, file-based response cache for claim judgments.

Re-running the pipeline over inputs that have already been scored (same
claim fields, same strategy, same model) should cost zero new API spend.
This module provides a tiny content-addressed cache: a cache key is derived
from the claim's actual varying inputs plus the strategy/model that would
produce the judgment, and the result (output row + usage + metadata) is
stored as one JSON file per key under code/pipeline/cache/.

Cache key inputs, deliberately narrow:
  - user_id, image_paths, user_claim, claim_object: the only fields that
    vary per claim row. user_history and evidence_requirements are NOT
    included because they are looked up deterministically FROM user_id and
    claim_object (see data.lookup_user_history / lookup_evidence_requirements)
    -- for a fixed user_id/claim_object pair those lookups always return the
    same data (as of a given dataset snapshot), so including them in the key
    would be redundant, not safer.
  - strategy_name, model: included so a future change to the strategy or
    model never silently reuses a stale prediction produced by a different
    configuration.

The cache directory (code/pipeline/cache/) is already gitignored.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Optional

from .data import Claim

CACHE_DIR = Path(__file__).resolve().parent / "cache"


def _canonical_key_payload(claim: Claim, strategy_name: str, model: str) -> dict[str, Any]:
    """Build the canonical (stable, order-independent) dict that the cache
    key hash is computed over.

    Only the claim's actual varying inputs are included, plus the
    strategy/model that would produce the judgment. image_paths is the raw
    semicolon-separated string already on the claim (the canonical ordered
    form), so no need to separately include image_paths_list/image_ids.
    """
    return {
        "user_id": claim.user_id,
        "image_paths": claim.image_paths,
        "user_claim": claim.user_claim,
        "claim_object": claim.claim_object,
        "strategy_name": strategy_name,
        "model": model,
    }


def cache_key(claim: Claim, strategy_name: str, model: str) -> str:
    """Compute a stable sha256 hex digest cache key for one claim under a
    given strategy/model configuration."""
    payload = _canonical_key_payload(claim, strategy_name, model)
    canonical_json = json.dumps(payload, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


def _cache_path(key: str) -> Path:
    return CACHE_DIR / f"{key}.json"


def get_cached(claim: Claim, strategy_name: str, model: str) -> Optional[dict[str, Any]]:
    """Return the previously-stored result dict for this claim under this
    strategy/model, or None if no cache entry exists (or it cannot be
    parsed, in which case it is treated as a miss rather than raising)."""
    key = cache_key(claim, strategy_name, model)
    path = _cache_path(key)
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def store_cached(
    claim: Claim,
    output_row: dict[str, str],
    usage: Optional[dict[str, Any]],
    strategy_name: str,
    model: str,
) -> None:
    """Persist a result dict (output row + usage + provenance + timestamp)
    for this claim under this strategy/model configuration."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = cache_key(claim, strategy_name, model)
    record = {
        "output_row": output_row,
        "usage": usage,
        "strategy_name": strategy_name,
        "model": model,
        "cached_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    path = _cache_path(key)
    tmp_path = path.with_suffix(".json.tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
    tmp_path.replace(path)
