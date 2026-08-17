from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import requests


KEYWORD_HINTS = {
    "sample_id": ["sample", "observation", "record"],
    "station_id": ["station", "site", "monitoring"],
    "lake_id": ["lake", "waterbody", "wb"],
    "sample_date": ["date", "time", "timestamp", "sampling"],
    "lat": ["lat", "latitude", "y"],
    "lon": ["lon", "longitude", "x"],
    "chl_a": ["chl", "chlorophyll"],
    "turbidity": ["turbidity", "ntu", "fnu"],
    "secchi_depth": ["secchi", "sdd", "depth"],
    "doc": ["doc", "dissolved organic carbon"],
}


@dataclass
class MappingSuggestion:
    canonical: str
    source_column: str | None
    confidence: float


def suggest_mapping_from_columns(columns: list[str]) -> list[MappingSuggestion]:
    lowered = {column: column.lower() for column in columns}
    suggestions: list[MappingSuggestion] = []

    for canonical, hints in KEYWORD_HINTS.items():
        best_col = None
        best_score = 0.0
        for source_col, source_low in lowered.items():
            score = 0.0
            for hint in hints:
                if hint in source_low:
                    score += 1.0
            if source_low == canonical:
                score += 2.0
            if score > best_score:
                best_score = score
                best_col = source_col

        conf = min(best_score / max(1.0, len(hints)), 1.0)
        suggestions.append(MappingSuggestion(canonical=canonical, source_column=best_col if conf > 0 else None, confidence=conf))

    return suggestions


def suggest_mapping_with_llm(columns: list[str], llm_cfg: dict[str, Any]) -> dict[str, str]:
    api_key = llm_cfg.get("api_key") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("LLM mapping requested but no API key provided (llm.api_key or OPENAI_API_KEY).")

    base_url = llm_cfg.get("base_url", "https://api.openai.com/v1").rstrip("/")
    model = llm_cfg.get("model", "gpt-4o-mini")

    prompt = {
        "task": "Map source table columns to canonical inland water-quality schema.",
        "canonical_fields": list(KEYWORD_HINTS.keys()),
        "columns": columns,
        "instructions": "Return strict JSON object: {canonical_field: source_column_or_null}.",
    }

    url = f"{base_url}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a strict JSON mapper for data columns."},
            {"role": "user", "content": json.dumps(prompt)},
        ],
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }

    response = requests.post(url, headers=headers, json=body, timeout=int(llm_cfg.get("timeout", 60)))
    response.raise_for_status()
    payload = response.json()
    content = payload["choices"][0]["message"]["content"]
    parsed = json.loads(content)

    result = {}
    for canonical in KEYWORD_HINTS:
        value = parsed.get(canonical)
        if isinstance(value, str) and value in columns:
            result[canonical] = value
    return result
