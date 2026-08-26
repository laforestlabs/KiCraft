"""Preflight exact OpenRouter role models before a paid campaign."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import requests

from kicraft.server.client import CappedOpenRouterClient
from kicraft.server.config import DESIGN_PROFILES, Settings

_ROLE_PARAMETERS = {
    "designer": {"reasoning", "response_format", "tools", "tool_choice"},
    "reviewer": {"reasoning", "response_format"},
    "judge": {"reasoning", "response_format"},
}
_SMOKE_SCHEMA = {
    "type": "object",
    "properties": {"ok": {"type": "boolean"}},
    "required": ["ok"],
    "additionalProperties": False,
}


def _provider_matches(endpoint: dict, selected: str) -> bool:
    selected = selected.lower()
    tag = str(endpoint.get("tag") or "").lower()
    name = str(endpoint.get("provider_name") or "").lower()
    return selected in {tag, name, tag.split("/")[0]}


def _price_mtok(raw) -> float:
    return float(raw or 0.0) * 1_000_000.0


def preflight_role(
    settings: Settings,
    *,
    role: str,
    model: str,
    smoke: bool = True,
    http=None,
    client_factory=CappedOpenRouterClient,
) -> dict:
    """Validate one exact role route and optionally make one bounded smoke call."""
    http = http or requests
    url = f"{settings.base_url}/models/{model}/endpoints"
    response = http.get(
        url,
        headers={"Authorization": f"Bearer {settings.api_key}", "X-Title": "KiCraft-preflight"},
        timeout=30,
    )
    response.raise_for_status()
    data = response.json().get("data") or {}
    errors: list[str] = []
    if data.get("id") != model:
        errors.append(f"exact model id missing: requested {model!r}, got {data.get('id')!r}")

    endpoints = data.get("endpoints") or []
    selected = []
    for provider in settings.provider_order:
        matches = [ep for ep in endpoints if _provider_matches(ep, provider)]
        if not matches:
            errors.append(f"selected provider {provider!r} does not serve {model}")
        selected.extend(matches)
    if not settings.provider_order:
        errors.append("role route has no selected provider")
    if settings.max_price_prompt <= 0 or settings.max_price_completion <= 0:
        errors.append("role route must have finite positive prompt and completion price caps")

    required = _ROLE_PARAMETERS[role]
    endpoint_rows = []
    viable = []
    seen_tags = set()
    for endpoint in selected:
        tag = str(endpoint.get("tag") or "")
        if tag in seen_tags:
            continue
        seen_tags.add(tag)
        supported = set(endpoint.get("supported_parameters") or [])
        pricing = endpoint.get("pricing") or {}
        prompt_price = _price_mtok(pricing.get("prompt"))
        completion_price = _price_mtok(pricing.get("completion"))
        missing = sorted(required - supported)
        under_cap = (
            prompt_price <= settings.max_price_prompt
            and completion_price <= settings.max_price_completion
        )
        row = {
            "provider": endpoint.get("provider_name"),
            "tag": tag,
            "model_id": endpoint.get("model_id"),
            "prompt_price_mtok": prompt_price,
            "completion_price_mtok": completion_price,
            "supported_parameters": sorted(supported),
            "missing_parameters": missing,
            "streaming": True,
            "under_cap": under_cap,
        }
        endpoint_rows.append(row)
        if endpoint.get("model_id") != model:
            errors.append(f"provider {tag!r} serves {endpoint.get('model_id')!r}, not {model!r}")
        if not missing and under_cap and endpoint.get("model_id") == model:
            viable.append(row)
    if selected and not viable:
        errors.append("no selected endpoint satisfies capabilities and price caps")

    smoke_result = None
    if smoke and not errors:
        client = client_factory(settings)
        messages = [
            {"role": "system", "content": "Return the schema object only."},
            {"role": "user", "content": 'Return {"ok": true}.'},
        ]
        call_kwargs = {
            "model": model,
            "max_tokens": 256,
            "temperature": 0.0,
            "reasoning": {"enabled": False},
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "kicraft_preflight_v1",
                    "strict": True,
                    "schema": _SMOKE_SCHEMA,
                },
            },
            "meta_ctx": {"phase": "model_preflight", "role": role},
        }
        if role == "designer":
            result = client.chat_with_tools(
                messages,
                [
                    {
                        "type": "function",
                        "function": {
                            "name": "preflight_noop",
                            "description": "Capability probe; do not call.",
                            "parameters": {
                                "type": "object",
                                "properties": {},
                                "additionalProperties": False,
                            },
                        },
                    }
                ],
                lambda *_: "{}",
                max_rounds=1,
                **call_kwargs,
            )
        else:
            result = client.chat(messages, **call_kwargs)
        try:
            parsed = json.loads(result.get("text") or "")
        except json.JSONDecodeError:
            parsed = None
        smoke_result = {
            "ok": parsed == {"ok": True},
            "provider": result.get("provider"),
            "model": result.get("model") or model,
            "finish_reason": result.get("finish_reason"),
            "cost_usd": float(result.get("cost_usd") or 0.0),
            "reply_head": (result.get("text") or "")[:200],
        }
        if not smoke_result["ok"]:
            errors.append("bounded schema-response smoke call failed")

    return {
        "role": role,
        "profile": settings.design_profile,
        "model": model,
        "provider_order": list(settings.provider_order),
        "price_caps_mtok": {
            "prompt": settings.max_price_prompt,
            "completion": settings.max_price_completion,
        },
        "required_parameters": sorted(required),
        "endpoints": endpoint_rows,
        "smoke": smoke_result,
        "ok": not errors,
        "errors": errors,
    }


def _role_settings(base: Settings, target: str) -> list[tuple[str, Settings, str]]:
    rows: list[tuple[str, Settings, str]] = []
    profile_names = list(DESIGN_PROFILES) if target == "all" else [target]
    if target in DESIGN_PROFILES or target == "all":
        for name in profile_names:
            profile = DESIGN_PROFILES[name]
            settings = replace(
                base,
                model=str(profile["model"]),
                design_profile=name,
                provider_order=list(profile["provider_order"]),
                provider_allow_fallbacks=False,
                max_price_prompt=float(profile["max_price_prompt"]),
                max_price_completion=float(profile["max_price_completion"]),
            )
            rows.append((f"designer:{name}", settings, settings.model))
    if target in ("reviewer", "all"):
        settings = base.for_review()
        rows.append(("reviewer", settings, str(base.review_model)))
    if target in ("judge", "all"):
        settings = base.for_judge()
        rows.append(("judge", settings, str(base.eval_judge_model)))
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--role",
        choices=["all", "flash", "pro", "reviewer", "judge"],
        default="all",
    )
    parser.add_argument("--metadata-only", action="store_true", help="skip the paid smoke call")
    parser.add_argument("--out", help="artifact path; defaults to a dated JSON file")
    args = parser.parse_args(argv)

    base = Settings.from_env()
    results = []
    for label, settings, model in _role_settings(base, args.role):
        role = "designer" if label.startswith("designer:") else label
        try:
            result = preflight_role(
                settings,
                role=role,
                model=model,
                smoke=not args.metadata_only,
            )
        except Exception as exc:  # one role failure must not hide the others
            result = {
                "role": role,
                "profile": settings.design_profile,
                "model": model,
                "ok": False,
                "errors": [str(exc)],
            }
        result["label"] = label
        results.append(result)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "results": results,
        "ok": all(row.get("ok") for row in results),
    }
    out = Path(args.out or f"model_preflight_{stamp}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(f"wrote {out}", file=sys.stderr)
    return 0 if payload["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
