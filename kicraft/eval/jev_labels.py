"""Ask Jev the ambiguous vocabulary cases, cache the answers, report agreement.

Step 3 of ``docs/plans/vocabulary-literalism-plan-2026-09-26.md``. The replay harness
(``check_replay``) reports which checks fire on boards that shipped; some of those firings are
genuine defects and some are wording. This module sends the genuinely ambiguous ones to Jev as
typed decisions (``noul`` / ``choice``) over the archived candidate, so the plan can measure a
model's reading against hand labels instead of guessing.

Two rules from the plan are enforced here:

* **Never over the live pipeline.** The tool reads archived ``state.json`` files and the
  ``--dump-hits`` corpus; it never calls ``diagnose_stage`` or touches a run in progress.
* **Cached as data.** Every answer is written under
  ``kicraft/eval/corpora/vocabulary-literalism/`` keyed by a hash of (model, question, state), so
  a re-run of the same candidate takes the same decision from the cache. The gate never calls
  this module.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from kicraft.eval.check_replay import STATE_RELATIVE
from kicraft.server.decision_layer import DEFAULT_MODEL, DecisionUnavailable, Question, decide

CORPUS_DIR = Path(__file__).parent / "corpora" / "vocabulary-literalism"

#: How sure Jev must be before a label counts. A label is evidence about the check, so a guess
#: is worse than a blank: below the floor the row is recorded as unlabelled, not coerced.
DEFAULT_CONFIDENCE = 0.7

_GROUND_RETURN_PROVIDED = "a ground return is provided elsewhere in the design"
_GROUND_RETURN_MISSING = "a ground return for that block is genuinely missing"
_NOT_ENOUGH = "the candidate does not say"

_ROLE_IS_REFERENCE = "the sheet only references that part, which is implemented on another sheet"
_ROLE_IS_CLAIM = "the sheet itself promises an active part the design does not build anywhere"

_COUNT_CLASS_MISSPELLED = "the design already carries that class under a different spelling"
_COUNT_CLASS_MISSING = "the count belongs to a part class the design is missing"
_COUNT_IS_PROPERTY = "the count is a property of one part, not a count of parts"


@dataclass(frozen=True)
class Hit:
    run: str
    stage: str
    code: str
    message: str
    brief: str = ""
    evidence: tuple[str, ...] = ()


@dataclass
class Label:
    key: str
    run: str
    stage: str
    code: str
    question: str
    value: Any
    confidence: float
    cached: bool
    confident: bool = False
    notes: Sequence[str] = field(default_factory=tuple)

    def as_dict(self) -> dict:
        row = asdict(self)
        row["notes"] = list(self.notes)
        return row


def iter_hits(path: Path) -> list[Hit]:
    hits: list[Hit] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        hits.append(
            Hit(
                run=str(row.get("run") or ""),
                stage=str(row.get("stage") or ""),
                code=str(row.get("code") or ""),
                message=str(row.get("message") or ""),
                brief=str(row.get("brief") or ""),
                evidence=tuple(str(item) for item in row.get("evidence") or ()),
            )
        )
    return hits


def _load_candidate(run_dir: Path) -> Mapping[str, Any] | None:
    for candidate in (run_dir / STATE_RELATIVE, run_dir / "state.json"):
        try:
            doc = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(doc, Mapping):
            return doc
    return None


def _state_for(hit: Hit) -> dict | None:
    """The state one decision call sees: the brief and the stage's own committed candidate."""
    doc = _load_candidate(Path(hit.run))
    if doc is None:
        return None
    candidate = doc.get(hit.stage)
    if not isinstance(candidate, Mapping):
        return None
    return {
        "brief": hit.brief,
        "stage": hit.stage,
        "candidate": candidate,
        "finding": {"code": hit.code, "message": hit.message, "evidence": list(hit.evidence)},
    }


def question_for(hit: Hit) -> Question | None:
    """The typed question this firing earns, or None when the check is not one we label."""
    if hit.code == "functional_spec_partial_ground_flow":
        missing = ", ".join(hit.evidence) or "some blocks"
        return Question(
            key="ground_return",
            prompt=(
                "This functional specification lists ground flows for some functional blocks but "
                f"not for: {missing}. Read the spec's own connections and assumptions. Does the "
                "design still provide a ground return for each of those blocks (for example "
                "through a shared ground net the connections do not enumerate per block), or is "
                "a ground return genuinely missing?"
            ),
            kind="choice",
            options=(_GROUND_RETURN_PROVIDED, _GROUND_RETURN_MISSING, _NOT_ENOUGH),
            descriptions={
                _GROUND_RETURN_PROVIDED: "the block is grounded; the list is only partial",
                _GROUND_RETURN_MISSING: "the block has no return path: a real defect",
                _NOT_ENOUGH: "the candidate does not settle it",
            },
        )
    if hit.code == "bom_architecture_role_unsupported":
        sheets = ", ".join(hit.evidence) or "a sheet"
        return Question(
            key="role_reference",
            prompt=(
                f"Sheet(s) {sheets} do not carry a U-reference, and a word in the architecture "
                "text matches an active-part role. Read each sheet's own title and its typed "
                "requirements against the whole BOM. Is the matched role a part the design "
                "implements on another sheet (so this sheet only references it), or does the "
                "sheet itself promise an active part that no sheet implements?"
            ),
            kind="choice",
            options=(_ROLE_IS_REFERENCE, _ROLE_IS_CLAIM, _NOT_ENOUGH),
            descriptions={
                _ROLE_IS_REFERENCE: "the prose names a counterpart part that exists elsewhere",
                _ROLE_IS_CLAIM: "an active part is promised and never built: a real defect",
                _NOT_ENOUGH: "the candidate does not settle it",
            },
        )
    if hit.code == "intent_quantity_subject_unbound":
        return Question(
            key="count_subject",
            prompt=(
                "One count row names a subject that matches no part class this design carries: "
                f"{', '.join(hit.evidence) or 'see the finding'}. Read the brief and the intent's "
                "own obligations. Does the count belong to a part class the design already carries "
                "under a different spelling, to a class the design is missing, or is it a property "
                "of a single part (for example pins on one header)?"
            ),
            kind="choice",
            options=(
                _COUNT_CLASS_MISSPELLED,
                _COUNT_CLASS_MISSING,
                _COUNT_IS_PROPERTY,
                _NOT_ENOUGH,
            ),
            descriptions={
                _COUNT_CLASS_MISSPELLED: "the obligation exists; the names only need to be related",
                _COUNT_CLASS_MISSING: "the intent should add the obligation it counts",
                _COUNT_IS_PROPERTY: "the row is a property, not a count of parts",
                _NOT_ENOUGH: "the candidate does not settle it",
            },
        )
    return None


def _cache_key(model: str, question: Question, state: Mapping[str, Any]) -> str:
    payload = json.dumps(
        {"model": model, "question": question.payload(), "state": state},
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def _cache_path(cache_dir: Path, key: str) -> Path:
    return cache_dir / f"{key}.json"


def label_hits(
    hits: Sequence[Hit],
    *,
    model: str = DEFAULT_MODEL,
    confidence: float = DEFAULT_CONFIDENCE,
    cache_dir: Path = CORPUS_DIR,
    limit: int | None = None,
    refresh: bool = False,
) -> list[Label]:
    """One Jev call per (run, stage); every answer cached content-addressed."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    batches: dict[tuple[str, str], list[Hit]] = {}
    for hit in hits:
        if question_for(hit) is not None:
            batches.setdefault((hit.run, hit.stage), []).append(hit)
    ordered = sorted(batches.items())
    if limit is not None:
        ordered = ordered[:limit]

    labels: list[Label] = []
    for (run, stage), grouped in ordered:
        state = _state_for(grouped[0])
        if state is None:
            continue
        questions: dict[str, Question] = {}
        for hit in grouped:
            question = question_for(hit)
            if question is not None:
                questions.setdefault(hit.code, question)

        pending: list[tuple[str, Question, str]] = []
        for code, question in sorted(questions.items()):
            key = _cache_key(model, question, state)
            path = _cache_path(cache_dir, key)
            if path.exists() and not refresh:
                record = json.loads(path.read_text(encoding="utf-8"))
                labels.append(
                    _label_from_record(run, stage, code, question, record, cached=True, floor=confidence)
                )
                continue
            pending.append((code, question, key))
        if not pending:
            continue

        try:
            answers = {
                answer.key: answer
                for answer in decide(state, [question for _code, question, _key in pending], model=model)
            }
        except DecisionUnavailable as exc:
            for code, question, _key in pending:
                labels.append(
                    Label(question.key, run, stage, code, question.prompt, None, 0.0, False,
                          (f"unavailable: {exc}",))
                )
            continue

        for code, question, key in pending:
            answer = answers.get(question.key)
            if answer is None:
                labels.append(
                    Label(question.key, run, stage, code, question.prompt, None, 0.0, False,
                          ("no answer returned",))
                )
                continue
            record = {
                "run": run,
                "stage": stage,
                "code": code,
                "question": question.prompt,
                "kind": question.kind,
                "options": list(question.options),
                "value": answer.value,
                "confidence": answer.confidence,
                "probabilities": dict(answer.probabilities),
                "model": model,
            }
            _cache_path(cache_dir, key).write_text(
                json.dumps(record, indent=2, default=str) + "\n", encoding="utf-8"
            )
            labels.append(
                _label_from_record(run, stage, code, question, record, cached=False, floor=confidence)
            )
    return labels


def _label_from_record(
    run: str, stage: str, code: str, question: Question, record: Mapping[str, Any], *,
    cached: bool, floor: float,
) -> Label:
    """Keep every answer; flag the ones a Tier-B gate would refuse to act on.

    The measurement wants the model's reading and how sure it was, so a below-floor answer is
    data with a flag rather than a blank -- only a gate application would drop it.
    """
    sure = float(record.get("confidence") or 0.0)
    confident = record.get("value") is not None and sure >= floor
    notes = () if confident else (f"below confidence floor {floor} ({sure:.2f})",)
    return Label(
        key=question.key,
        run=run,
        stage=stage,
        code=code,
        question=question.prompt,
        value=record.get("value"),
        confidence=sure,
        cached=cached,
        confident=confident,
        notes=notes,
    )


def iter_hand_labels(path: Path) -> dict[tuple[str, str, str], dict]:
    """Hand labels keyed by (run, stage, code); absent file means no adjudication yet."""
    if not path.exists():
        return {}
    rows: dict[tuple[str, str, str], dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rows[(str(row["run"]), str(row["stage"]), str(row["code"]))] = row
    return rows


def agreement(labels: Iterable[Label], hand: Mapping[tuple[str, str, str], dict]) -> dict:
    """Where a hand label exists, does the model's answer agree?"""
    checked = matched = confident_checked = confident_matched = 0
    rows: list[dict] = []
    for label in labels:
        truth = hand.get((label.run, label.stage, label.code))
        if truth is None or label.value is None or "label" not in truth:
            continue
        checked += 1
        agree = str(truth["label"]).strip().casefold() == str(label.value).strip().casefold()
        matched += int(agree)
        if label.confident:
            confident_checked += 1
            confident_matched += int(agree)
        rows.append(
            {
                "run": label.run,
                "stage": label.stage,
                "code": label.code,
                "hand": truth["label"],
                "jev": label.value,
                "confidence": round(label.confidence, 3),
                "confident": label.confident,
                "agree": agree,
            }
        )
    return {
        "checked": checked,
        "agreed": matched,
        "rate": round(matched / checked, 3) if checked else None,
        "checked_confident": confident_checked,
        "agreed_confident": confident_matched,
        "rate_confident": (
            round(confident_matched / confident_checked, 3) if confident_checked else None
        ),
        "rows": rows,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--hits", required=True, help="JSONL from check_replay --dump-hits")
    parser.add_argument("--codes", default="", help="comma-separated diagnostic codes to label")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--confidence", type=float, default=DEFAULT_CONFIDENCE)
    parser.add_argument("--cache-dir", default=str(CORPUS_DIR))
    parser.add_argument("--limit", type=int, default=None, help="max (run, stage) calls")
    parser.add_argument("--refresh", action="store_true", help="ignore the cache")
    parser.add_argument("--out", default=None, help="write labels JSONL here")
    parser.add_argument("--hand", default=str(CORPUS_DIR / "hand_labels.jsonl"))
    args = parser.parse_args(argv)

    hits = iter_hits(Path(args.hits))
    if args.codes:
        wanted = {code.strip() for code in args.codes.split(",") if code.strip()}
        hits = [hit for hit in hits if hit.code in wanted]
    labels = label_hits(
        hits,
        model=args.model,
        confidence=args.confidence,
        cache_dir=Path(args.cache_dir),
        limit=args.limit,
        refresh=args.refresh,
    )

    report = agreement(labels, iter_hand_labels(Path(args.hand)))
    for label in labels:
        state = "cache" if label.cached else "live "
        value = "<unlabelled>" if label.value is None else str(label.value)
        print(f"{state} {label.stage:16.16} {label.code:44.44} {label.confidence:.2f} {value[:60]}")
    print()
    print(
        f"labels: {len(labels)} (confident: {sum(1 for label in labels if label.confident)})  "
        f"hand-checked: {report['checked']}  agreement: {report['agreed']}/{report['checked']}"
        + (f" ({report['rate']:.0%})" if report["rate"] is not None else "")
        + "  "
        f"[above-floor: {report['agreed_confident']}/{report['checked_confident']}]"
    )
    if args.out:
        Path(args.out).write_text(
            "".join(json.dumps(label.as_dict(), separators=(",", ":")) + "\n" for label in labels),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":  # pragma: no cover - module entry point
    raise SystemExit(main())
