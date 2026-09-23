"""Dependency-free clarification contract shared with the pinned native runner."""

QUESTION_OPTIONS_INSTRUCTION = (
    "Ask one decision per question, at most three questions. Supply 2–4 distinct, "
    "concise choices in options, with the recommended default first. Keep freeform "
    "answers possible. Never repeat an already answered question. If missing "
    "information is a physical fact rather than a preference, recommend obtaining "
    "that fact rather than inventing it."
)


def normalize_question_options(raw_options) -> list[str]:
    if not isinstance(raw_options, list) or any(not isinstance(o, str) for o in raw_options):
        raise ValueError("Clarification questions require at least two distinct options.")
    options = []
    for raw in raw_options:
        option = raw.strip()[:200]
        if option and option not in options:
            options.append(option)
        if len(options) == 4:
            break
    if len(options) < 2:
        raise ValueError("Clarification questions require at least two distinct options.")
    return options
