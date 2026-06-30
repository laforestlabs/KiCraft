"""Guards for the landing-page onboarding content (kicraft.server.examples).

``EXAMPLE_PROMPTS`` drives the animated placeholder. The "Surprise me" button no
longer draws from this list — it streams the vetted self-eval corpus
(``kicraft.tuning.benchmark``) in order, so those briefs are guarded here too.
"""
from __future__ import annotations

from kicraft.server.examples import EXAMPLE_PROMPTS
from kicraft.tuning.benchmark import briefs as selfeval_briefs


def test_example_prompts_nonempty_strings():
    assert EXAMPLE_PROMPTS, "EXAMPLE_PROMPTS must not be empty"
    for p in EXAMPLE_PROMPTS:
        assert isinstance(p, str) and p.strip(), f"bad example prompt: {p!r}"


def test_surprise_me_corpus_nonempty_strings():
    """The "Surprise me" button cycles these and runs them, so an empty/malformed
    entry would launch a blank or broken design."""
    briefs = selfeval_briefs()
    assert briefs, "self-eval corpus must not be empty"
    for b in briefs:
        assert isinstance(b, str) and b.strip(), f"bad self-eval brief: {b!r}"
