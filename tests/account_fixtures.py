"""Historical project rows for tests that do not exercise design admission.

Keep this outside the shipped package: production callers must use the atomic
``AccountStore.create_project`` verification/quota admission API.
"""
from __future__ import annotations

from kicraft.server.accounts import AccountStore


def seed_project(
    store: AccountStore,
    user_id: int,
    brief: str,
    *,
    is_public: bool | None = None,
    auto_default_questions: bool = True,
) -> int:
    owner = store.get_user(user_id)
    if owner is None:
        raise ValueError("historical project fixture requires an existing owner")
    if is_public is None:
        is_public = owner.tier not in ("pro", "max")
    with store._conn() as conn:
        return store._insert_project(conn, user_id, brief, is_public, auto_default_questions)
