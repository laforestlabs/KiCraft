"""NiceGUI page for the KiCraft upstream chat pipeline.

Two-pane layout:
- Left: chat transcript + input row.
- Right: state panel with the expert-mode toggle. Off renders the most
  recent stage outputs in prose; on shows the structured ConversationState
  as JSON.

Session state lives in `app.storage.client["upstream_state"]` — per
browser tab, lost on reload. Cross-session persistence is out of scope
per the brief; the CLI's `--save state.json` is the durable path.
"""
from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from nicegui import app, run, ui

from kicraft.upstream.models import ConversationState
from kicraft.upstream.orchestrator import run_turn
from kicraft.upstream.stages.synthesis import SynthesisInputError, run as run_synth
from kicraft.upstream.synthesis.validation import SynthesisValidationError


logger = logging.getLogger(__name__)

_STORAGE_KEY = "upstream_state"


def _load_state() -> ConversationState:
    raw = app.storage.client.get(_STORAGE_KEY)
    if raw is None:
        return ConversationState()
    if isinstance(raw, str):
        raw = json.loads(raw)
    try:
        return ConversationState.model_validate(raw)
    except Exception:  # noqa: BLE001
        logger.exception("could not restore upstream chat state; starting fresh")
        return ConversationState()


def _save_state(state: ConversationState) -> None:
    app.storage.client[_STORAGE_KEY] = state.model_dump(mode="json")


def upstream_chat_page() -> None:
    """Render the upstream chat tab."""
    state = _load_state()

    with ui.row().classes("w-full no-wrap gap-4"):
        # ---- LEFT: chat ----
        with ui.column().classes("col-grow"):
            ui.label("KiCraft Upstream Chat").classes("text-h6")
            ui.label(
                "Describe the project you want to build. The pipeline will "
                "capture intent, decompose into blocks, commit to topologies, "
                "build a BOM, and synthesize the KiCad file set."
            ).classes("text-caption text-grey-7")

            transcript = ui.scroll_area().classes("w-full bg-grey-1 rounded-borders").style(
                "height: 540px; padding: 12px"
            )

            def _render_transcript() -> None:
                transcript.clear()
                with transcript:
                    for msg in state.history:
                        sent = msg.role == "user"
                        ui.chat_message(
                            text=msg.content,
                            sent=sent,
                            name="you" if sent else "kicraft",
                        )
                transcript.scroll_to(percent=1.0)

            _render_transcript()

            input_box: dict[str, Any] = {"el": None}

            async def _send() -> None:
                msg = (input_box["el"].value or "").strip()
                if not msg:
                    return
                input_box["el"].value = ""
                input_box["el"].update()
                # Optimistic echo so the user sees their message immediately.
                state.history.append(state.history[-1].model_copy() if False else _user_chat_msg(msg))
                _render_transcript()

                send_btn.disable()
                send_btn.props("loading")
                try:
                    # Run the LLM call in a worker to avoid blocking the event loop.
                    updated = await run.io_bound(_run_turn_blocking, state, msg)
                    state.history[:] = updated.history
                    state.intent = updated.intent
                    state.functional_spec = updated.functional_spec
                    state.architecture = updated.architecture
                    state.bom = updated.bom
                    state.open_questions = updated.open_questions
                    state.project_stem = updated.project_stem
                except Exception as e:  # noqa: BLE001
                    state.history.append(
                        _asst_chat_msg(f"error: {type(e).__name__}: {e}")
                    )
                finally:
                    send_btn.props(remove="loading")
                    send_btn.enable()
                    _save_state(state)
                    _render_transcript()
                    _render_state_panel()

            with ui.row().classes("w-full no-wrap items-center"):
                input_box["el"] = ui.input(placeholder="Describe your project...").classes(
                    "col-grow"
                )
                input_box["el"].on("keydown.enter", _send)
                send_btn = ui.button("Send", icon="send", on_click=_send)

        # ---- RIGHT: state panel ----
        with ui.column().classes("col-4 q-pa-sm"):
            with ui.row().classes("w-full items-center justify-between"):
                ui.label("State").classes("text-subtitle1")
                expert_switch = ui.switch("Expert mode", value=state.expert_mode)

            state_area = ui.scroll_area().classes("w-full bg-grey-2 rounded-borders").style(
                "height: 460px; padding: 8px"
            )

            def _render_state_panel() -> None:
                state_area.clear()
                with state_area:
                    if state.expert_mode:
                        ui.code(
                            json.dumps(state.model_dump(mode="json"), indent=2),
                            language="json",
                        ).classes("w-full")
                    else:
                        _render_state_prose(state)

            def _toggle_expert(_) -> None:
                state.expert_mode = bool(expert_switch.value)
                _save_state(state)
                _render_state_panel()

            expert_switch.on("update:model-value", _toggle_expert)

            _render_state_panel()

            ui.separator().classes("q-my-md")
            ui.label("Synthesize").classes("text-subtitle1")
            project_dir_input = ui.input(
                label="Project directory",
                value=str(Path.cwd() / "generated"),
            ).classes("w-full")

            synth_status = ui.label("").classes("text-caption text-grey-7")

            async def _synthesize() -> None:
                synth_status.text = "running..."
                synth_btn.props("loading")
                try:
                    target = Path(project_dir_input.value)
                    if state.project_stem:
                        target = target / state.project_stem
                    artifacts, results = await run.io_bound(
                        _synth_blocking, state, target
                    )
                    ok = all(r.ok for r in results)
                    if ok:
                        synth_status.text = f"wrote {artifacts.project_dir}"
                        ui.notify(
                            f"Synthesized {artifacts.project_stem}",
                            type="positive",
                            position="top",
                        )
                    else:
                        bad = [r.name for r in results if not r.ok]
                        synth_status.text = "failed: " + ", ".join(bad)
                        ui.notify(
                            "Synthesis failed; see status",
                            type="negative",
                            position="top",
                        )
                except SynthesisInputError as e:
                    synth_status.text = f"input error: {e}"
                    ui.notify(str(e), type="warning", position="top")
                except SynthesisValidationError as e:
                    synth_status.text = "validation failed"
                    ui.notify(str(e), type="negative", position="top", multi_line=True)
                except Exception as e:  # noqa: BLE001
                    synth_status.text = f"error: {type(e).__name__}: {e}"
                    ui.notify(str(e), type="negative", position="top")
                finally:
                    synth_btn.props(remove="loading")

            synth_btn = ui.button(
                "Synthesize",
                icon="construction",
                on_click=_synthesize,
            ).classes("w-full")


# ---------- helpers ----------


def _user_chat_msg(text: str):
    from kicraft.upstream.models import ChatMsg

    return ChatMsg(role="user", content=text)


def _asst_chat_msg(text: str):
    from kicraft.upstream.models import ChatMsg

    return ChatMsg(role="assistant", content=text)


def _run_turn_blocking(state: ConversationState, user_message: str) -> ConversationState:
    """Synchronous wrapper so we can call run_turn from `run.io_bound`."""
    # The user message was already appended by the caller; pop it so
    # `run_turn` adds it cleanly (it always appends, so duplicates would
    # accumulate otherwise).
    if state.history and state.history[-1].role == "user" and state.history[-1].content == user_message:
        state.history.pop()
    return run_turn(state, user_message)


def _synth_blocking(state: ConversationState, project_dir: Path):
    return run_synth(state, project_dir)


def _render_state_prose(state: ConversationState) -> None:
    """Render a human-readable summary of populated slots."""
    if state.project_stem:
        ui.markdown(f"**Project**: `{state.project_stem}`")

    if state.intent:
        ui.markdown("### Intent")
        ui.markdown(state.intent.goal)
        if state.intent.constraints:
            ui.markdown("**Constraints**: " + "; ".join(state.intent.constraints))
        if state.intent.named_parts:
            ui.markdown("**Named parts**: " + ", ".join(state.intent.named_parts))
        if state.intent.assumptions:
            with ui.expansion("Assumptions").classes("w-full"):
                for a in state.intent.assumptions:
                    ui.markdown(f"- {a}")
    else:
        ui.label("(intent not yet captured)").classes("text-italic text-grey")

    if state.functional_spec:
        ui.markdown("### Functional spec")
        for b in state.functional_spec.blocks:
            ui.markdown(f"- **{b.name}** ({b.category}): {b.purpose}")
    else:
        ui.label("(no functional spec)").classes("text-italic text-grey")

    if state.architecture:
        ui.markdown("### Architecture")
        ui.markdown(
            f"Sheets: {', '.join(s.name for s in state.architecture.sheets)}"
        )
        ui.markdown(f"Power nets: {', '.join(state.architecture.power_nets)}")
        ui.markdown(
            f"Inter-sheet nets: {', '.join(n.name for n in state.architecture.inter_sheet_nets)}"
        )

    if state.bom:
        ui.markdown(f"### BOM ({len(state.bom.parts)} parts)")
        with ui.expansion("Parts list").classes("w-full"):
            for p in state.bom.parts:
                ui.markdown(f"- `{p.ref}` {p.value} @ `{p.footprint}` ({p.sheet})")

    if state.open_questions:
        ui.markdown("### Open questions")
        for q in state.open_questions:
            badge = "blocking" if q.blocking else ("material" if q.material else "default applied")
            ui.markdown(f"- [{badge}] {q.text}")
