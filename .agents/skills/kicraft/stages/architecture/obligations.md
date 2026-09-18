Architecture section: requirement ownership.

This call answers **one section** of the architecture intent — where each committed typed
obligation is implemented. The user's original intent and functional spec carry rows such as a
physical component class, an adjustability, a conversion behaviour or a numerical limit, and the
architecture must name the requirement that implements each one.

Answer exactly one field — `owners` — and do not emit `sheets`, `requirements` or `signals`.

Shape: `{"owners": [{"kind": "<the row's kind>", "original_obligation_id": "<the row's id>",
"requirement_ids": ["<requirement id>", …]}]}`.

Rules:

- Every committed obligation listed in the prompt MUST appear exactly once as an owner entry,
  except a row that names no implementation: a `quantity` row (it counts a class across the
  design), a `fabrication` row (a printed-board property the PCB side owns, not a part), and a
  `negative` row (it forbids a class). The architecture is refused when an obligation that *does*
  name an implementation is attached to no requirement. Leave the exempt rows out entirely.
- `kind` and `original_obligation_id` copy the committed row's own values — they are how the row is
  identified. **Do not restate, paraphrase or trim the row itself**: the compiler writes the
  committed intent/functional-spec row verbatim onto the requirements you name. An ownership answer
  carries no row content.
- `requirement_ids` names the requirement(s) that implement the obligation. Use the exact
  `requirements[].id` values already in the prompt. Several requirements may implement one
  obligation in different places (three binding posts, three identical axis drivers) — name every
  one that really implements it, and none that does not.
- A `quantity` obligation counts a class across the whole design (two binding posts, three stepper
  axes). It is not an implementation claim and needs no owner: leave it out.
- An obligation about a *component class* is owned by the requirement whose part really is that
  class; a requirement implementing something else is not an owner. Never name a requirement to
  silence the check, and never drop an obligation you cannot place — report it in the refusal
  instead, because a placed-but-wrong owner is the defect this call exists to prevent.
