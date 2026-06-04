# KiCraft backlog: alpha-launch features

## What this is

A holding list for substantial features deferred until there is time to do them
properly. The theme is taking the hosted app (kicraft.io) from a single-password
MVP to something a real cohort of alpha testers can use, while capturing
everything valuable from that testing. These are not quick wins; each deserves
focused design time.

## Where we are today (the starting point)

So future work starts from facts, not guesses:

- The hosted app `kicraft/server/web.py` is a NiceGUI app behind ONE shared
  password (`KICRAFT_ACCESS_PASSWORD`). There is no concept of an individual user.
- Design runs happen in ephemeral per-session tempdir workspaces
  (`tempfile.mkdtemp(prefix="kicraft_web_")`), hidden by `PrivateTmp=true` and not
  durably stored. When a run ends, nothing of the user's input survives except
  whatever they chose to download.
- The agent loop runs through the capped OpenRouter gateway (`SpendGuard` ledger
  at `~/.kicraft/spend_ledger.db`, `CappedOpenRouterClient`), default model
  deepseek-v4-flash, roughly $0.02 of tokens per full design. The
  synth/place/route/fab pipeline is deterministic and spends zero tokens.
- We already measure cost: `token-report` and the eval harness
  `metrics.token_usage`.
- There is a free `/demo` replay (`KICRAFT_WEB_DEMO=1`).

References: memory `project-kicraft-commercialization`; full plan at
`~/.claude/plans/make-a-plan-to-woolly-bee.md`.

## 1. Accounts, project persistence, and input capture

**Goal (user's words):** track users by login and email, associate each account
with its projects, and save 100% of user input, which is very valuable for
guiding the product.

This is the foundation the other two items lean on. Three layers:

- **Identity / auth.** Replace the single shared password with real per-user
  accounts keyed on email + login. Start simple (email + password, or magic-link)
  over a `users` table; SQLite to begin with (the spend ledger already sets that
  precedent), Postgres when it matters. NiceGUI's `app.storage.user`
  (cookie-backed) can hold the session; the durable identity lives in the DB.
- **Project association.** Persist each design run against a `user_id`: the
  committed `state.json` slots, the synthesized artifacts (the zip we already
  build), the model used, the cost, the timestamps. Today these die with the
  tempdir. Move from ephemeral tempdirs to a durable per-user store (filesystem
  under a per-user dir plus DB rows; object storage later).
- **Input capture (the high-value one).** Log every byte of user input
  append-only: each typed message, every clarification in the design interview,
  the full event stream the client already emits (`reasoning_delta` /
  `answer_delta` / `tool` / `tool_result`), and the final state. This is the
  product-guidance and (eventually) training corpus. The streaming infra already
  produces these events; this is about durably persisting them keyed to user +
  run, not re-instrumenting.

**Flag before building:** real accounts plus saved input means storing PII
(emails) and potentially proprietary circuit designs from alpha testers. That
needs a consent step and a privacy policy before the first external tester signs
in, not after. "Save 100% of user input" is a promise you have to be able to
honor and disclose.

**What done looks like:** an alpha tester signs in with their email, runs a
design, comes back next week and finds it, and on our side every keystroke of
that session is durably stored and queryable.

## 2. Onboarding and capability showcase

**Goal (user's words):** show new users how to operate the app, how to get the
most out of it, and what KiCraft is capable of.

New users land on a blank chat box with no idea that this is a back-and-forth
design interview, not a one-shot prompt. Onboarding has to teach the interaction
model and sell the capability at the same time.

- **Lead with the demo.** The `/demo` replay already exists, is free, and is
  browser-light. Promote it to the front door: "watch a real board get designed"
  before the user types anything. It doubles as proof of capability and a tutorial
  on the interaction model.
- **Example gallery + prompt starters.** Ship a few showcase designs (the USB-C
  night light from the test suite is a ready seed) with the exact prompts that
  produced them, so a newcomer can copy a known-good prompt and see the funnel end
  to end. Solves the blank-box problem.
- **Teach the conversation.** Inline hints that the stages ask clarifying
  questions, that a richer first prompt yields a better board, and what a good
  prompt contains. Make the interview model obvious rather than something users
  discover by accident.
- **Frame the value and the limits.** A landing page that explains chat to KiCad
  files, walks the pipeline (intent, functional_spec, architecture, bom, wiring,
  synth, place, route, fab), and is honest about current limits (board
  complexity, MCU coverage) so testers arrive with calibrated expectations.

**What done looks like:** a first-time visitor understands within a minute what
KiCraft does, has seen it work, and has a one-click way to try a known-good
example without staring at an empty box.

## 3. Billing and monetization

**Goal (user's open question):** how should monetization work? What
infrastructure, what is a fair markup on tokens? BYO API key plus a flat monthly
fee, a percent markup on tokens purchased through my key, or both?

Cost structure first, because it drives everything:

- **Variable cost** is LLM tokens per design: roughly $0.02 on deepseek-v4-flash,
  more on stronger models. The deterministic pipeline (synth/place/route/fab) is
  free.
- **Fixed cost** is the box, the domain, and the KiCad worker compute: tens of
  dollars a month, flat.
- We can already measure the variable cost per run, and the skill-eval rubric
  plus token tracking together answer "the cheapest model that still passes,"
  which sets the COGS floor and protects margin.

The three models you listed, with the trade-offs:

- **(a) BYO API key + flat monthly fee.** User brings their own key; KiCraft never
  pays for tokens; you charge for the software and hosting. Predictable margin,
  zero token-cost risk, no per-token billing infra. But it adds real friction
  (many hardware users will not have an OpenRouter or Anthropic key), forfeits the
  token-markup upside, and makes you hold or proxy user keys (a security surface).
  Best as a power-user / pro lane.
- **(b) Percent markup on tokens through your key.** Zero friction (no key
  needed), revenue scales with usage, and you already have the metering primitives
  (SpendGuard ledger, per-call real cost, token-report). But you carry token-cost
  risk and cashflow (you pay OpenRouter first), need per-user spend caps
  (SpendGuard covers this) against abuse, and the absolute dollars are thin: at
  $0.02 a design, even a 100% markup is $0.04. Markup-only is not a business at
  these volumes; it matters only at scale or on expensive models.
- **(c) Both (hybrid).** Matches the already-chosen direction (hybrid free to
  paid). A free, capped, metered tier on KiCraft's key removes all first-touch
  friction and feeds the input-capture goal from item 1; paid tiers layer on top.

**Recommendation (your call, but this is where I would start):**

- **Do not put raw token markup in the headline.** Tokens are too cheap for
  percent-markup to mean anything, and "you spent 14,212 tokens" is a terrible
  thing to show a customer. Price on value, not on token math.
- **Sell a simple unit: designs (or design credits).** A free tier of a few
  designs a month (funnel plus data capture), and a flat monthly paid tier with a
  generous design allowance on KiCraft's key, priced against the labor it replaces
  (a finished board is worth far more than its $0.02 of tokens), not against COGS.
- **Offer BYO-key as the "unlimited / pro" lane** at a smaller flat fee for
  technical users who would rather pay their own token cost for no cap. That
  captures power users without you carrying their variable cost.
- **Keep SpendGuard caps on every tier** as abuse protection regardless of model.

**Genuinely open (needs alpha data to answer well):** exact price points, how many
free designs, credits versus unlimited-with-fair-use, and which model is the
cheapest that still passes the rubric at the quality testers accept. Item 1's
input capture plus the existing token tracking are exactly what generate the data
to settle these. That is the closing loop: instrument the alpha, then price from
what you learn.

**Infra to add when this is real:** Stripe (subscriptions and/or metered
billing), a credits/balance ledger if you go the credit route (the spend ledger
is a starting pattern), and tier checks wired into the gateway you already have.

## How these connect

These are one theme, not three errands. Item 2 (onboarding) drives the funnel.
Item 1 (accounts plus 100% input capture) measures and retains what comes through
it. Item 3 (billing) monetizes it, and is best decided from the data item 1
collects. Sequencing when the time comes: accounts plus capture first (nothing
else is safe to run with external testers without it), onboarding alongside,
billing last, once there is usage data to price against.
