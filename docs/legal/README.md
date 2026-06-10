# KiCraft legal documents

Canonical source for the user-facing Terms of Service and Privacy Policy. The web
app serves these at `/terms` and `/privacy` (rendered from this directory; see
`kicraft/server/config.py` `legal_dir` and `web.py`).

## Current version

`LEGAL_VERSION = 2026-06-09`

This string is defined in `kicraft/server/config.py` and stamped into each user's
consent record (`accepted_terms_version`) at signup. Bumping the version date in
both the documents and `config.py` forces existing users to re-accept on their
next visit.

## Status: DRAFT, pending attorney review

These documents are **first-draft templates** written to unblock the consent gate,
not finished legal text. Before exposing the Service to any external user:

1. **Replace every `[BRACKETED]` placeholder**: legal entity name, controller name
   and address, governing-law jurisdiction, hosting provider/region, contact
   email, response-time and liability-cap numbers.
2. **Have qualified counsel review both documents.** Pay particular attention to:
   - the model-training license grant and the publication-of-anonymized-examples
     grant (Terms Section 5, Privacy Section 2);
   - the free-tier public-projects grant and the cross-user view/clone license
     (Terms Section 5 "Public projects and the community browser", Privacy Section
     2e): free users' completed designs are published to other signed-in users and
     are cloneable by default, and are not de-identified beyond hiding the email;
   - the manufacture / safety liability disclaimer (Terms Section 8): boards that
     get physically built create real product-liability exposure;
   - GDPR / CCPA obligations if any user may be in the EU, UK, or California
     (lawful basis per purpose, data-subject request handling, a data-processing
     agreement covering OpenRouter and the underlying model provider).

## How consent and data controls are wired

- Signup requires accepting the Terms + Privacy Policy (a required checkbox) and
  records `accepted_terms_version` + `accepted_terms_at` on the user row.
- A second checkbox (default on) records the model-training preference
  (`allow_training`); users can change it later in the app.
- `kicraft-accounts export <email>` and `kicraft-accounts delete <email>` provide
  the access/export and deletion paths the Privacy Policy promises.
- Any future training-corpus tooling must exclude users with `allow_training = 0`.
