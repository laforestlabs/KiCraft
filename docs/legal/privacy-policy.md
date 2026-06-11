# KiCraft Privacy Policy

**Version: 2026-06-11**

> **DRAFT, pending attorney review.** This document is a first-draft template. It
> has not been reviewed by a lawyer. Replace every `[BRACKETED]` placeholder and
> have qualified counsel review it before exposing the service to any external
> user, especially if any user may be in the EU, UK, or California (GDPR / CCPA
> add specific obligations). See `README.md` in this directory.

This Privacy Policy explains what KiCraft (`[LEGAL ENTITY NAME]`, "we", "us")
collects, why, and what choices you have. It works together with the
[Terms of Service](./terms-of-service.md). The data controller is
`[CONTROLLER NAME AND ADDRESS]`.

## 1. What we collect

When you use the hosted Service we collect and store:

- **Account data**: your email address, a salted scrypt hash of your password
  (never the password itself), your plan tier, and account/login timestamps.
- **Your prompts and design interview**: the brief you type and every answer you
  give during the back-and-forth design session. We store the **full event
  stream** of each session (your messages, the model's reasoning and answers, and
  the tool calls made on your behalf). This is durably saved per project.
- **Generated designs**: the KiCad schematic and board files the Service produces
  for you, and the project archive.
- **Usage and cost metering**: which design stages ran, token counts, and the
  computed cost of each run.
- **Billing data (paid plans)**: if you subscribe to a paid plan, our payment
  processor Stripe collects your card details directly; **we never see or store
  your card number**. We store only your Stripe customer and subscription
  identifiers and the subscription's status, so we can match payments to your
  account.

We store these per account so that you can find your past projects, and we
associate them with your `user_id`.

## 2. How we use it

We use the data above to:

a. **Operate the Service**: run the design pipeline and let you return to your
   saved projects;
b. **Improve the product**: analyze prompts and usage patterns to understand what
   works and what to build next;
c. **Train and improve machine-learning models**, using your prompts and designs
   as training data. **You can opt out of this use** (Section 5);
d. **Publish de-identified examples**: show anonymized designs and prompts, with
   identifying details removed, in galleries, documentation, or marketing;
e. **Community sharing (free tier)**: if you are on the free tier, your completed
   designs are published to the in-app community browser, where other signed-in
   users can view them and clone them into their own accounts. Unlike (d), these
   are **not** de-identified beyond your email being hidden from non-staff viewers.
   Paid tiers can keep projects private. See the Terms of Service, Section 5.

## 3. Who we share it with

- **AI model providers.** To generate designs we transmit your prompts and
  interview answers to third-party inference providers (currently OpenRouter and
  the model it routes to, such as DeepSeek). Your prompt content leaves our
  servers and is processed under those providers' terms. We do not control their
  retention.
- **Infrastructure providers.** Hosting and storage for the Service
  (`[HOSTING PROVIDER]`).
- **Payment processor.** Paid-plan payments are handled by Stripe, Inc., which
  receives your email and payment details and processes them under
  [Stripe's privacy policy](https://stripe.com/privacy).
- We do **not** sell your personal information.
- We may disclose data if required by law.

## 4. Where it is stored and for how long

Account records and per-project data (prompts, the full event stream, generated
designs) are stored on `[HOSTING / REGION]`. We retain them for as long as your
account is active and as needed to provide and improve the Service. When you
request deletion (Section 5), we remove your account record and your stored
project data. Content already incorporated into a trained model or a published
anonymized example may not be practically removable; we will stop using your
identifiable data for those purposes going forward.

## 5. Your choices and rights

- **Training opt-out.** You can turn off the use of your content to train models
  at any time in your account settings. Existing analytics and operation of the
  Service are not affected.
- **Access and export.** You can request a copy of your account data and stored
  projects.
- **Deletion.** You can request deletion of your account and stored project data.
- To exercise these, use the in-app controls where available or contact
  `[CONTACT EMAIL]`. We action requests within `[N]` days.

Depending on where you live, you may have additional rights (for example under
GDPR or CCPA), including the right to object to processing, to data portability,
and to lodge a complaint with a supervisory authority. `[Counsel to confirm the
lawful basis for each processing purpose and add jurisdiction-specific language.]`

## 6. Security

Passwords are stored only as salted scrypt hashes. Access to stored data is
limited to operators of the Service. The Service is alpha software; while we take
reasonable measures, no system is perfectly secure, so do not submit content you
cannot afford to have exposed.

## 7. A note on proprietary designs

Because we store your full input and may use it as described above, **do not
upload circuit designs or other intellectual property you are not permitted to
share or that you need to keep secret.** Prompts are also sent to third-party
model providers (Section 3). This matters most on the **free tier**, where your
completed projects are public in the community browser and can be cloned by other
users (Section 2e); never put confidential or proprietary content into a free-tier
project. Upgrade to a paid tier if you need projects to stay private.

## 8. Children

The Service is not directed to anyone under 18, and we do not knowingly collect
data from them.

## 9. Changes

We may update this Policy. The version date appears at the top. For material
changes we will ask you to review and accept the update before continuing.

## 10. Contact

Privacy questions or data requests: `[CONTACT EMAIL]`.
