# KiCraft Terms of Service

**Version: 2026-06-11**

> **DRAFT, pending attorney review.** This document is a first-draft template. It
> has not been reviewed by a lawyer. Replace every `[BRACKETED]` placeholder and
> have qualified counsel review it before exposing the service to any external
> user. See `README.md` in this directory.

These Terms of Service ("Terms") govern your access to and use of KiCraft (the
"Service"), an AI-assisted printed-circuit-board design tool operated by
`[LEGAL ENTITY NAME]` ("KiCraft", "we", "us"). By creating an account or using
the Service, you agree to these Terms and to our [Privacy Policy](./privacy-policy.md).
If you do not agree, do not use the Service.

## 1. Eligibility

You must be at least 18 years old and able to form a binding contract. The
Service is an early-access product; access may be limited or invite-gated.

## 2. The Service, and the nature of its output

KiCraft turns a natural-language description into KiCad schematic and board
files using large language models and a deterministic synthesis pipeline.

**Outputs are AI-generated and are not guaranteed to be correct, complete,
manufacturable, or safe.** They are a starting point for a qualified engineer,
not a finished product. See the safety disclaimer in Section 9.

## 3. Accounts

You agree to provide an accurate email address, keep your password confidential,
and accept responsibility for activity under your account. One account is for one
person. We may suspend or terminate accounts that violate these Terms.

## 4. Acceptable use

You agree not to:

- upload content you do not have the rights to, including third-party
  confidential information or trade secrets;
- use the Service for unlawful, infringing, or harmful purposes;
- attempt to abuse, overload, reverse-engineer, or circumvent the metered
  gateway, usage quotas, or spend limits;
- design anything intended to cause harm or that is illegal to produce.

## 5. Your content and the license you grant us

**You retain ownership of the content you submit** (your prompts, briefs, the
answers you give during the design interview) and of the designs the Service
generates for you ("Your Content").

To operate and improve the Service, you grant KiCraft a worldwide, non-exclusive,
royalty-free, sublicensable license to host, store, copy, transmit, process,
analyze, and create derivative works from Your Content for the following
purposes:

a. **Operating the Service**, including saving your projects so you can find them
   again, and transmitting your prompts to our model providers to generate
   designs (see Section 8);
b. **Product analytics**: understanding how the Service is used so we can improve
   it;
c. **Training and improving machine-learning models**, including using Your
   Content as training data. You may opt out of this use at any time (see Section
   6 and the Privacy Policy);
d. **Publishing de-identified examples**: showing anonymized designs and prompts
   in galleries, documentation, or marketing, with identifying details removed.

This license is what allows the data collection and uses described in the Privacy
Policy. It survives only as needed to provide and improve the Service and, for
content already incorporated into a trained model or a published anonymized
example, it survives account deletion to the extent that content cannot
practically be withdrawn.

### Public projects and the community browser

**If you are on the free tier, the projects you create are public.** When a design
completes, it is added to the KiCraft community browser, where any other signed-in
user can view it, see its bill of materials, and **clone it** into their own
account to study or build upon. By using the free tier you grant other users a
non-exclusive, royalty-free license to view and clone your completed designs for
their own use, and you grant KiCraft the right to display them in the community
browser. **Do not put anything you need to keep confidential into a free-tier
project.**

**Paid tiers (Pro, Max) may keep projects private.** A private project is not
listed in the community browser and cannot be cloned by others. You choose whether
each completed project is public or private in your account settings.

Cloning copies the design files and starting state into the cloning user's account
so they can iterate on it; it does not transfer ownership of, or any rights in,
your original project beyond the view-and-clone license above. Likewise, when you
clone another user's public project, you receive only that same view-and-clone
license, not ownership of their work.

## 6. Your choices

You can opt out of the use of Your Content to train models (Section 5c) in your
account settings, and you can request export or deletion of your data as described
in the Privacy Policy. Opting out of training does not affect uses (a), (b), or
the publication of already de-identified examples. If you are on a paid tier, you
can also choose whether each completed project is public in the community browser
or kept private, in your account settings (see Section 5).

## 7. Paid plans, billing, cancellation, and refunds

Some features (higher design quotas, private projects) require a paid
subscription ("Pro", "Max"). Current prices and quotas are listed on the
pricing page of the Service.

a. **Billing.** Paid plans are billed as a recurring monthly subscription, in
   USD, in advance. Payment is processed by Stripe, Inc.; we never see or
   store your card number. By subscribing you authorize recurring charges to
   your payment method until you cancel.
b. **Renewal and cancellation.** Subscriptions renew automatically each month.
   You can cancel at any time from the billing portal in your profile.
   Cancellation stops future charges; your plan stays active until the end of
   the period you already paid for, after which your account moves to the free
   tier.
c. **Failed payments.** If a renewal charge fails, Stripe retries it. If
   payment is not completed within a short grace period, your account moves to
   the free tier until payment resumes.
d. **Refunds.** `[REFUND POLICY: e.g. "Fees are non-refundable except where
   required by law", or a goodwill window; counsel to advise.]`
e. **Price changes.** We may change prices. Changes take effect from your next
   billing period, and we will give you at least `[NOTICE PERIOD, e.g. 30
   days]` notice before a price increase applies to an existing subscription.
f. **Taxes.** Prices `[include / exclude]` applicable taxes; you are
   responsible for any taxes we are required to collect.
g. **Downgrades.** Moving to the free tier does not delete your projects.
   Projects you made private while on a paid plan stay private, but new
   free-tier projects are public (Section 5).

## 8. Third-party model providers

To generate designs, the Service sends your prompts and interview answers to
third-party AI providers (currently OpenRouter and the underlying model provider
it routes to, such as DeepSeek) for inference. **Your prompt content leaves our
servers when this happens** and is handled under those providers' terms. Do not
submit anything you are not comfortable sending to those providers.

## 9. No warranty, and your duty to verify before building

THE SERVICE AND ITS OUTPUT ARE PROVIDED "AS IS" AND "AS AVAILABLE", WITHOUT
WARRANTIES OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE, AND NON-INFRINGEMENT. The Service is alpha software and
may change, break, lose data, or be discontinued.

**You are solely responsible for independently reviewing, verifying, and testing
every design before you fabricate, assemble, power on, sell, or otherwise rely on
it.** A generated design may contain errors that are unsafe or that damage
equipment or property. Treat all output as unverified.

## 10. Limitation of liability

To the maximum extent permitted by law, KiCraft and its operators will not be
liable for any indirect, incidental, special, consequential, or punitive damages,
or for any loss of profits, data, or goodwill, arising from your use of the
Service or from any design it produces. Our total liability for any claim will not
exceed the greater of the amount you paid us in the 12 months before the claim or
`[USD 100]`.

## 11. Indemnification

You agree to indemnify and hold KiCraft harmless from claims arising out of Your
Content, your use of the Service, your violation of these Terms, or any design you
fabricate or distribute.

## 12. Termination

You may stop using the Service and delete your account at any time; deleting
your account cancels any active subscription. We may suspend or terminate
access for any violation of these Terms or to protect the Service. Sections
that by their nature should survive (5, 9, 10, 11, 13), and any payment
obligations already accrued under Section 7, survive termination.

## 13. Changes to these Terms

We may update these Terms. The version date appears at the top of this document.
For material changes we will require you to accept the updated Terms before you
continue using the Service.

## 14. Governing law

These Terms are governed by the laws of `[GOVERNING-LAW JURISDICTION]`, without
regard to conflict-of-law rules.

## 15. Contact

Questions about these Terms: `[CONTACT EMAIL]`.
