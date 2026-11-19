"""Deterministic build outcomes shared by the CLI and its consumers."""

# Zero retains its fabrication-ready meaning; this is a usable, unapproved draft.
REVIEW_REQUIRED_RC = 9
USB_REVIEW_WARNING = (
    "USB trace geometry was calculated from the nominal board stack-up; "
    "manufacturer confirmation is missing. Draft only, not fabrication approved."
)
