"""Shared signal allocation limits for the DSL compiler."""

# Factorio exposes 26 built-in virtual letter signals (signal-A through signal-Z).
# The compiler reuses those for implicit temporaries; exceeding this pool leads to
# ambiguous wiring inside Factorio, so we fail fast when the allocation budget is
# exhausted.
MAX_IMPLICIT_VIRTUAL_SIGNALS = 26
