"""Explicit opt-in for paid providers in a local-only archived demo."""
import os


class PaidServicesDisabled(RuntimeError):
    pass


def require_paid_services():
    # Credentials alone must never enable spending or revive a hosted demo.
    if os.environ.get("VERCEL") or os.environ.get("TRIRIGHT_ALLOW_PAID_SERVICES") != "true":
        raise PaidServicesDisabled("Paid services are disabled for this archived demo")
