"""Vouchers: a fixed set of codes, each worth a fixed number of questions.

The codes are derived from a secret rather than stored, so the server can tell
a real code from a made-up one without a database and without the codes ever
being committed. The same secret reproduces the same hundred codes, which is
how vouchers.txt is generated and how a lost list can be regenerated.

Ten digits is 10^10 possibilities for 100 live codes, so guessing one is a
1-in-100-million shot per attempt; `LOOKUP_DELAY` makes grinding that slow
enough not to be worth anyone's time.

A voucher is bound to the first browser that spends a question on it, claimed
with SET NX so two browsers racing cannot both win. That is what stops a code
being passed around: the balance belongs to a device, not to whoever has seen
the number. It is deliberately a soft binding -- clearing site data or opening
a private window presents as a new device, and someone determined can do that.
It stops casual sharing, which is what it is for.
"""
from __future__ import annotations

import hashlib
import hmac
import os
import re

from .store import get_store, now

QUESTIONS_PER_VOUCHER = int(os.environ.get("VOUCHER_QUESTIONS", "10"))
VOUCHER_COUNT = int(os.environ.get("VOUCHER_COUNT", "100"))
CODE_RE = re.compile(r"^\d{10}$")

# Slows a brute-force sweep without being noticeable to a person typing a code.
LOOKUP_DELAY = 0.05


def secret() -> str:
    return os.environ.get("VOUCHER_SECRET", "").strip()


def enabled() -> bool:
    return bool(secret())


def derive_codes(key: str, count: int = VOUCHER_COUNT) -> list[str]:
    """The canonical voucher list for a secret, in order."""
    codes: list[str] = []
    seen: set[str] = set()
    index = 0
    while len(codes) < count:
        digest = hmac.new(key.encode(), f"voucher:{index}".encode(),
                          hashlib.sha256).digest()
        code = f"{int.from_bytes(digest[:8], 'big') % 10**10:010d}"
        index += 1
        if code in seen:          # astronomically unlikely; handled anyway
            continue
        seen.add(code)
        codes.append(code)
    return codes


_valid: frozenset[str] | None = None


def valid_codes() -> frozenset[str]:
    global _valid
    if _valid is None:
        _valid = frozenset(derive_codes(secret())) if enabled() else frozenset()
    return _valid


def is_valid(code: str) -> bool:
    return bool(CODE_RE.match(code or "")) and code in valid_codes()


def _used_key(code: str) -> str:
    return f"v:{code}:used"


def _device_key(code: str) -> str:
    return f"v:{code}:device"


def status(code: str, device: str) -> dict:
    """Report a voucher without spending anything from it."""
    if not enabled():
        return {"ok": True, "unlimited": True, "remaining": None,
                "reason": "vouchers are not enabled on this server"}
    if not is_valid(code):
        return {"ok": False, "reason": "invalid", "remaining": 0}

    store = get_store()
    owner = store.get(_device_key(code))
    if owner and device and owner != device:
        return {"ok": False, "reason": "bound_elsewhere", "remaining": 0}

    used = int(store.get(_used_key(code)) or 0)
    remaining = max(0, QUESTIONS_PER_VOUCHER - used)
    return {
        "ok": remaining > 0,
        "reason": "spent" if remaining == 0 else "ok",
        "remaining": remaining,
        "limit": QUESTIONS_PER_VOUCHER,
        "bound": bool(owner),
    }


def claim(code: str, device: str) -> dict:
    """Bind a voucher to this browser, or confirm it already is.

    Binding happens here rather than on first question so that redeeming a code
    that belongs to someone else fails immediately, with nothing spent.
    """
    if not enabled():
        return {"ok": True, "unlimited": True, "remaining": None}
    if not is_valid(code):
        return {"ok": False, "reason": "invalid", "remaining": 0}
    if not device:
        return {"ok": False, "reason": "no_device", "remaining": 0}

    store = get_store()
    if not store.setnx(_device_key(code), device):
        if store.get(_device_key(code)) != device:
            return {"ok": False, "reason": "bound_elsewhere", "remaining": 0}
    else:
        store.set(f"v:{code}:claimed_at", str(now()))

    return status(code, device)


def spend(code: str, device: str) -> dict:
    """Take one question off a voucher, atomically.

    INCR is the whole point: the balance is decided by the server in one step,
    so two tabs sending the last question at the same moment cannot both be
    told yes.
    """
    if not enabled():
        return {"ok": True, "unlimited": True, "remaining": None}

    checked = claim(code, device)
    if not checked.get("ok"):
        return checked

    store = get_store()
    used = store.incr(_used_key(code))
    if used > QUESTIONS_PER_VOUCHER:
        # Over-spent by a race; the voucher is finished either way.
        return {"ok": False, "reason": "spent", "remaining": 0,
                "limit": QUESTIONS_PER_VOUCHER}
    store.set(f"v:{code}:last_used", str(now()))
    return {"ok": True, "remaining": QUESTIONS_PER_VOUCHER - used,
            "limit": QUESTIONS_PER_VOUCHER}
