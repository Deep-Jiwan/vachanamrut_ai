"""Vouchers: a fixed set of codes, each worth a fixed number of questions.

The codes are derived from a secret rather than stored, so the server can tell
a real code from a made-up one without a lookup table and without the codes
ever being committed. The same secret reproduces the same hundred codes, which
is how vouchers.txt is generated and how a lost list is regenerated.

Ten digits is 10^10 possibilities for 100 live codes, so a guess is a
1-in-100-million shot.

A voucher is bound to the first browser that redeems it, and the balance is
moved by SQL rather than read-modify-write -- `UPDATE ... WHERE used < limit`,
with the affected row count deciding the outcome, so two tabs racing for the
last question cannot both be told yes.

The binding is deliberately soft: clearing site data or opening a private
window presents as a new device. It stops a code being passed around casually,
which is what it is for, not a determined person.
"""
from __future__ import annotations

import hashlib
import hmac
import os
import re

from .db import execute, now, one

QUESTIONS_PER_VOUCHER = int(os.environ.get("VOUCHER_QUESTIONS", "10"))
VOUCHER_COUNT = int(os.environ.get("VOUCHER_COUNT", "100"))
CODE_RE = re.compile(r"^\d{10}$")


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


def _row(code: str) -> dict | None:
    return one("SELECT code, device, used FROM vouchers WHERE code = ?", (code,))


def _report(row: dict | None) -> dict:
    used = int((row or {}).get("used") or 0)
    remaining = max(0, QUESTIONS_PER_VOUCHER - used)
    return {
        "ok": remaining > 0,
        "reason": "spent" if remaining == 0 else "ok",
        "remaining": remaining,
        "limit": QUESTIONS_PER_VOUCHER,
        "bound": bool((row or {}).get("device")),
    }


def status(code: str, device: str) -> dict:
    """Report a voucher without spending anything from it."""
    if not enabled():
        return {"ok": True, "unlimited": True, "remaining": None,
                "reason": "vouchers are not enabled on this server"}
    if not is_valid(code):
        return {"ok": False, "reason": "invalid", "remaining": 0}

    row = _row(code)
    if row and row.get("device") and device and row["device"] != device:
        return {"ok": False, "reason": "bound_elsewhere", "remaining": 0}
    return _report(row)


def claim(code: str, device: str) -> dict:
    """Bind a voucher to this browser, or confirm it already is.

    Binding happens on redemption rather than on the first question, so a code
    that belongs to someone else fails immediately with nothing spent.
    """
    if not enabled():
        return {"ok": True, "unlimited": True, "remaining": None}
    if not is_valid(code):
        return {"ok": False, "reason": "invalid", "remaining": 0}
    if not device:
        return {"ok": False, "reason": "no_device", "remaining": 0}

    execute("INSERT INTO vouchers (code, used) VALUES (?, 0) "
            "ON CONFLICT(code) DO NOTHING", (code,))
    # Only succeeds while the voucher is unclaimed, so two browsers racing to
    # redeem the same code cannot both take it.
    execute("UPDATE vouchers SET device = ?, claimed_at = ? "
            "WHERE code = ? AND device IS NULL", (device, now(), code))

    row = _row(code)
    if not row or row.get("device") != device:
        return {"ok": False, "reason": "bound_elsewhere", "remaining": 0}
    return _report(row)


def spend(code: str, device: str) -> dict:
    """Take one question off a voucher, atomically."""
    if not enabled():
        return {"ok": True, "unlimited": True, "remaining": None}

    checked = claim(code, device)
    if not checked.get("ok"):
        return checked

    _, changed = execute(
        "UPDATE vouchers SET used = used + 1, last_used_at = ? "
        "WHERE code = ? AND device = ? AND used < ?",
        (now(), code, device, QUESTIONS_PER_VOUCHER))
    if not changed:
        return {"ok": False, "reason": "spent", "remaining": 0,
                "limit": QUESTIONS_PER_VOUCHER}

    # Reported directly rather than through _report, because the question that
    # takes the balance to zero has still been paid for and must be answered.
    used = int((_row(code) or {}).get("used") or QUESTIONS_PER_VOUCHER)
    return {"ok": True, "reason": "ok",
            "remaining": max(0, QUESTIONS_PER_VOUCHER - used),
            "limit": QUESTIONS_PER_VOUCHER}
