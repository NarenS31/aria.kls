"""Input hardening for the summit demo: whatever arrives becomes a short, clean string, and never raises.

`sanitize_input(raw)` -> (text, flags).  `flags` is a list of short fixed tokens (never student text).
Handles: None / numbers / bytes / dicts / objects whose __str__ raises; NUL and other control characters;
zero-width and bidi-control characters; lone surrogates; non-ASCII decimal digits (mapped to 0-9);
full-width ASCII forms; whitespace floods; very long input (cut to MAX_INPUT_CHARS after cleaning).
Superscripts, minus signs and pi stay as they are: agent/math_check.normalize() already understands them.
"""
from __future__ import annotations

import re
import unicodedata

MAX_INPUT_CHARS = 1000
_PRESCAN = 4 * MAX_INPUT_CHARS          # never walk more than this many characters of a huge paste
_WS = re.compile(r"\s+")


def _to_str(raw) -> tuple:
    if isinstance(raw, str):
        return raw, []
    if raw is None:
        return "", ["non_text"]
    if isinstance(raw, (bytes, bytearray)):
        return bytes(raw[:_PRESCAN * 4]).decode("utf-8", errors="replace"), ["non_text", "bytes"]
    try:
        return str(raw), ["non_text"]
    except Exception:  # noqa: BLE001 - an object whose __str__ raises is still just "not text"
        return "", ["non_text", "unprintable"]


def sanitize_input(raw) -> tuple:
    s, flags = _to_str(raw)
    if len(s) > _PRESCAN:
        s = s[:_PRESCAN]
        flags.append("truncated")
    s = s.encode("utf-8", errors="replace").decode("utf-8", errors="replace")       # lone surrogates -> U+FFFD
    out, odd = [], set()
    for ch in s:
        o = ord(ch)
        if ch.isascii():
            if o < 32 or o == 127:
                if ch in "\t\n\r":
                    out.append(" ")
                else:
                    odd.add("control_chars")
                    out.append(" ")
            else:
                out.append(ch)
            continue
        cat = unicodedata.category(ch)
        if cat == "Nd":                                   # Arabic-Indic, Devanagari, ... digits
            out.append(str(unicodedata.decimal(ch)))
            odd.add("unicode_digits")
        elif 0xFF01 <= o <= 0xFF5E:                       # full-width ASCII forms
            out.append(chr(o - 0xFEE0))
            odd.add("fullwidth")
        elif cat == "Cf":                                 # zero-width, bidi overrides, soft hyphen
            odd.add("format_chars")
        elif cat in ("Cc", "Cs", "Co", "Cn"):
            odd.add("control_chars")
            out.append(" ")
        elif cat in ("Zs", "Zl", "Zp"):
            out.append(" ")
        elif ch == "�":
            odd.add("invalid_chars")
            out.append(" ")
        else:
            out.append(ch)
    text = _WS.sub(" ", "".join(out)).strip()
    if len(text) > MAX_INPUT_CHARS:
        text = text[:MAX_INPUT_CHARS]
        flags.append("truncated")
    flags += sorted(odd)
    if not text:
        flags.append("empty")
    return text, sorted(set(flags))
