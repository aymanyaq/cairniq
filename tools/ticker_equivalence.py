"""
Cross-listing / CDR ticker equivalence.

Maps economically equivalent listings so portfolio checks don't tag a US
ticker as "not held" when the user owns its Canadian twin (or vice versa) —
e.g. candidate MA vs held MA.TO (Mastercard CDR), or candidate SU.TO vs
held SU (Suncor, interlisted).

Two equivalence classes:
  1. CDRs (Canadian Depositary Receipts, Cboe Canada/NEO): CAD-hedged
     fractional wrappers on US large-caps. Same underlying economics.
  2. Interlisted names: the same company listed on both a Canadian
     exchange and NYSE/NASDAQ.

Matching is by curated root list, never naive suffix-stripping: T.TO is
Telus while T is AT&T; MG.TO is Magna while MG is Mistras. Roots not in
the curated sets never match across markets.
"""

from __future__ import annotations

# Canadian exchange suffixes (Yahoo-style symbols).
_CA_SUFFIXES = (".TO", ".NE", ".V", ".CN")

# US roots that trade as CDRs on Cboe Canada (candidate .TO/.NE root → same
# US root). CDR line-up grows over time; extend as needed.
_CDR_ROOTS = {
    "AAPL", "ABBV", "ABNB", "AMD", "AMZN", "AVGO", "BA", "CAT", "COIN",
    "COST", "CRM", "CSCO", "CVX", "DELL", "DIS", "GE", "GOOG", "GOOGL",
    "GS", "HD", "HON", "IBM", "INTC", "JNJ", "JPM", "KO", "LLY", "LOW",
    "MA", "MCD", "META", "MRK", "MSFT", "MU", "NFLX", "NKE", "NVDA",
    "ORCL", "PEP", "PFE", "PG", "PLTR", "PYPL", "QCOM", "RDDT", "RTX",
    "SBUX", "SMCI", "SNOW", "SQ", "TMO", "TSLA", "TXN", "UBER", "UNH",
    "UPS", "V", "VZ", "WMT", "XOM",
}

# Canadian companies interlisted in the US under the SAME root
# (RY.TO ↔ RY on NYSE, etc.). Only roots verified to be the same company
# on both sides belong here.
_INTERLISTED_SAME_ROOT = {
    "AEM", "AQN", "BAM", "BB", "BMO", "BN", "BNS", "CM", "CNQ", "CP",
    "CVE", "ENB", "FNV", "FTS", "GFL", "IMO", "MFC", "NTR", "OTEX",
    "OVV", "QSR", "RY", "SHOP", "SLF", "STN", "SU", "TD", "TRI", "TRP",
    "WCN", "WPM",
}

# Canadian root → US ticker where the roots DIFFER (or a class-share root
# maps to a plain US root). These are exactly the cases naive suffix
# stripping gets wrong.
_CA_TO_US_PAIRS = {
    "T": "TU",        # Telus (T.TO) → TU; NOT AT&T
    "CNR": "CNI",     # Canadian National Railway
    "PPL": "PBA",     # Pembina Pipeline; NOT PPL Corp
    "MG": "MGA",      # Magna; NOT Mistras Group
    "CCO": "CCJ",     # Cameco
    "K": "KGC",       # Kinross Gold
    "DOO": "DOOO",    # BRP Inc.
    "TECK.A": "TECK", # Teck Resources class shares
    "TECK.B": "TECK",
    "GIB.A": "GIB",   # CGI Inc.
    "BRK": "BRK.B",   # Berkshire CDR → US B shares
}

_SHARED_US_ROOTS = _CDR_ROOTS | _INTERLISTED_SAME_ROOT


def normalize_symbol(symbol: str) -> str:
    """Uppercase, trim, and unify class-share separators (BRK-B → BRK.B)."""
    return str(symbol or "").strip().upper().replace("-", ".")


def split_canadian_listing(symbol: str) -> tuple[str, str | None]:
    """Return (root, ca_suffix) — suffix is None for non-Canadian listings."""
    sym = normalize_symbol(symbol)
    for suffix in _CA_SUFFIXES:
        if sym.endswith(suffix) and len(sym) > len(suffix):
            return sym[: -len(suffix)], suffix
    return sym, None


def economic_key(symbol: str) -> str:
    """
    Canonical key for economic exposure: equivalent listings share a key.
    Unknown Canadian roots key to their full suffixed symbol so they can
    never collide with an unrelated US root.
    """
    sym = normalize_symbol(symbol)
    root, ca_suffix = split_canadian_listing(sym)
    if ca_suffix is None:
        return sym
    if root in _CA_TO_US_PAIRS:
        return _CA_TO_US_PAIRS[root]
    if root in _SHARED_US_ROOTS:
        return root
    return sym


def find_equivalent_holding(candidate: str, holding_symbols: list[str] | set[str]) -> str | None:
    """
    Return the first held symbol that is economically equivalent to
    ``candidate`` but not the identical listing (exact matches are the
    caller's plain already-held case). None when no twin is held.
    """
    cand = normalize_symbol(candidate)
    if not cand:
        return None
    key = economic_key(cand)
    for holding in holding_symbols:
        held = normalize_symbol(holding)
        if not held or held == cand:
            continue
        if economic_key(held) == key:
            return held
    return None
