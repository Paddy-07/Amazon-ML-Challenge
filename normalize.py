#!/usr/bin/env python3
"""
Amazon ML Challenge 2026 — Text Normalization Module (v2 — bugs fixed)

Normalizes business_name / business_address strings so that noisy variants
of the same real business collapse toward the same representation before
any similarity scoring happens.

Pipeline order (each step matters — see inline notes for why):
  1. Literal "null"/"NULL" token removal (distinct from real NaN handling,
     which callers should do separately with pandas .isna()).
  2. Strip invisible Unicode "format" characters (category Cf), e.g. the
     zero-width non-joiner seen inside Telugu text in the sample data.
  3. Regional Indian state-name lookup (native script -> English), applied
     to raw text before romanization so it works even without the optional
     transliteration library.
  4. Optional Indic-script -> Latin romanization via `indic_transliteration`
     (degrades gracefully — leaves text as-is + warns once — if not
     installed).
  5. Diacritic stripping (NFKD + drop combining marks) — but ONLY applied
     to text that is NOT still in a native Indic script. Applying this
     blindly (as v1 did) strips vowel signs out of un-romanized
     Devanagari/Kannada/Telugu text, corrupting it further instead of
     leaving it untouched.
  6. Ampersand normalization ("&" -> "and") — done BEFORE punctuation
     stripping, since punctuation stripping deletes "&" first and makes
     the old post-hoc dictionary entry unreachable.
  7. State/country-subdivision code protection — 2-letter tokens that
     exactly match a known US or India state/UT code (e.g. "CT", "OR",
     "KA") are protected from the address-abbreviation dictionary so
     "CT" (Connecticut) never gets expanded into "court" the way "ct"
     (street suffix) does.
  8. Domain-suffix stripping (.com/.in/.org/.net/.co, "www.").
  9. Lowercasing, punctuation -> whitespace, whitespace collapsing.
 10. Legal-suffix / common-abbreviation expansion (Corp -> corporation,
     Rd -> road, etc.), skipping any token protected in step 7.

Design note: this only touches text formatting/script — it never looks up
or augments with any external business data, so it stays within the
challenge's "no external entity lookup" rule.

Usage as a library:
    from normalize import normalize_name, normalize_address

Usage as a script (runs a self-test against real sampled pairs from the
matched-pair inspection):
    python3 normalize.py
"""

import re
import unicodedata

# --- optional Indic transliteration ------------------------------------
try:
    from indic_transliteration import sanscript
    from indic_transliteration.sanscript import transliterate as _indic_transliterate
    _HAVE_INDIC = True
except ImportError:
    _HAVE_INDIC = False

_SCRIPT_MAP = {
    "DEVANAGARI": "devanagari",
    "KANNADA": "kannada",
    "TELUGU": "telugu",
    "GUJARATI": "gujarati",
    "TAMIL": "tamil",
    "MALAYALAM": "malayalam",
    "BENGALI": "bengali",
    "GURMUKHI": "gurmukhi",
}

_INDIC_SCRIPT_RANGES = [
    (0x0900, 0x097F, "DEVANAGARI"),
    (0x0980, 0x09FF, "BENGALI"),
    (0x0A00, 0x0A7F, "GURMUKHI"),
    (0x0A80, 0x0AFF, "GUJARATI"),
    (0x0B80, 0x0BFF, "TAMIL"),
    (0x0C00, 0x0C7F, "TELUGU"),
    (0x0C80, 0x0CFF, "KANNADA"),
    (0x0D00, 0x0D7F, "MALAYALAM"),
]


def _detect_indic_script(text):
    """Return the sanscript scheme name if `text` is predominantly one
    Indic script, else None."""
    counts = {}
    for ch in text:
        cp = ord(ch)
        for lo, hi, name in _INDIC_SCRIPT_RANGES:
            if lo <= cp <= hi:
                counts[name] = counts.get(name, 0) + 1
                break
    if not counts:
        return None
    dominant = max(counts, key=counts.get)
    return _SCRIPT_MAP.get(dominant)


_warned_missing_indic = False


def romanize(text):
    """Best-effort romanization of Indic-script text to Latin script.
    No-op for text that isn't predominantly an Indic script. Degrades
    gracefully (returns text unchanged, warns once) if the
    `indic_transliteration` package isn't installed."""
    global _warned_missing_indic
    scheme = _detect_indic_script(text)
    if scheme is None:
        return text
    if not _HAVE_INDIC:
        if not _warned_missing_indic:
            print(
                "[normalize] NOTE: Indic-script text detected but "
                "`indic_transliteration` isn't installed — those fields will "
                "stay unromanized (and left un-diacritic-stripped, so they "
                "aren't corrupted). Install with: pip install indic_transliteration"
            )
            _warned_missing_indic = True
        return text
    try:
        return _indic_transliterate(text, scheme, sanscript.ITRANS)
    except Exception:
        return text


def _strip_format_chars(text):
    """Remove invisible Unicode 'format' characters (category Cf), e.g.
    zero-width non-joiners seen inside Telugu text in the sample data."""
    return "".join(ch for ch in text if unicodedata.category(ch) != "Cf")


# --- Indian state names in regional script -> English -------------------
_STATE_LOOKUP = {
    "महाराष्ट्र": "maharashtra",
    "कर्नाटक": "karnataka",
    "ಕರ್ನಾಟಕ": "karnataka",
    "గుజరాత్": "gujarat",
    "ગુજરાત": "gujarat",
    "तमिलनाडु": "tamil nadu",
    "தமிழ்நாடு": "tamil nadu",
    "पश्चिम बंगाल": "west bengal",
    "তেলেঙ্গানা": "telangana",
    "तेलंगाना": "telangana",
    "తెలంగాణ": "telangana",
    "केरल": "kerala",
    "കേരളം": "kerala",
    "राजस्थान": "rajasthan",
    "पंजाब": "punjab",
    "ਪੰਜਾਬ": "punjab",
    "दिल्ली": "delhi",
    "उत्तर प्रदेश": "uttar pradesh",
    "बिहार": "bihar",
    "मध्य प्रदेश": "madhya pradesh",
}

# --- 2-letter subdivision codes, protected from abbreviation expansion --
# Without this, "CT" (Connecticut) gets mangled into "court" by the "ct"
# -> "street" style dictionary below, since both are legitimately 2-letter
# tokens. Confirmed against real sample data ('...Prospect, CT').
_US_STATE_CODES = {
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI", "ID",
    "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS",
    "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "ND", "OH", "OK",
    "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV",
    "WI", "WY", "DC",
}
_INDIA_STATE_CODES = {
    "AP", "AR", "AS", "BR", "CG", "GA", "GJ", "HR", "HP", "JH", "KA", "KL",
    "MP", "MH", "MN", "ML", "MZ", "NL", "OD", "PB", "RJ", "SK", "TN", "TG",
    "TR", "UP", "UK", "WB", "DL", "JK", "LA", "PY",
}
_PROTECTED_CODES = _US_STATE_CODES | _INDIA_STATE_CODES
_STATE_CODE_RE = re.compile(
    r"\b(" + "|".join(sorted(_PROTECTED_CODES)) + r")\b"
)

# --- abbreviation / legal-suffix expansion ------------------------------
# Keys are already-punctuation-stripped, lowercase tokens (periods are
# gone by the time this dict is consulted).
_ABBREV = {
    "corp": "corporation",
    "ltd": "limited",
    "pvt": "private",
    "inc": "incorporated",
    "llc": "llc",
    "llp": "llp",
    "pllc": "pllc",
    "rd": "road",
    "st": "street",
    "ave": "avenue",
    "blvd": "boulevard",
    "dr": "drive",
    "ln": "lane",
    "ct": "court",
}

_NULL_TOKEN_RE = re.compile(r"\bnull\b", flags=re.IGNORECASE)
_DOMAIN_SUFFIX_RE = re.compile(r"\b(www\.)?[\w.-]+\.(com|in|org|net|co)\b", flags=re.IGNORECASE)
_AMPERSAND_RE = re.compile(r"\s*&\s*")
_WS_RE = re.compile(r"\s+")


def _strip_punctuation(text):
    """Replace punctuation/symbols with a space, but — unlike a plain
    `[^\\w\\s]` regex — correctly preserve combining marks (Unicode
    category M), which is how vowel signs are represented in Devanagari/
    Kannada/Telugu/etc. Python's regex `\\w` does NOT count those as word
    characters, so a naive punctuation-strip regex silently deletes vowel
    signs from un-romanized Indic text (confirmed against real sample
    data). Underscore is explicitly preserved too, since it's used as the
    glue character in the state-code placeholder tokens above.
    """
    out = []
    for ch in text:
        if ch.isspace() or ch == "_" or unicodedata.category(ch)[0] in ("L", "M", "N"):
            out.append(ch)
        else:
            out.append(" ")
    return "".join(out)


def _strip_diacritics(text):
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in nfkd if not unicodedata.combining(ch))


def _apply_state_lookup(text):
    for native, english in _STATE_LOOKUP.items():
        text = text.replace(native, english)
    return text


def _protect_state_codes(text):
    """Replace standalone US/India state codes with a placeholder token
    so the abbreviation dictionary below can't mangle them. Must run
    BEFORE lowercasing — state codes in this dataset appear ALL CAPS."""
    return _STATE_CODE_RE.sub(lambda m: f"__state_{m.group(1).lower()}__", text)


def _base_normalize(text):
    if text is None:
        return ""
    text = str(text)
    if not text.strip() or text.strip().lower() == "nan":
        return ""

    text = _NULL_TOKEN_RE.sub(" ", text)
    text = _strip_format_chars(text)
    text = _apply_state_lookup(text)
    text = romanize(text)

    # Only strip combining diacritics once the text is confirmed to be out
    # of native Indic script (either it never was, or romanization just
    # converted it to Latin) — stripping combining marks from un-romanized
    # Devanagari/Kannada/Telugu would delete vowel signs and corrupt it.
    if _detect_indic_script(text) is None:
        text = _strip_diacritics(text)

    text = _AMPERSAND_RE.sub(" and ", text)
    text = _DOMAIN_SUFFIX_RE.sub(lambda m: m.group(0).rsplit(".", 1)[0], text)
    text = _protect_state_codes(text)
    text = text.lower()
    text = _strip_punctuation(text)
    text = _WS_RE.sub(" ", text).strip()

    tokens = [_ABBREV.get(tok, tok) for tok in text.split(" ") if tok]
    return " ".join(tokens)


def normalize_name(name):
    """Normalize a business_name field."""
    return _base_normalize(name)


def normalize_address(address):
    """Normalize a business_address field."""
    return _base_normalize(address)


# --- self-test against the real sampled pairs from inspect_matches.py --
_SAMPLE_PAIRS = [
    ("Advanced Circle Group", "Advanced Cilce Group"),
    ("J 8 Innovative Motors LLC", "[LLC] J 8 Innovative Motors"),
    ("J 8 Innovative Motors LLC", "J 8 MOTORS INNOVATIVE INNOVATIVE LLC"),
    ("Gwen R. Stagner, O.D.", "Gwen R. (Stagner,)"),
    ("Lina Garvin Martin Inc", "linagarvinmartin.com"),
    ("United Bny Clinic", "United Clinic Bny"),
    ("Teamsters Local 425", "teamsterslocal425.com"),
    ("Bright Seafood Inc.", "Bright Seafhiigod (Inc.) | www.brightsea.com"),
    ("Jemmott Tax Service Corp", "Jemmott Táx Service Corp"),
    ("Anand United Global Private Limited", "ಆನಂದ್ ಯುನೈಟೆಡ್ ಗ್ಲೋಬಲ್ ಪ್ರೈವೇಟ್ ಲಿಮಿಟೆಡ್"),
    ("Maa Management Private Limited", "माँ मैनेजमेंट प्राइवेट लिमिटेड"),
    ("Golden Foods LLP", "గోల్డెన్ ఫుడ్స్ ఎల్\u200cఎల్\u200cపీ"),
    ("Q & K Apartment Corp", "Corp Q + K Apartment"),
    ("Bay Disciplined", "Bay Discip1ined"),
]

_SAMPLE_ADDRESSES = [
    (
        "Embassy Tech Village, Lblock, Devarabisanahalli, Bangalore South, Bangalore, Karnataka",
        "EMBASSY TECH VILLAGE, LBLOCK, DEVARABISANAHALLI, BANGALORE SOUTH, Karnataka",
    ),
    (
        "1171 Witherspoon Road, Columbia, TN",
        "1171 WITHERSPOON RD, COLUMBIA, NULL, TN",
    ),
    (
        "No 16 Office No 9 Narasaraju Complex Basava Nagar Main Road Hoodi Whitefield, Road, Bangalore, Karnataka",
        "16 OFFICE NO 9 NARASARAJU COMPLEX BASAVA NAGAR MAIN ROAD HOODI WHITEFIELD, ROAD, BANGALORE, ಕರ್ನಾಟಕ",
    ),
    # regression check for the CT (Connecticut) vs "court" bug fix:
    (
        "15 Cheryl Lane, Prospect, CT",
        "15 Cheryl Lane, Propect CITY, Connecticut",
    ),
]


def _jaccard(a, b):
    sa, sb = set(a.split()), set(b.split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _self_test():
    print(f"indic_transliteration installed: {_HAVE_INDIC}\n")
    print("=" * 100)
    print("NAME PAIRS")
    print("=" * 100)
    for a, b in _SAMPLE_PAIRS:
        na, nb = normalize_name(a), normalize_name(b)
        j = _jaccard(na, nb)
        print(f"\nRAW A: {a!r}\nRAW B: {b!r}")
        print(f"NORM A: {na!r}\nNORM B: {nb!r}")
        print(f"token Jaccard: {j:.2f}")

    print("\n" + "=" * 100)
    print("ADDRESS PAIRS")
    print("=" * 100)
    for a, b in _SAMPLE_ADDRESSES:
        na, nb = normalize_address(a), normalize_address(b)
        j = _jaccard(na, nb)
        print(f"\nRAW A: {a!r}\nRAW B: {b!r}")
        print(f"NORM A: {na!r}\nNORM B: {nb!r}")
        print(f"token Jaccard: {j:.2f}")


if __name__ == "__main__":
    _self_test()