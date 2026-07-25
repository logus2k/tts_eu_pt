#!/usr/bin/env python3
"""European Portuguese G2P via TugaPhone — the permissive replacement for espeak-ng.

WHY: espeak-ng is GPLv3, which forces copyleft on anyone redistributing a product built
on it. The project's goal is unrestricted commercial reuse, so espeak must go.
TugaPhone is Apache-2.0 and won an objective bake-off against espeak on a neutral
Wiktionary reference (PER 0.0994 vs 0.1147, n=30).

CRITICAL — THE TUGAPHONE VERSION MATTERS:
  Sub-regional lects need tugaphone >= 1.2.0a1 (pinned in pyproject.toml). On the OLDER
  STABLE release, get_dialect_inventory() only branched on pt-BR/AO/MZ/TL and defaulted
  everything else to EuropeanPortuguese(), so "pt-PT-x-lisbon", "pt-PT-x-porto" and even
  nonsense strings all returned generic pt-PT output and LisbonPortuguese was unreachable
  dead code. That fix has since shipped to PyPI, so the dev branch is no longer required:
  measured on 1.2.0a1, "pt-PT-x-lisbon" -> "u ˈviɲu ˈveɾd" vs generic "o ˈviɲu ˈveɾd".
  assert_lect_support() enforces this at run time, whatever release is installed.

  CAVEAT: an UNKNOWN lect string still falls back to generic pt-PT silently rather than
  raising, so a typo in LECT degrades quality without any error.

LECT: pt-PT-x-lisbon. TugaPhone's own scoreboard puts it far ahead of generic pt-PT
(PER 0.1007 vs 0.2294; word accuracy 0.4330 vs 0.1540).

This module mirrors scripts/ptpt_g2p.py so it can be swapped in as the single source of
truth for text -> input_ids. Training and inference MUST use the same one.
"""
import json
import sys
import unicodedata
from pathlib import Path

_ASSETS = Path(__file__).resolve().parent / "assets"

CFG = json.loads((_ASSETS / "kokoro_config.json").read_text())
VOCAB: dict[str, int] = CFG["vocab"]
N_TOKEN: int = CFG["n_token"]
ID2SYM: dict[int, str] = {i: s for s, i in VOCAB.items()}

MAX_PHONEME_LEN = 510   # kokoro/model.py caps input at 510 chars
BOUNDARY_ID = 0         # input_ids = [0, *ids, 0]
LECT = "pt-PT-x-lisbon"

# TugaPhone emits 4 symbols outside Kokoro's 178-token vocab. Measured over the full
# hard-case corpus; counts were r-uvular 12, dark-l 12, ascii-g 9, apico-alveolar-s 3.
SYMBOL_MAP = {
    "ʀ": "ʁ",   # uvular trill -> uvular fricative (id 128); pt-PT rhotic, same phoneme class
    "ɫ": "l",   # velarised (dark) l -> plain l (id 54); Kokoro has no dark-l token
    "g": "ɡ",   # ASCII g -> IPA script g (U+0261, id 92). Classic IPA trap.
    "̺": "",  # COMBINING INVERTED BRIDGE BELOW (apico-alveolar s). A genuine pt-PT
                   # feature Kokoro's vocab cannot represent -- dropped, not approximated.
}
SYLLABLE_SEP = "·"      # TugaPhone marks syllables; not a phoneme

_phonemizer = None


class UnmappedSymbol(ValueError):
    """Raised at prep time so unmapped symbols fail loudly instead of being silently
    dropped by kokoro's vocab.get(p) at inference time."""


def assert_lect_support() -> None:
    """Fail fast if the installed tugaphone ignores sub-regional lects.

    Probes behaviour rather than the version string, so it stays correct no matter which
    release or branch is installed.
    """
    tp = _get()
    a = tp.phonemize_sentence("o vinho verde", "pt-PT")
    b = tp.phonemize_sentence("o vinho verde", "pt-PT-x-porto")
    if a == b:
        raise RuntimeError(
            "The installed tugaphone ignores sub-regional lects, so "
            f"{LECT} would silently degrade to generic pt-PT. Upgrade:\n"
            '  pip install "tugaphone>=1.2.0a1"'
        )


# Former name, kept so existing callers keep working. The check is no longer about a
# branch -- the lect fix shipped to PyPI in 1.2.0a1.
assert_dev_branch = assert_lect_support


def _get():
    global _phonemizer
    if _phonemizer is None:
        from tugaphone import TugaPhonemizer
        _phonemizer = TugaPhonemizer()
    return _phonemizer


# European-Portuguese speech drops the connector "e" only INSIDE the twenties, and only
# before units 5-9:  25-29 -> "vinte cinco" ... "vinte nove", BUT 21/22/23/24 KEEP it ->
# "vinte e um/dois/três/quatro". Holds at any scale (124 -> "cento e vinte e quatro",
# 126 -> "cento e vinte seis", 2026 -> "dois mil e vinte seis"). Every other tens keeps it
# too (39 -> "trinta e nove"). Confirmed by the native speaker. num2words/tugaphone always
# insert the "e", so we strip it only for vinte + {cinco..nove}.
_ELIDE_E_AFTER = {"vinte"}
_ELIDE_UNITS = {"cinco", "seis", "sete", "oito", "nove"}


def _drop_tens_units_e(text: str) -> str:
    words = text.split()
    out, i = [], 0
    n = len(words)
    while i < n:
        if (i + 2 < n and words[i].lower() in _ELIDE_E_AFTER
                and words[i + 1].lower() == "e" and words[i + 2].lower() in _ELIDE_UNITS):
            out.append(words[i])         # keep "vinte"
            out.append(words[i + 2])     # keep the unit (4-9), drop the "e" between
            i += 3
        else:
            out.append(words[i])
            i += 1
    return " ".join(out)


_DASHES = {"-", "–", "—", "−"}


def _split_number_ranges(text: str) -> str:
    """A dash BETWEEN two digits is a numeric range: "1139-1185" -> "1139 a 1185", read
    "mil ... a mil ...". Without this the glued token "1139-1185" reaches num2words as one
    unparseable string, stays as raw digits, and gets DROPPED (digits aren't in the vocab).
    Char-scan (no regex): only a dash flanked by digits is touched, so word hyphens
    ("chamo-me", "guarda-chuva") stay intact."""
    n = len(text)
    out = []
    for i, ch in enumerate(text):
        if ch in _DASHES and 0 < i < n - 1 and text[i - 1].isdigit() and text[i + 1].isdigit():
            out.append(" a ")
        else:
            out.append(ch)
    return "".join(out)


def normalize_numbers(text: str) -> str:
    """Expand digits to words. TugaPhone does NOT do this inside phonemize_sentence --
    raw digits pass straight through and would reach the tokenizer. Uses long scale for
    pt-PT and handles gender agreement. Expands numeric ranges (1139-1185 -> "... a ...")
    and drops the tens+units "e" for natural pt-PT (25 -> "vinte cinco")."""
    try:
        from tugaphone.number_utils import normalize_numbers as _nn
        return _drop_tens_units_e(_nn(_split_number_ranges(text)))
    except Exception:
        return text


def _clean(ps: str) -> str:
    """Strip syllable separators, decompose precomposed nasals, apply the symbol map."""
    ps = ps.replace(SYLLABLE_SEP, "")
    # TugaPhone emits BOTH precomposed (ẽ) and decomposed (ɐ + U+0303) nasals.
    # NFD makes it uniform, and the combining tilde is vocab id 17.
    ps = unicodedata.normalize("NFD", ps)
    for src, dst in SYMBOL_MAP.items():
        ps = ps.replace(src, dst)
    return " ".join(ps.split())


def _strip_ordinal_marks(text: str) -> str:
    """Remove masculine/feminine ordinal indicators (3.º, 2.ª).

    TugaPhone's number parser raises decimal.InvalidOperation on these, so they are
    stripped as a fallback. Plain string replacement, no pattern matching.
    """
    for mark in ("º", "ª", "°"):
        text = text.replace("." + mark, "").replace(mark, "")
    return text


# Loanword override: English-pronounced borrowings ("online", "site", "email") that
# TugaPhone would lusophonize. Phonemes generated OFFLINE with espeak (GPL at prep time);
# only the phoneme dict ships. Curated to lowercase common nouns -- proper nouns and
# acronyms are excluded (Portuguese does not use their English pronunciation).
_LOANWORDS = None


def _loanwords() -> dict:
    global _LOANWORDS
    if _LOANWORDS is None:
        try:
            _LOANWORDS = json.loads((_ASSETS / "loanwords.json").read_text())
        except Exception:
            _LOANWORDS = {}
    return _LOANWORDS


def _loan_key(w: str) -> str:
    return w.strip(".,!?;:—…\"'()“”").lower()


def _phon_one(text: str, lect: str) -> str:
    try:
        return _clean(_get().phonemize_sentence(normalize_numbers(text), lect))
    except Exception:
        # ordinals (3.º) crash tugaphone's number parser; retry without the markers
        t = _strip_ordinal_marks(text)
        return _clean(_get().phonemize_sentence(normalize_numbers(t), lect))


# Acronyms spoken as Portuguese LETTER NAMES, not read as a syllable. "IA" (inteligência
# artificial) must sound "i á", not "ia"; "LLM" must sound "éle-éle-éme", not "lãm".
# The letter names are written with accents so TugaPhone gives the open-e spelling form
# (L = ˈɛlɨ, not the pronoun "ele" ˈel). The trailing comma injects a pause so the last
# letter does not merge into the next word. Expanded at the TEXT level (see phonemize) so
# the comma becomes a real KEEP_PUNCT pause token.
_ACRONYMS = {
    "IA": "i á,", "LLM": "éle éle éme,", "UTC": "u tê cê,",
    # The project's own name. Left raw, TugaPhone returns vowel-less clusters that
    # Portuguese phonotactics cannot realise -- "tts" -> ˈttʃ and "pt" -> ˈpt -- and the
    # decoder emits noise rather than speech. Confirmed by ear on 0.1.0.
    "tts_eu_pt": "tê tê ésse eu pê tê,",
    "tts-eu-pt": "tê tê ésse eu pê tê,",
    "TTS": "tê tê ésse,", "tts": "tê tê ésse,",
    "PT": "pê tê,",
    # NOTE: keys are matched case-SENSITIVELY on purpose. Lowercase "eu" is the pronoun
    # ("eu vou") and lowercase "ia" is a verb form ("ele ia"); mapping those to letter
    # names would corrupt ordinary text. Only add a lowercase key when the token is not
    # also a real Portuguese word -- "tts" qualifies, "eu" does not.
}

# Abbreviations expanded to full words at the TEXT level (before punctuation splitting,
# so the trailing "." is consumed here instead of becoming a sentence pause and the letter
# is not read on its own). "D." is the Portuguese honorific: "D. Afonso Henriques" ->
# "Dom Afonso Henriques". NOTE: "D." is gender-ambiguous ("D. Maria" = "Dona Maria"); this
# maps only the masculine "Dom" — female-name handling would need a name lookup.
_ABBREVIATIONS = {"D.": "Dom"}


_LEAD_PUNCT = '(“"‘¿¡'
_TRAIL_PUNCT = '.,;:!?)"”…'


def _split_edges(w: str) -> tuple[str, str, str]:
    """Return (leading punctuation, core, trailing punctuation) of a token."""
    i, j = 0, len(w)
    while i < j and w[i] in _LEAD_PUNCT:
        i += 1
    while j > i and w[j - 1] in _TRAIL_PUNCT:
        j -= 1
    return w[:i], w[i:j], w[j:]


def _expand_acronyms(text: str) -> str:
    out = []
    for w in text.split():
        # Honorific abbreviations key on the EXACT token — the trailing '.' is PART of "D."
        # (it's what distinguishes it from a lone "D"), so it must not be stripped.
        if w in _ABBREVIATIONS:
            out.append(_ABBREVIATIONS[w])
            continue
        # Acronyms (IA, LLM, UTC) may carry surrounding punctuation ("UTC.", "(UTC)"):
        # strip it for the lookup and reattach, so they still expand at a sentence end.
        lead, core, trail = _split_edges(w)
        repl = _ACRONYMS.get(core)
        if repl is not None:
            if trail and repl.endswith(","):
                repl = repl[:-1]   # the real punctuation already supplies the pause
            out.append(lead + repl + trail)
            continue
        out.append(w)
    return " ".join(out)


# Roman regnal numbers after a monarch's name are read as ORDINALS, not letters:
# "D. Afonso I" -> "Dom Afonso Primeiro", "D. João VI" -> "Dom João Sexto".
# Masculine ordinals only — queens ("D. Maria II" = "Maria Segunda") would need name
# gender we don't have; this maps to the masculine form. Extend the map for values > XX.
_ROMAN_ORDINAL_M = {
    "I": "Primeiro", "II": "Segundo", "III": "Terceiro", "IV": "Quarto", "V": "Quinto",
    "VI": "Sexto", "VII": "Sétimo", "VIII": "Oitavo", "IX": "Nono", "X": "Décimo",
    "XI": "Décimo primeiro", "XII": "Décimo segundo", "XIII": "Décimo terceiro",
    "XIV": "Décimo quarto", "XV": "Décimo quinto", "XVI": "Décimo sexto",
    "XVII": "Décimo sétimo", "XVIII": "Décimo oitavo", "XIX": "Décimo nono",
    "XX": "Vigésimo",
}


def _split_trailing_punct(w: str) -> tuple[str, str]:
    """Split a token into (core, trailing punctuation), e.g. 'I.' -> ('I', '.')."""
    i = len(w)
    while i > 0 and w[i - 1] in '.,;:!?)"“”':
        i -= 1
    return w[:i], w[i:]


def _expand_regnal_numerals(text: str) -> str:
    """Roman numerals -> Portuguese masculine ordinals. Multi-character numerals (II, III,
    …) always convert; single-character I/V/X convert only right after a capitalised
    name-like word, so stray initials/letters aren't turned into ordinals."""
    words = text.split()
    out = []
    for i, w in enumerate(words):
        core, trail = _split_trailing_punct(w)
        ordinal = _ROMAN_ORDINAL_M.get(core)
        if ordinal is not None:
            if len(core) >= 2:
                out.append(ordinal + trail)
                continue
            if i > 0:
                prev = _split_trailing_punct(words[i - 1])[0]
                if prev.isalpha() and prev[:1].isupper() and len(prev) > 1:
                    out.append(ordinal + trail)
                    continue
        out.append(w)
    return " ".join(out)


# Whole-hour times ("16:00") are read "<hora> horas", NOT "... e zero". "hora" is
# feminine, so 1/2 (and 21/22) take feminine forms; every other hour has no gender issue
# and is left as digits for num2words. Hours are 0-23.
_FEM_HOURS = {1: "uma", 2: "duas", 21: "vinte e uma", 22: "vinte e duas"}


def _expand_clock_times(text: str) -> str:
    """"HH:00" -> "<hora(s)>": 16:00 -> "dezasseis horas", 1:00 -> "uma hora". Only whole
    hours; other times keep the colon for _split_time_colons ("16:54" -> "... e ..."). Token
    scan (no regex), so it must run BEFORE _split_time_colons turns the colon into " e "."""
    out = []
    for tok in text.split():
        core, trail = _split_trailing_punct(tok)
        hh_s, sep, mm_s = core.partition(":")
        if sep and hh_s.isdigit() and mm_s == "00" and 0 <= int(hh_s) <= 23:
            hh = int(hh_s)
            unit = "hora" if hh == 1 else "horas"
            hour = _FEM_HOURS.get(hh, hh_s)   # feminine word, else digits (num2words expands)
            out.append(f"{hour} {unit}{trail}")
        else:
            out.append(tok)
    return " ".join(out)


def _split_time_colons(text: str) -> str:
    """A colon BETWEEN two digits is a clock time, read with "e": "16:54" ->
    "dezasseis e cinquenta e quatro". The colon is a KEEP_PUNCT token, so left alone it
    splits into a pause; this runs at the TEXT level before that split. Char-scan (no
    regex): only a colon flanked by digits is touched, so "Nota:", "são:" keep their pause."""
    n = len(text)
    out = []
    for i, ch in enumerate(text):
        if ch == ":" and 0 < i < n - 1 and text[i - 1].isdigit() and text[i + 1].isdigit():
            out.append(" e ")
        else:
            out.append(ch)
    return "".join(out)


def _normalize_separator_dashes(text: str) -> str:
    """A SPACED en/em-dash or hyphen used as a separator (" – ", " — ", " - ") becomes a
    comma pause, so "Henriques – Primeiro rei" doesn't run together (and isn't misheard as
    a regnal "Henriques Primeiro"). Number ranges ("1139-1185") have no surrounding spaces
    and are left for _split_number_ranges. Literal string replace, no regex."""
    for d in (" – ", " — ", " - "):
        text = text.replace(d, ", ")
    return text


def _normalize_number_separators(text: str) -> str:
    """European number formatting, handled BEFORE the '.'/',' splitter would break it:
    "." groups thousands and is dropped (92.073 -> 92073 -> "noventa e dois mil e setenta e
    três"); "," is the decimal mark, spoken "vírgula" (10,4 -> "dez vírgula quatro").
    Char-scan (no regex): only a separator BETWEEN two digits is touched, so sentence-final
    periods and list commas keep their pause."""
    n = len(text)
    out = []
    for i, ch in enumerate(text):
        between = 0 < i < n - 1 and text[i - 1].isdigit() and text[i + 1].isdigit()
        if between and ch == ".":
            continue                       # thousands separator -> drop
        if between and ch == ",":
            out.append(" vírgula ")        # decimal mark -> spoken word
            continue
        out.append(ch)
    return "".join(out)


def normalize_text(text: str) -> str:
    """Text-level normalisation applied BEFORE phonemization or any '.'-based sentence
    splitting: number separators (92.073 / 10,4), separator dashes (" – " -> ", "), acronyms
    (IA -> 'i á'), honorific abbreviations (D. -> Dom) and regnal numerals (Afonso I ->
    Afonso Primeiro). Idempotent, so it is safe to call more than once (the TTS server calls
    it before its streaming split; phonemize() calls it too)."""
    return _expand_regnal_numerals(_expand_acronyms(_normalize_separator_dashes(
        _split_time_colons(_expand_clock_times(_normalize_number_separators(text))))))


def _raw_phonemize(text: str, lect: str) -> str:
    lw = _loanwords()
    words = text.split()
    # Only drop to word-level when a loanword is actually present, so ordinary text
    # keeps TugaPhone's full-phrase quality.
    if lw and any(_loan_key(w) in lw for w in words):
        parts = []
        for w in words:
            k = _loan_key(w)
            parts.append(lw[k] if k in lw else _phon_one(w, lect))
        return " ".join(p for p in parts if p)
    return _phon_one(text, lect)


# Punctuation Kokoro can actually represent (verified against config.json vocab).
KEEP_PUNCT = {c for c in ';:,.!?—…"()“”' if c in VOCAB}


def phonemize(text: str, lect: str = LECT, keep_punct: bool = True) -> str:
    """Text -> European Portuguese IPA, ready for Kokoro's vocab.

    PUNCTUATION: phonemizers drop it, but Kokoro's vocab contains 13 punctuation tokens
    and the model uses them for prosody -- sentence-final pauses, comma breaths,
    intonation resets. Training without them produces flat, run-on delivery.

    So the text is split at punctuation, each span phonemized separately, and the marks
    re-inserted between the spans as real tokens.
    """
    # Normalise first (acronyms -> "i á,", D. -> Dom, Afonso I -> Afonso Primeiro). This
    # may inject punctuation (the "IA" comma), so the split below treats it as a pause.
    text = normalize_text(text)
    if not keep_punct:
        return _raw_phonemize(text, lect)

    out, buf = [], []
    for ch in text:
        if ch in KEEP_PUNCT:
            span = "".join(buf).strip()
            buf = []
            if span:
                ps = _raw_phonemize(span, lect)
                if ps:
                    out.append(ps)
            out.append(ch)          # the punctuation token itself
        else:
            buf.append(ch)
    span = "".join(buf).strip()
    if span:
        ps = _raw_phonemize(span, lect)
        if ps:
            out.append(ps)

    # join so punctuation hugs the preceding phonemes and a space follows, e.g. "dˈiɐ. ʃˈɐ̃mʊ"
    s = ""
    for tok in out:
        if tok in KEEP_PUNCT:
            s = s.rstrip() + tok
        else:
            s = (s + " " + tok) if s else tok
    return " ".join(s.split())


def phonemes_to_input_ids(ps: str) -> list[int]:
    """Reproduce kokoro/model.py tokenization exactly, but RAISE on unmapped symbols."""
    ids, dropped = [], []
    for s in ps:
        i = VOCAB.get(s)
        if i is None:
            dropped.append(s)
        else:
            ids.append(i)
    if dropped:
        detail = ", ".join(
            f"{d!r} (U+{ord(d):04X} {unicodedata.name(d, '?')})" for d in sorted(set(dropped)))
        raise UnmappedSymbol(f"symbols absent from Kokoro vocab: {detail}")
    return [BOUNDARY_ID, *ids, BOUNDARY_ID]


def text_to_input_ids(text: str) -> tuple[str, list[int]]:
    ps = phonemize(text)
    if len(ps) > MAX_PHONEME_LEN:
        raise ValueError(
            f"phoneme string is {len(ps)} chars, over the {MAX_PHONEME_LEN} cap; chunk upstream")
    return ps, phonemes_to_input_ids(ps)


if __name__ == "__main__":
    assert_lect_support()
    for t in ["Bom dia. Chamo-me Teodoro e vivo em Lisboa.", "Custa 1234 euros."]:
        ps, ids = text_to_input_ids(t)
        print(f"{t}\n  -> {ps}\n  -> {len(ids)} ids\n")
