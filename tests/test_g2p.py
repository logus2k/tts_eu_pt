"""G2P / text-normalisation tests. These need no model weights — pure text -> phonemes.

Run: pytest -q  (from the repo root)
"""
from tts_eu_pt import g2p


def _norm(t):
    return g2p.normalize_text(t)


def test_numbers_vinte_elides_others_keep_e():
    assert g2p.normalize_numbers("125") == "cento e vinte cinco"
    assert g2p.normalize_numbers("1139") == "mil cento e trinta e nove"
    assert g2p.normalize_numbers("39") == "trinta e nove"


def test_twenties_elide_only_units_5_to_9():
    # 21-24 keep the "e"; 25-29 drop it. Holds at every scale.
    assert g2p.normalize_numbers("24") == "vinte e quatro"
    assert g2p.normalize_numbers("26") == "vinte seis"
    assert g2p.normalize_numbers("124") == "cento e vinte e quatro"
    assert g2p.normalize_numbers("126") == "cento e vinte seis"
    assert g2p.normalize_numbers("2024") == "dois mil e vinte e quatro"
    assert g2p.normalize_numbers("2026") == "dois mil e vinte seis"
    assert g2p.normalize_numbers("21") == "vinte e um"
    assert g2p.normalize_numbers("25") == "vinte cinco"


def test_honorific_and_regnal():
    assert _norm("D. Afonso I") == "Dom Afonso Primeiro"
    assert _norm("D. João VI") == "Dom João Sexto"


def test_clock_times():
    # whole hour -> "horas"; other minutes -> "e"
    assert _norm("16:00") == "16 horas"
    assert _norm("1:00") == "uma hora"
    assert "16 e 54" == _norm("16:54")


def test_acronyms_letter_names_punctuation_tolerant():
    assert _norm("UTC") == "u tê cê,"
    assert _norm("São 16:00 UTC.") == "São 16 horas u tê cê."


def test_european_number_separators():
    # "." = thousands (dropped); "," = decimal (spoken "vírgula")
    assert g2p.normalize_numbers(_norm("92.073")) == "noventa e dois mil e setenta e três"
    assert g2p.normalize_numbers(_norm("10,4")) == "dez vírgula quatro"


def test_word_hyphens_untouched():
    assert _norm("chamo-me Teodoro") == "chamo-me Teodoro"


def test_date_range_reads_as_a():
    # digit-dash-digit becomes " a " so both numbers are spoken
    assert "1139 a 1185" in g2p.normalize_numbers(_norm("(1139-1185)"))


def test_phonemes_tokenise_without_unmapped_symbols():
    ps = g2p.phonemize("Olá! São dezasseis horas. D. Afonso I.")
    ids = g2p.phonemes_to_input_ids(ps)   # raises UnmappedSymbol on anything off-vocab
    assert len(ids) > 5
