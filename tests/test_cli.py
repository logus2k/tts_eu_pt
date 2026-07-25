"""CLI argument-resolution tests. No model weights needed — parsing only.

Regression guard for the bug where `--text "Bom dia." ola.wav` silently wrote to the
default out.wav, because argparse bound the lone positional to `text` (nargs="?") rather
than to `out`.
"""
import pytest

from tts_eu_pt.cli import DEFAULT_OUT, _parse


def _resolve(argv):
    _, text, out = _parse(argv)
    return text, out


def test_positional_text_and_out():
    assert _resolve(["Bom dia.", "ola.wav"]) == ("Bom dia.", "ola.wav")


def test_positional_text_only_uses_default_out():
    assert _resolve(["Bom dia."]) == ("Bom dia.", DEFAULT_OUT)


def test_text_option_with_positional_out():
    # THE REGRESSION: the output path must be honoured, not silently replaced by out.wav.
    assert _resolve(["--text", "Bom dia.", "ola.wav"]) == ("Bom dia.", "ola.wav")


def test_text_option_only_uses_default_out():
    assert _resolve(["--text", "Bom dia."]) == ("Bom dia.", DEFAULT_OUT)


def test_out_option_is_equivalent_to_positional():
    assert _resolve(["--text", "Bom dia.", "--out", "ola.wav"]) == ("Bom dia.", "ola.wav")
    assert _resolve(["Bom dia.", "--out", "ola.wav"]) == ("Bom dia.", "ola.wav")


def test_out_given_twice_is_an_error():
    with pytest.raises(SystemExit):
        _parse(["--text", "Bom dia.", "a.wav", "--out", "b.wav"])


def test_too_many_positionals_after_text_option():
    with pytest.raises(SystemExit):
        _parse(["--text", "Bom dia.", "a.wav", "b.wav"])


def test_too_many_positionals():
    with pytest.raises(SystemExit):
        _parse(["Bom dia.", "a.wav", "b.wav"])


def test_no_text_is_an_error():
    with pytest.raises(SystemExit):
        _parse([])


def test_blank_text_is_an_error():
    with pytest.raises(SystemExit):
        _parse(["--text", "   "])


def test_speed_and_device_pass_through():
    args, text, out = _parse(["--text", "Bom dia.", "--speed", "1.3", "--device", "cpu", "x.wav"])
    assert (text, out, args.speed, args.device) == ("Bom dia.", "x.wav", 1.3, "cpu")
