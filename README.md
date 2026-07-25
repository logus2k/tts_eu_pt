# tts_eu_pt - an European Portuguese text-to-speech based on Kokoro TTS

A small, **permissively-licensed** (Apache-2.0) **European Portuguese (pt-PT)** voice,
fine-tuned from [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M). It runs on **CPU**,
ships as a tiny Python package.

## Why it's different

The hard part of pt-PT TTS isn't the acoustics — it's reading text the way a Portuguese
speaker actually does. `tts_eu_pt` handles, out of the box:

| Input | Spoken as |
|---|---|
| `25`, `125`, `1139` | *vinte cinco*, *cento e vinte cinco*, *mil cento e trinta e nove* |
| `16:54`, `16:00` | *dezasseis e cinquenta e quatro*, *dezasseis horas* |
| `1139-1185` | *mil cento e trinta e nove **a** mil cento e oitenta e cinco* |
| `D. Afonso I` | *Dom Afonso Primeiro* |
| `IA`, `LLM`, `UTC` | *i-á*, *éle-éle-éme*, *u-tê-cê* |
| `online`, `software` | kept in their English pronunciation (as Portuguese speakers say them) |

The G2P (Grapheme-to-Phoneme) is built on the Apache-2.0 [TugaPhone](https://github.com/TigreGotico/tugaphone)
front-end (Lisbon lect).

## Install

```bash
pip install tts_eu_pt
```

## Quick start

```python
from tts_eu_pt import TTS

tts = TTS()                                       # downloads the voice on first run
wav = tts.say("Olá! São dezasseis horas.")        # numpy float32 @ 24 kHz
tts.save("ola.wav", "Bem-vindo ao tts_eu_pt.")
```

Command line:

```bash
python -m examples.cli "D. Afonso I reinou de 1139 a 1185." reis.wav
```

## Web echo demo

A one-page demo: type a sentence, press Enter, the page speaks it back in European Portuguese.

```bash
pip install "tts_eu_pt[server]"
python -m examples.server            # open http://localhost:8000
```

See [`examples/web-echo/`](examples/web-echo/) — a static page plus a ~50-line FastAPI
backend (browsers can't run the torch model directly).

## How it works

- **Weights** encode the *language* (pt-PT phonology/accent); a small **voicepack** encodes
  the *speaker*. They are separate files.
- Text → `tts_eu_pt.g2p` (normalisation + TugaPhone) → phoneme ids → Kokoro acoustic model
  → 24 kHz audio.
- The Kokoro acoustic model is vendored (`tts_eu_pt/kokoro/`, Apache-2.0). Only the acoustic
  `KModel` is used; the upstream pipeline's GPL-dependent G2P is not.

## Licensing

Apache-2.0.

## Status

Early release: the runtime + SDK + examples.
