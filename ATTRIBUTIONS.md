# Attributions & third-party licenses

`tts_eu_pt` is Apache-2.0. Everything required to **run** it is permissively licensed.

## Bundled / vendored

| Component | Where | License | Notes |
|---|---|---|---|
| **Kokoro** acoustic model code | `tts_eu_pt/kokoro/` | Apache-2.0 | Vendored from [hexgrad/kokoro](https://github.com/hexgrad/kokoro). Only the acoustic `KModel` — the upstream `KPipeline` (which pulls GPL espeak-ng via `misaki`) is **not** included. Full text in `LICENSE.kokoro`. |
| Model weights | Hugging Face (downloaded) | Apache-2.0 | Fine-tuned from [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) (Apache-2.0). |
| `kokoro_config.json` | `tts_eu_pt/assets/` | Apache-2.0 | Kokoro vocab / config. |

## Runtime dependencies

| Package | License |
|---|---|
| [TugaPhone](https://github.com/TigreGotico/tugaphone) (pt-PT G2P) | Apache-2.0 |
| PyTorch | BSD-3-Clause |
| transformers, huggingface_hub | Apache-2.0 |
| loguru | MIT |
| numpy, soundfile, num2words | BSD / MIT-style |

The Kokoro model architecture derives from [StyleTTS2](https://github.com/yl4579/StyleTTS2) (MIT).

## espeak-ng (GPLv3) — offline only, not shipped

The English-loanword pronunciation table (`tts_eu_pt/assets/loanwords.json`) was generated
**offline** with espeak-ng at data-prep time. Only the resulting phonemes are distributed
(as data); the espeak-ng tool is **not** a runtime dependency and is not redistributed. No
GPL code is required to install or run `tts_eu_pt`.
