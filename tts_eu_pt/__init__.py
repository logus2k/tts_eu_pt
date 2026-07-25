"""tts_eu_pt — a permissively-licensed European Portuguese (pt-PT) text-to-speech voice.

Fine-tuned from Kokoro-82M, driven by a pt-PT G2P front-end (TugaPhone, Apache-2.0) with
European-Portuguese text normalisation: numbers, clock times, dates, honorifics
("D. Afonso I" -> "Dom Afonso Primeiro"), acronyms and English loanwords.

Quick start
-----------
    from tts_eu_pt import TTS

    tts = TTS()                              # downloads weights on first run
    wav = tts.say("Olá! São dezasseis horas.")   # numpy float32 @ 24 kHz
    tts.save("ola.wav", "Bem-vindo ao tts_eu_pt.")
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np

from . import g2p
from .download import ensure_weights

SAMPLE_RATE = 24000
_MAX_PHONEMES = 510          # Kokoro caps input at 510 tokens (kokoro/model.py)
__version__ = "0.1.0"
__all__ = ["TTS", "SAMPLE_RATE"]


def _sentences(text: str) -> List[str]:
    """Split into sentence-sized chunks on . ! ? and newlines, so no chunk exceeds the
    510-token cap. Honorifics/times are already expanded by g2p.normalize_text upstream,
    so a bare '.' here is a real sentence end."""
    out, buf = [], []
    for ch in text:
        buf.append(ch)
        if ch in ".!?\n":
            s = "".join(buf).strip()
            if s:
                out.append(s)
            buf = []
    tail = "".join(buf).strip()
    if tail:
        out.append(tail)
    return out or ([text.strip()] if text.strip() else [])


class TTS:
    """European Portuguese speech synthesiser.

    Parameters
    ----------
    device:
        "cuda", "cpu", or None to auto-select CUDA when available.
    model_path, voicepack_path:
        Optional local overrides. When omitted, the weights are downloaded once from the
        Hugging Face repo (see tts_eu_pt.download) and cached.
    """

    def __init__(
        self,
        device: Optional[str] = None,
        model_path: Optional[str] = None,
        voicepack_path: Optional[str] = None,
        repo_id: Optional[str] = None,
    ) -> None:
        import torch
        from .kokoro import KModel

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model_path, voicepack_path = ensure_weights(model_path, voicepack_path, repo_id)
        cfg = str(Path(__file__).resolve().parent / "assets" / "kokoro_config.json")

        self._torch = torch
        self.km = KModel(config=cfg, model=str(model_path)).to(self.device).eval()
        self.pack = torch.load(str(voicepack_path), map_location=self.device, weights_only=True)

    # -- public API -------------------------------------------------------------
    def say(self, text: str, speed: float = 1.0) -> np.ndarray:
        """Synthesise ``text`` to a float32 waveform at 24 kHz. ``speed`` > 1 is faster."""
        torch = self._torch
        chunks: List[np.ndarray] = []
        for sentence in _sentences(text):
            ps = g2p.phonemize(sentence)[:_MAX_PHONEMES]
            if not ps.strip():
                continue
            ref = self.pack[len(ps) - 1].to(self.device)
            with torch.no_grad():
                audio = self.km(ps, ref, float(speed), return_output=True).audio
            a = audio.detach().cpu().numpy() if hasattr(audio, "detach") else np.asarray(audio)
            chunks.append(a.astype(np.float32))

        if not chunks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(chunks) if len(chunks) > 1 else chunks[0]

    def save(self, path: str, text: str, speed: float = 1.0) -> str:
        """Synthesise ``text`` and write a 24 kHz mono WAV to ``path``. Returns the path."""
        import soundfile as sf

        wav = self.say(text, speed=speed)
        sf.write(path, wav, SAMPLE_RATE, subtype="PCM_16")
        return path
