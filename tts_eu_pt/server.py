#!/usr/bin/env python3
"""Minimal HTTP backend for the web-echo demo: POST text -> WAV bytes.

Browsers can't run the torch model directly, so this ~50-line FastAPI app loads the voice
once and exposes a single /tts endpoint. It also serves the static web-echo page.

    pip install "tts_eu_pt[server]"
    python -m tts_eu_pt.server            # then open http://localhost:8000

This lives inside the package (not in examples/) so it is importable after a plain
``pip install``, not only from a repo checkout.
"""
import io
import os
from pathlib import Path

import soundfile as sf
from fastapi import FastAPI, Body
from fastapi.responses import Response, FileResponse
from fastapi.staticfiles import StaticFiles

from tts_eu_pt import TTS, SAMPLE_RATE

app = FastAPI(title="tts_eu_pt web-echo")
_tts: TTS | None = None
_WEB = Path(__file__).resolve().parent / "web"


def get_tts() -> TTS:
    """Load the voice once, on first request.

    TTS_EU_PT_MODEL / TTS_EU_PT_VOICEPACK let you point the demo at local weights, which
    is the only way to run it until the Hugging Face model repo is published (see
    tts_eu_pt.download).
    """
    global _tts
    if _tts is None:
        _tts = TTS(
            model_path=os.environ.get("TTS_EU_PT_MODEL"),
            voicepack_path=os.environ.get("TTS_EU_PT_VOICEPACK"),
        )
    return _tts


@app.post("/tts")
def tts(text: str = Body(..., embed=True)):
    """Synthesize `text` and return a 24 kHz mono WAV."""
    wav = get_tts().say(text)
    buf = io.BytesIO()
    sf.write(buf, wav, SAMPLE_RATE, format="WAV", subtype="PCM_16")
    return Response(content=buf.getvalue(), media_type="audio/wav")


@app.get("/")
def index():
    return FileResponse(_WEB / "index.html")


# static assets (app.js, etc.)
app.mount("/static", StaticFiles(directory=_WEB), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
