"""Resolve the model weights + voicepack, downloading them from Hugging Face on first use.

The weights (~313 MB) are too large for git, so they live in a Hugging Face model repo and
are fetched + cached on demand. Pass explicit local paths to ``TTS(...)`` (or set
``TTS_EU_PT_REPO``) to run fully offline.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

# The Hugging Face repo that hosts the weights. Override per-call via ``TTS(repo_id=...)``
# or globally via the TTS_EU_PT_REPO environment variable.
DEFAULT_REPO_ID = "CHANGEME/tts_eu_pt"
MODEL_FILENAME = "tts_eu_pt.pth"
VOICEPACK_FILENAME = "voicepack.pt"


def ensure_weights(
    model_path: Optional[str] = None,
    voicepack_path: Optional[str] = None,
    repo_id: Optional[str] = None,
) -> Tuple[Path, Path]:
    """Return (model_path, voicepack_path), downloading whichever is not supplied locally."""
    mp: Optional[Path] = Path(model_path) if model_path else None
    vp: Optional[Path] = Path(voicepack_path) if voicepack_path else None
    if mp and vp and mp.exists() and vp.exists():
        return mp, vp

    repo = repo_id or os.environ.get("TTS_EU_PT_REPO") or DEFAULT_REPO_ID
    if (mp is None or not mp.exists()) or (vp is None or not vp.exists()):
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "huggingface_hub is required to download the weights. Install it, or pass "
                "model_path= and voicepack_path= to TTS(...) to run offline."
            ) from e
        if mp is None or not mp.exists():
            mp = Path(hf_hub_download(repo, MODEL_FILENAME))
        if vp is None or not vp.exists():
            vp = Path(hf_hub_download(repo, VOICEPACK_FILENAME))
    return mp, vp
