"""Resolve the model weights + voicepack, downloading them from Hugging Face on first use.

The weights (~313 MB) are too large for git, so they live in a Hugging Face model repo and
are fetched + cached on demand. Pass explicit local paths to ``TTS(...)`` (or set
``TTS_EU_PT_REPO``) to run fully offline.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

# The Hugging Face MODEL repo that hosts the weights. Override per-call via
# ``TTS(repo_id=...)`` or globally via the TTS_EU_PT_REPO environment variable.
DEFAULT_REPO_ID = "logus2k/kokoro_tts_eu_pt"
_PLACEHOLDER_OWNER = "CHANGEME"

# These must match the filenames IN the repo above, EXACTLY. A mismatch surfaces as
# RemoteEntryNotFoundError (a 404 on .../resolve/main/<name>) rather than as a missing
# repo, so check these two first if a download 404s while the repo itself resolves.
MODEL_FILENAME = "tuga_kokoro.pth"
VOICEPACK_FILENAME = "tuga_voicepack.pt"


def _check_placeholder(repo: str) -> None:
    """Fail with instructions rather than a bare 404 if pointed at an unpublished repo."""
    if repo.split("/", 1)[0] != _PLACEHOLDER_OWNER:
        return
    raise RuntimeError(
        f"{repo!r} is a placeholder, so there is nothing to download.\n"
        "Run it with local weights instead, either:\n"
        "  1. pass explicit paths:\n"
        "       TTS(model_path='/path/model.pth', voicepack_path='/path/voicepack.pt')\n"
        "  2. or point at a Hugging Face model repo that hosts them:\n"
        "       export TTS_EU_PT_REPO='your-org/your-repo'\n"
        "     (or per call: TTS(repo_id='your-org/your-repo'))"
    )


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
        _check_placeholder(repo)
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
