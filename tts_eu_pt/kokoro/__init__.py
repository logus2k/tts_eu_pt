"""Vendored Kokoro acoustic model (Apache-2.0, from hexgrad/kokoro).

Only the acoustic ``KModel`` is exposed. The upstream ``KPipeline`` is deliberately NOT
imported here: it pulls in ``misaki`` for grapheme-to-phoneme, which relies on GPLv3
espeak-ng. ``tts_eu_pt`` ships its own permissively-licensed European-Portuguese G2P
(``tts_eu_pt.g2p``, built on Apache-2.0 TugaPhone) instead, so nothing GPL is required at
runtime. See ATTRIBUTIONS.md.
"""
from .model import KModel

__all__ = ["KModel"]
