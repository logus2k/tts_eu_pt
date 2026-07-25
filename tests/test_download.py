"""Weight-resolution tests. No network and no model weights needed.

Two things are guarded here:
  * a placeholder repo id must raise an actionable RuntimeError BEFORE any network call,
    rather than the bare HTTP 404 (RepositoryNotFoundError) it used to die with;
  * the shipped DEFAULT_REPO_ID must be a real, configured repo -- a release guard, so the
    package can never go out still pointing at CHANGEME.
"""
import pytest

from tts_eu_pt import download
from tts_eu_pt.download import ensure_weights


def test_local_paths_skip_the_download_entirely(tmp_path, monkeypatch):
    mp = tmp_path / "m.pth"
    vp = tmp_path / "v.pt"
    mp.write_bytes(b"x")
    vp.write_bytes(b"y")

    def boom(*a, **k):                       # any network access is a failure here
        raise AssertionError("hf_hub_download must not be called when both paths exist")

    monkeypatch.setattr("huggingface_hub.hf_hub_download", boom)
    assert ensure_weights(str(mp), str(vp)) == (mp, vp)


def test_placeholder_repo_raises_actionable_error(monkeypatch):
    monkeypatch.delenv("TTS_EU_PT_REPO", raising=False)

    def boom(*a, **k):
        raise AssertionError("must fail before any network call")

    monkeypatch.setattr("huggingface_hub.hf_hub_download", boom)
    with pytest.raises(RuntimeError) as e:
        ensure_weights(repo_id=f"{download._PLACEHOLDER_OWNER}/tts_eu_pt")
    msg = str(e.value)
    assert "placeholder" in msg
    assert "model_path" in msg and "TTS_EU_PT_REPO" in msg


def test_shipped_default_repo_is_not_a_placeholder():
    """Release guard: the package must never ship pointing at CHANGEME."""
    owner, _, name = download.DEFAULT_REPO_ID.partition("/")
    assert owner != download._PLACEHOLDER_OWNER, "DEFAULT_REPO_ID is still a placeholder"
    assert owner and name, f"DEFAULT_REPO_ID is not 'owner/name': {download.DEFAULT_REPO_ID!r}"


def test_weight_filenames_are_configured():
    """Both artefacts must be named; a mismatch with the repo surfaces at download time."""
    assert download.MODEL_FILENAME.endswith(".pth")
    assert download.VOICEPACK_FILENAME.endswith(".pt")


def test_placeholder_check_is_skipped_for_a_real_repo(monkeypatch):
    """A configured repo must reach the downloader rather than hitting the guard."""
    seen = []
    monkeypatch.setattr(download, "hf_hub_download", None, raising=False)
    monkeypatch.setattr(
        "huggingface_hub.hf_hub_download",
        lambda repo, filename, *a, **k: seen.append((repo, filename)) or f"/tmp/{filename}",
    )
    ensure_weights(repo_id="some-org/tts_eu_pt")
    assert seen == [
        ("some-org/tts_eu_pt", download.MODEL_FILENAME),
        ("some-org/tts_eu_pt", download.VOICEPACK_FILENAME),
    ]


def test_env_var_overrides_the_placeholder(monkeypatch):
    monkeypatch.setenv("TTS_EU_PT_REPO", "env-org/tts_eu_pt")
    seen = []
    monkeypatch.setattr(
        "huggingface_hub.hf_hub_download",
        lambda repo, filename, *a, **k: seen.append(repo) or f"/tmp/{filename}",
    )
    ensure_weights()
    assert seen == ["env-org/tts_eu_pt", "env-org/tts_eu_pt"]
