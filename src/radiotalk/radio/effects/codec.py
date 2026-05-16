from __future__ import annotations

import io

import numpy as np
import soundfile as sf


def apply_codec_roundtrip(
    samples: np.ndarray,
    sample_rate: int,
    codec: str,
) -> np.ndarray:
    """Encode then decode through a lossy codec to mimic LiveATC-style ingest.

    Supported codecs: ``ogg`` (Vorbis), ``flac`` (lossless, identity check).
    ``mp3`` and ``gsm`` are not in soundfile's default build; use pedalboard
    GSMFullRateCompressor / pydub if needed at training time.
    """
    if codec == "ogg":
        return _roundtrip_via_soundfile(samples, sample_rate, "OGG", "VORBIS")
    if codec == "flac":
        return _roundtrip_via_soundfile(samples, sample_rate, "FLAC", "PCM_16")
    raise ValueError(f"unsupported codec: {codec!r}")


def _roundtrip_via_soundfile(
    samples: np.ndarray,
    sample_rate: int,
    fmt: str,
    subtype: str,
) -> np.ndarray:
    buf = io.BytesIO()
    sf.write(buf, samples.astype(np.float32, copy=False), sample_rate,
             format=fmt, subtype=subtype)
    buf.seek(0)
    decoded, _ = sf.read(buf, dtype="float32", always_2d=False)
    if decoded.ndim > 1:
        decoded = decoded.mean(axis=-1)
    if decoded.shape[0] > samples.shape[0]:
        decoded = decoded[: samples.shape[0]]
    elif decoded.shape[0] < samples.shape[0]:
        pad = np.zeros(samples.shape[0] - decoded.shape[0], dtype=np.float32)
        decoded = np.concatenate([decoded, pad])
    return decoded.astype(np.float32, copy=False)
