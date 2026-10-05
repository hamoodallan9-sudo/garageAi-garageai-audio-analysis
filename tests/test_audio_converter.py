"""Verifies ffmpeg conversion actually produces a valid, non-empty WAV for every
format the upload pipeline is supposed to accept -- this is the safety net for the
format-expansion work (AMR/FLAC/M4A/OGG/WMA support).

convert_to_wav deliberately no longer resamples or downmixes (librosa does that
downstream, matching the training pipeline's resampler), so these only assert a
decodable PCM WAV comes out, not a particular sample rate or channel count."""

import subprocess

import numpy as np
import pytest
import soundfile as sf

from app.services.audio_converter import convert_to_wav


def _ffmpeg_exe():
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def _make_source_wav(tmp_path, duration_sec=2.0, sr=22050):
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
    y = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    path = tmp_path / "source.wav"
    sf.write(path, y, sr)
    return path


def _encode(tmp_path, src_wav, ext, codec_args):
    out_path = tmp_path / f"encoded{ext}"
    cmd = [_ffmpeg_exe(), "-y", "-i", str(src_wav), *codec_args, str(out_path)]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        pytest.skip(f"ffmpeg could not encode {ext} in this environment: {result.stderr.decode(errors='ignore')[:300]}")
    return out_path


@pytest.mark.parametrize(
    "ext, codec_args",
    [
        (".flac", []),
        (".m4a", ["-c:a", "aac"]),
        (".ogg", ["-c:a", "libvorbis"]),
        (".wma", ["-c:a", "wmav2"]),
        (".amr", ["-ar", "8000", "-ac", "1"]),
    ],
)
def test_converts_every_supported_format_to_a_valid_wav(tmp_path, ext, codec_args):
    source = _make_source_wav(tmp_path)
    encoded = _encode(tmp_path, source, ext, codec_args)

    out_wav = tmp_path / "converted.wav"
    convert_to_wav(str(encoded), str(out_wav))

    assert out_wav.exists()
    data, sr = sf.read(out_wav)
    assert sr > 0
    assert len(data) > 0


def test_raises_on_a_nonexistent_input_file(tmp_path):
    with pytest.raises(RuntimeError):
        convert_to_wav(str(tmp_path / "does_not_exist.wav"), str(tmp_path / "out.wav"))
