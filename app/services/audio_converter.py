import subprocess
import imageio_ffmpeg

# Safety net independent of the /predict route's own overall timeout: subprocess.run's
# timeout= actually terminates the ffmpeg child process on expiry (unlike
# asyncio.wait_for around an asyncio.to_thread call upstream, which can only stop US
# from waiting -- it has no way to kill the thread/process actually doing the work).
FFMPEG_TIMEOUT_SECONDS = 30


def convert_to_wav(input_path: str, output_path: str) -> None:
    """Decodes any input audio container/codec to a plain PCM WAV, WITHOUT
    resampling or downmixing.

    ffmpeg here only does what librosa/soundfile can't: unwrap webm/opus/m4a/aac/
    wma/amr/... into linear PCM. Sample-rate conversion and stereo->mono are left
    to librosa downstream (load_clean_audio / the per-model loaders in
    ensemble.py), so the audio the models see is produced by the exact same
    resampler (librosa's soxr_hq) that car_test/preprocessing.py used to build the
    training features.

    Why this matters: ffmpeg's default resampler is lower quality than librosa's,
    and forcing `-ar 22050` here put a ~0.2% distortion into every clip. The
    deployed XGBoost model (shallow max_depth=3 trees) is sensitive enough near
    its split thresholds that this flipped confident correct predictions -- e.g. a
    brake clip scoring brake 0.999 direct dropped to belt 0.52 after `-ar 22050`.
    Letting librosa do the single 44.1k->22.05k step restores the 0.999.
    """
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg_exe, "-y",
        "-i", input_path,
        output_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=FFMPEG_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"ffmpeg conversion did not finish within {FFMPEG_TIMEOUT_SECONDS}s.")
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg conversion failed: {result.stderr.decode(errors='ignore')}")
