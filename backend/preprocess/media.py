"""Low-level media helpers: ffmpeg discovery, audio decoding, frame sampling."""
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000


@lru_cache(maxsize=1)
def ffmpeg_exe() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "ffmpeg not found. Install it or `pip install imageio-ffmpeg`."
        ) from e


def load_audio(path, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Decode any audio/video file to mono float32 PCM at `sr` Hz."""
    proc = subprocess.run(
        [ffmpeg_exe(), "-nostdin", "-v", "error", "-i", str(path),
         "-vn", "-f", "f32le", "-ac", "1", "-ar", str(sr), "-"],
        capture_output=True,
    )
    if proc.returncode != 0:
        err = proc.stderr.decode(errors="ignore").strip()
        if "does not contain any stream" in err or "matches no streams" in err:
            return np.zeros(0, dtype=np.float32)
        raise ValueError(f"ffmpeg failed to decode audio: {err[:300]}")
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()


def video_info(path) -> dict:
    import cv2
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError("Could not open video file")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    return {"frames": n, "fps": fps, "width": w, "height": h,
            "duration_s": round(n / fps, 2) if fps else None}


def sample_frames(path, num: int) -> list:
    """Uniformly sample `num` RGB frames (H, W, 3 uint8) from a video."""
    import cv2
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError("Could not open video file")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frames = []
    if total > 0:
        idxs = np.linspace(0, max(total - 1, 0), num).round().astype(int)
        for i in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, f = cap.read()
            if ok:
                frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    if not frames:  # some containers report no frame count: read sequentially
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        allf = []
        while True:
            ok, f = cap.read()
            if not ok:
                break
            allf.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        if allf:
            idxs = np.linspace(0, len(allf) - 1, num).round().astype(int)
            frames = [allf[i] for i in idxs]
    cap.release()
    if not frames:
        raise ValueError("No frames could be decoded from the video")
    while len(frames) < num:  # pad very short clips
        frames.append(frames[-1])
    return frames


def suffix(path) -> str:
    return Path(path).suffix.lower()
