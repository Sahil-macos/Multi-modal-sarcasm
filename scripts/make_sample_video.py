"""Build samples/sample_video.mp4: frames with an on-screen caption + the TTS audio track."""
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backend.preprocess.media import ffmpeg_exe  # noqa: E402

S = Path(__file__).resolve().parent.parent / "samples"
silent, out = S / "_silent.mp4", S / "sample_video.mp4"
w, h, fps, secs = 640, 360, 25, 5
vw = cv2.VideoWriter(str(silent), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
for i in range(fps * secs):
    f = np.full((h, w, 3), (40 + i % 60, 60, 90), np.uint8)
    cv2.circle(f, (100 + i * 3, 150), 40, (0, 200, 255), -1)
    cv2.rectangle(f, (0, 270), (w, 340), (0, 0, 0), -1)
    cv2.putText(f, "Yeah, Mondays are SO much fun", (30, 315),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    vw.write(f)
vw.release()
subprocess.run([ffmpeg_exe(), "-y", "-v", "error", "-i", str(silent), "-i", str(S / "sarcastic_speech.wav"),
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(out)], check=True)
silent.unlink()
print(out, out.stat().st_size)
