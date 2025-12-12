"""Video frame extraction for offline video processing."""

import logging
from pathlib import Path

import cv2
from PIL import Image

logger = logging.getLogger(__name__)


class VideoFrameExtractor:
    """Extract frames from video files at a target FPS."""

    def __init__(self, video_path: str | Path, target_fps: float = 5.0):
        """Initialize video frame extractor.

        Args:
            video_path: Path to video file
            target_fps: Target frames per second for extraction
        """
        self.video_path = Path(video_path)
        self.target_fps = target_fps
        self.cap = None

    def extract_frames(self) -> list[Image.Image]:
        """Extract frames from video at target FPS.

        Returns:
            List of PIL Images extracted from video
        """
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file not found: {self.video_path}")

        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            raise ValueError(f"Failed to open video: {self.video_path}")

        # Get video properties
        video_fps = self.cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if video_fps <= 0:
            logger.warning(f"Invalid FPS {video_fps}, using default 30")
            video_fps = 30.0

        # Calculate frame interval for target FPS
        frame_interval = max(1, int(video_fps / self.target_fps))

        logger.info(
            f"Extracting from {self.video_path.name}: "
            f"video_fps={video_fps:.2f}, target_fps={self.target_fps:.2f}, "
            f"interval={frame_interval}, total_frames={total_frames}"
        )

        frames = []
        frame_count = 0

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            # Extract every Nth frame based on interval
            if frame_count % frame_interval == 0:
                # Convert BGR (OpenCV) to RGB (PIL)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                frames.append(pil_image)

            frame_count += 1

        self.cap.release()

        logger.info(f"Extracted {len(frames)} frames from {frame_count} total frames")
        return frames

    def get_video_info(self) -> dict:
        """Get video metadata.

        Returns:
            Dictionary with video properties
        """
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file not found: {self.video_path}")

        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {self.video_path}")

        info = {
            "filename": self.video_path.name,
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "duration_seconds": (
                cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
                if cap.get(cv2.CAP_PROP_FPS) > 0
                else 0
            ),
        }

        cap.release()
        return info
