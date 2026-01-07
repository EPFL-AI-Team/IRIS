"""Module for video file capture"""

import cv2
import numpy as np


class VideoFileCapture:
    """Video file capture class for processing pre-recorded videos.

    Provides sequential video reading with configurable simulation speed.
    Mirrors the CameraCapture interface but reads from a video file instead of a camera.
    """

    def __init__(
        self,
        video_path: str,
        width: int = 640,
        height: int = 480,
        simulation_fps: float = 5.0,
    ):
        """Initialize VideoFileCapture.

        Args:
            video_path: Path to the video file.
            width: Target frame width.
            height: Target frame height.
            simulation_fps: Playback speed in frames per second (for throttling).
        """
        self.video_path = video_path
        self.width = width
        self.height = height
        self.simulation_fps = float(simulation_fps)
        self.fps = float(simulation_fps)  # Alias for compatibility with StreamingClient
        self.cap = None
        self.total_frames = 0
        self.current_frame_number = 0
        self.native_fps = 0.0
        self.is_finished = False

    def start(self) -> bool:
        """Start the video file capture and read metadata.

        Returns:
            bool: True if video opened successfully, False otherwise.
        """
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            return False

        # Read video metadata
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.native_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.current_frame_number = 0
        self.is_finished = False

        return True

    def stop(self) -> None:
        """Stop the video capture and release resources."""
        if self.cap:
            self.cap.release()
            self.cap = None
        self.is_finished = True

    def get_frame(self) -> np.ndarray | None:
        """Read the next frame from the video file.

        Returns:
            Optional[np.ndarray]: The frame, or None if finished or read failed.
        """
        if self.cap is None or self.is_finished:
            return None

        ret, frame = self.cap.read()
        if not ret:
            self.is_finished = True
            return None

        self.current_frame_number += 1
        return frame

    def _crop_to_aspect_ratio(self, frame: np.ndarray) -> np.ndarray:
        """Crop frame center to match configured aspect ratio.

        Args:
            frame: Input frame to crop.

        Returns:
            Cropped frame preserving aspect ratio.
        """
        h, w = frame.shape[:2]
        target_ar = self.width / self.height
        source_ar = w / h

        # If aspect ratios are close enough (within 1%), skip crop
        if abs(target_ar - source_ar) < 0.01:
            return frame

        if source_ar > target_ar:
            # Source is wider - crop width (sides)
            new_w = int(h * target_ar)
            start_x = (w - new_w) // 2
            return frame[:, start_x : start_x + new_w]
        else:
            # Source is taller - crop height (top/bottom)
            new_h = int(w / target_ar)
            start_y = (h - new_h) // 2
            return frame[start_y : start_y + new_h, :]

    def get_frame_jpeg(self, quality: int = 80) -> bytes | None:
        """Read a frame and encode it as JPEG.

        Args:
            quality: JPEG compression quality (0-100).

        Returns:
            Optional[bytes]: JPEG-encoded frame bytes, or None if capture failed.
        """
        frame = self.get_frame()
        if frame is None:
            return None

        # Crop and resize frame if it doesn't match configured dimensions
        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            frame = self._crop_to_aspect_ratio(
                frame
            )  # Crop first to preserve aspect ratio
            frame = cv2.resize(frame, (self.width, self.height))  # Then resize

        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]  # pylint: disable=no-member
        _, buffer = cv2.imencode(".jpg", frame, encode_param)  # pylint: disable=no-member
        return buffer.tobytes()

    def seek(self, frame_number: int) -> bool:
        """Seek to a specific frame in the video.

        Args:
            frame_number: The frame number to seek to (0-indexed).

        Returns:
            bool: True if seek succeeded, False otherwise.
        """
        if self.cap is None:
            return False

        if frame_number < 0 or frame_number >= self.total_frames:
            return False

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        self.current_frame_number = frame_number
        self.is_finished = False
        return True

    def get_position_ms(self) -> float:
        """Get current playback position in milliseconds.

        Returns:
            float: Current position in milliseconds.
        """
        if self.native_fps <= 0:
            return 0.0
        return (self.current_frame_number / self.native_fps) * 1000.0

    def get_duration_ms(self) -> float:
        """Get total video duration in milliseconds.

        Returns:
            float: Total duration in milliseconds.
        """
        if self.native_fps <= 0:
            return 0.0
        return (self.total_frames / self.native_fps) * 1000.0
