"""Module for creating an asynchronous camera capture thread"""

import cv2
import numpy as np
from threaded_videocapture import ThreadedVideoCapture


class CameraCapture:
    """Asynchronous camera capture class for video streaming.

    Provides threaded video capture with configurable resolution and FPS.
    """

    def __init__(
        self, camera_index: int = 0, width: int = 640, height: int = 480, fps: int = 30
    ):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.fps = fps
        self.cap = None

    def start(self) -> bool:
        """Start the camera capture with configured settings.

        Returns:
            bool: True if camera started successfully, False otherwise.
        """
        self.cap = ThreadedVideoCapture(self.camera_index)
        if not self.cap.isOpened():
            return False

        # Set properties - pylint: disable=no-member
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        return True

    def stop(self) -> None:
        """Stop the camera capture and release resources."""
        if self.cap:
            self.cap.release()

    def get_frame(self) -> np.ndarray | None:
        """Capture a single frame from the camera.

        Returns:
            Optional[np.ndarray]: The captured frame, or None if capture failed.
        """
        if self.cap is None:
            return None
        ret, frame = self.cap.read()
        return frame if ret else None

    def get_frame_jpeg(self, quality: int = 80) -> bytes | None:
        """Capture a frame and encode it as JPEG.

        Args:
            quality: JPEG compression quality (0-100).

        Returns:
            Optional[bytes]: JPEG-encoded frame bytes, or None if capture failed.
        """
        frame = self.get_frame()
        if frame is None:
            return None

        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]  # pylint: disable=no-member
        _, buffer = cv2.imencode(".jpg", frame, encode_param)  # pylint: disable=no-member
        return buffer.tobytes()
