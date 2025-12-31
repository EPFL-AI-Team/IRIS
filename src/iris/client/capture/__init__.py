"""Capture module for camera and video file input."""

from .camera import CameraCapture
from .video_file import VideoFileCapture

__all__ = ["CameraCapture", "VideoFileCapture"]
