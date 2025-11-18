"""Models for all configurations in the client"""

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class VideoConfig(BaseModel):
    """Video capture configuration."""

    width: int = Field(default=640, ge=320, le=1920)
    height: int = Field(default=480, ge=240, le=1080)
    fps: int = Field(default=10, ge=1, le=30)
    jpeg_quality: int = Field(default=80, ge=10, le=100)
    camera_index: int = Field(default=0, ge=0)


class ServerConfig(BaseSettings):
    """Target server configuration."""

    host: str = Field(default="localhost")
    port: int = Field(default=8001, ge=1024, le=65535)
    endpoint: str = Field(default="/ws/stream")

    @property
    def ws_url(self) -> str:
        """Returns full ip endpoint for target server"""
        return f"ws://{self.host}:{self.port}{self.endpoint}"


class ClientConfig(BaseModel):
    """Complete client configuration."""

    video: VideoConfig = Field(default_factory=VideoConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
