"""Models for all configurations in the client"""

from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

from iris.config import _yaml_config


class VideoConfig(BaseModel):
    """Video capture configuration."""

    width: int = Field(
        default=_yaml_config.get("client", {}).get("video", {}).get("width", 640),
        ge=320,
        le=1920,
    )
    height: int = Field(
        default=_yaml_config.get("client", {}).get("video", {}).get("height", 480),
        ge=240,
        le=1080,
    )
    capture_fps: float = Field(
        default=_yaml_config.get("client", {}).get("video", {}).get("capture_fps", 10),
        ge=1,
        le=30,
    )
    jpeg_quality: int = Field(
        default=_yaml_config.get("client", {}).get("video", {}).get("jpeg_quality", 80),
        ge=10,
        le=100,
    )
    camera_index: int = Field(
        default=_yaml_config.get("client", {}).get("video", {}).get("camera_index", 0),
        ge=0,
    )


class ServerConfig(BaseSettings):
    """Target server configuration."""

    host: str = Field(
        default=_yaml_config.get("client", {})
        .get("server", {})
        .get("host", "localhost")
    )
    port: int = Field(
        default=_yaml_config.get("client", {}).get("server", {}).get("port", 8001),
        ge=1024,
        le=65535,
    )
    endpoint: str = Field(
        default=_yaml_config.get("client", {})
        .get("server", {})
        .get("endpoint", "/ws/stream")
    )
    use_ssl: bool = Field(
        default=_yaml_config.get("client", {}).get("server", {}).get("use_ssl", False)
    )

    @property
    def ws_url(self) -> str:
        """Returns full ip endpoint for target server"""
        protocol = "wss" if self.use_ssl else "ws"
        return f"{protocol}://{self.host}:{self.port}{self.endpoint}"


class WebConfig(BaseModel):
    """Web server configuration for the client UI."""

    host: str = Field(
        default=_yaml_config.get("client", {}).get("web", {}).get("host", "0.0.0.0")
    )
    port: int = Field(
        default=_yaml_config.get("client", {}).get("web", {}).get("port", 8006),
        ge=1024,
        le=65535,
    )
    use_ssl: bool = Field(
        default=_yaml_config.get("client", {}).get("web", {}).get("use_ssl", False)
    )
    cert_dir: str = Field(
        default=_yaml_config.get("client", {})
        .get("web", {})
        .get("cert_dir", "~/iris-certs")
    )

    @property
    def cert_path(self) -> Path:
        """Returns expanded path to certificate directory."""
        return Path(self.cert_dir).expanduser()

    @property
    def ssl_keyfile(self) -> Path:
        """Returns path to SSL key file."""
        return self.cert_path / "key.pem"

    @property
    def ssl_certfile(self) -> Path:
        """Returns path to SSL certificate file."""
        return self.cert_path / "cert.pem"


class ClientConfig(BaseModel):
    """Complete client configuration."""

    video: VideoConfig = Field(default_factory=VideoConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    web: WebConfig = Field(default_factory=WebConfig)
