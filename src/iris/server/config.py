from pydantic import BaseModel, Field


class ServerConfig(BaseModel):
    """Server inference configuration."""

    model_key: str = Field(default="smolvlm2")
    max_queue_size: int = Field(default=10)
    num_workers: int = Field(default=1)
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8001)
