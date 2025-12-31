"""Module for webserver hosted on the client"""

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from iris.client.web.database import init_db
from iris.client.web.routes import api_router, ws_router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan handler for startup/shutdown."""
    # Startup
    logger.info("Initializing SQLite database...")
    init_db()
    logger.info("Database initialized")

    yield

    # Shutdown
    logger.info("Client web server shutting down")


app = FastAPI(title="IRIS Streaming client", lifespan=lifespan)

app.include_router(api_router)
app.include_router(ws_router)

# Mount static files directory (for videos)
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)  # Ensure directory exists
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Mount frontend files (must be last to catch all other routes)
frontend_dist = Path(__file__).parent / "frontend" / "dist"
app.mount("/", StaticFiles(directory=str(frontend_dist), html=True), name="frontend")




def main() -> None:
    import uvicorn

    from iris.client.web.dependencies import get_app_state

    # Get configuration from app state
    state = get_app_state()
    web_config = state.config.web

    # Check if SSL is enabled and certificates exist
    if (
        web_config.use_ssl
        and web_config.ssl_keyfile.exists()
        and web_config.ssl_certfile.exists()
    ):
        print(f"Using certificates from {web_config.cert_path}")
        uvicorn.run(
            "iris.client.web.app:app",
            host=web_config.host,
            port=web_config.port,
            reload=True,
            log_level="warning",
            ssl_keyfile=str(web_config.ssl_keyfile),
            ssl_certfile=str(web_config.ssl_certfile),
        )
        print(f"Started HTTPS server on {web_config.host}:{web_config.port}")
    else:
        if web_config.use_ssl:
            print(f"SSL enabled but certificates not found at: {web_config.cert_path}")
            print("Falling back to HTTP server")
        uvicorn.run(
            "iris.client.web.app:app",
            host=web_config.host,
            port=web_config.port,
            reload=True,
            log_level="warning",
        )
        print(f"Started HTTP server on {web_config.host}:{web_config.port}")


if __name__ == "__main__":
    main()
