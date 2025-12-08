"""Module for webserver hosted on the client"""

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from iris.client.web import routes

app = FastAPI(title="IRIS Streaming client")

static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

app.include_router(routes.router)


def main() -> None:
    import uvicorn

    from iris.client.web.dependencies import get_app_state

    # Get configuration from app state
    state = get_app_state()
    web_config = state.config.web

    # Check if SSL is enabled and certificates exist
    if web_config.use_ssl and web_config.ssl_keyfile.exists() and web_config.ssl_certfile.exists():
        print(f"Starting HTTPS server on {web_config.host}:{web_config.port}")
        print(f"Using certificates from {web_config.cert_path}")
        uvicorn.run(
            "iris.client.web.app:app",
            host=web_config.host,
            port=web_config.port,
            reload=True,
            ssl_keyfile=str(web_config.ssl_keyfile),
            ssl_certfile=str(web_config.ssl_certfile),
        )
    else:
        if web_config.use_ssl:
            print(f"SSL enabled but certificates not found at: {web_config.cert_path}")
            print("Falling back to HTTP server")
        print(f"Starting HTTP server on {web_config.host}:{web_config.port}")
        uvicorn.run(
            "iris.client.web.app:app",
            host=web_config.host,
            port=web_config.port,
            reload=True,
        )


if __name__ == "__main__":
    main()
