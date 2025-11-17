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

    uvicorn.run("iris.client.web.app:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main()
