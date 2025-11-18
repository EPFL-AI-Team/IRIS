#!/usr/bin/env python3
"""
Main entry point for IRIS streaming client.
Runs FastAPI app on localhost:8080.
"""

import uvicorn


def main():
    """Run the client application."""
    uvicorn.run(
        "iris.client.web.app:app",
        host="0.0.0.0",  # Allow access from network
        port=8080,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
