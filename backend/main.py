"""
Main entry point for the Microcontroller API Assistant backend.
"""

import uvicorn
from loguru import logger

from app.config import settings
from app.api import app


def main():
    """Main function to run the FastAPI application."""
    logger.info(f"Starting {settings.api_title} v{settings.api_version}")
    logger.info(f"Server will run on {settings.api_host}:{settings.api_port}")
    
    uvicorn.run(
        "app.api:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
        log_level=settings.log_level.lower(),
        access_log=True,
    )


if __name__ == "__main__":
    main()
