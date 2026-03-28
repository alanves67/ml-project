#!/usr/bin/env python
import uvicorn
from src.utils.logger import setup_logger

logger = setup_logger("main")

if __name__ == "__main__":
    logger.info("Starting ML API server...")
    uvicorn.run(
        "src.api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )