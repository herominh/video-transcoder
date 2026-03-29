"""Docker/GCR/self-hosted entry point — runs the FastAPI app with uvicorn."""

import logging
import os
import sys

# Ensure the project root is on the path so `core` package is importable.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))

    uvicorn.run("core.api:app", host=host, port=port, log_level="info")
