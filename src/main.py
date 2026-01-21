"""LangGraph dev entrypoint.

This module is meant to be *imported* by `langgraph dev`, which expects a
top-level variable named `app` (or `graph`) containing a compiled graph.

IMPORTANT:
- Do NOT pass a custom checkpointer here. LangGraph API/server mode owns persistence.
- Keep this file side-effect light: it should mainly build/compile and expose `app`.
"""

import logging
import os
import sys
from dotenv import load_dotenv

from .config import LOG_LEVEL, LOG_FORMAT, ANTHROPIC_API_KEY_ENV_VAR
from .llm_utils import setup_anthropic_client
from .agent import CodeImprovingAgent


# --- Load .env (same behavior as your original main.py) ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
dotenv_path = os.path.join(project_root, ".env")
if os.path.exists(dotenv_path):
    print(f"INFO: Loading environment variables from: {dotenv_path}")
    load_dotenv(dotenv_path=dotenv_path)
else:
    print("INFO: .env file not found, relying on system environment variables.")


# --- Logging ---
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)


def _build_app():
    """Build and compile the graph for LangGraph API/server mode."""
    if not os.getenv(ANTHROPIC_API_KEY_ENV_VAR):
        raise ValueError(f"Setup failed: {ANTHROPIC_API_KEY_ENV_VAR} not set.")

    logger.info("Setting up LLM client...")
    llm_client = setup_anthropic_client()

    logger.info("Initializing Agent...")
    agent = CodeImprovingAgent(llm_client=llm_client)

    logger.info("Compiling Agent Graph (NO custom checkpointer for langgraph dev)...")
    agent.compile()  # <-- critical: do not pass MemorySaver/other custom checkpointer here

    if agent.graph is None:
        raise RuntimeError("Agent compilation failed: compiled graph is None.")

    logger.info("Compiled graph 'app' is ready for LangGraph server.")
    return agent.graph


# Exported variable that `langgraph dev` will import.
# If build fails, raise a clear exception (avoid sys.exit during import).
app = _build_app()
