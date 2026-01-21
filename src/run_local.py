"""Local runner for the CodeImprovingAgent.

Use this for running the agent *without* `langgraph dev`, e.g.:

  python -m src.run_local --problem "..." --max-iterations 2

This runner compiles with MemorySaver so checkpoints are available in-process.
"""

import argparse
import asyncio
import logging
import os
import sys
from dotenv import load_dotenv
import uuid

from langgraph.checkpoint.memory import MemorySaver

from .config import (
    LOG_LEVEL,
    LOG_FORMAT,
    ANTHROPIC_API_KEY_ENV_VAR,
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_IMPROVEMENT_PROMPT,
)
from .llm_utils import setup_anthropic_client
from .agent import CodeImprovingAgent


def _load_env():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    dotenv_path = os.path.join(project_root, ".env")
    if os.path.exists(dotenv_path):
        print(f"INFO: Loading environment variables from: {dotenv_path}")
        load_dotenv(dotenv_path=dotenv_path)
    else:
        print("INFO: .env file not found, relying on system environment variables.")


def _parse_args():
    p = argparse.ArgumentParser(description="Run the CodeImprovingAgent locally (no langgraph dev).")
    p.add_argument("--problem", "-p", required=True, help="Problem description for the agent.")
    p.add_argument(
        "--max-iterations",
        "-n",
        type=int,
        default=DEFAULT_MAX_ITERATIONS,
        help=f"Max improvement iterations (default: {DEFAULT_MAX_ITERATIONS}).",
    )
    p.add_argument(
        "--improvement-prompt",
        "-i",
        default=DEFAULT_IMPROVEMENT_PROMPT,
        help="Prompt used for requesting improvements from the model.",
    )
    p.add_argument(
        "--thread-id",
        "-t",
        default=None,
        help="Optional thread_id for checkpointing / resuming within the same process.",
    )
    return p.parse_args()


async def _amain():
    args = _parse_args()

    if args.thread_id is None:
        args.thread_id = f"local-{uuid.uuid4()}"

    if not os.getenv(ANTHROPIC_API_KEY_ENV_VAR):
        raise ValueError(f"Setup failed: {ANTHROPIC_API_KEY_ENV_VAR} not set.")

    logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT, stream=sys.stdout)
    logger = logging.getLogger(__name__)

    logger.info("Setting up LLM client...")
    llm_client = setup_anthropic_client()

    logger.info("Initializing Agent...")
    agent = CodeImprovingAgent(llm_client=llm_client)

    logger.info("Compiling Agent Graph with MemorySaver (local run)...")
    agent.compile(checkpointer=MemorySaver())

    logger.info("Starting local run...")
    await agent.run(
        problem_description=args.problem,
        max_iterations=args.max_iterations,
        improvement_prompt=args.improvement_prompt,
        thread_id=args.thread_id,
    )


def main():
    _load_env()
    try:
        asyncio.run(_amain())
    except KeyboardInterrupt:
        print("\nInterrupted.")
    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
