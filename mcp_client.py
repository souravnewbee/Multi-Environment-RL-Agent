"""
UMORDA — MCP Client Wrapper
File: mcp_client.py

Connects to mcp_local_server.py (as a subprocess, over stdio — the
standard MCP transport for local servers) and calls its tools.

Groq and Ollama don't speak MCP natively the way Claude does, so this
file plays the role of a minimal hand-written MCP client: it launches
the server, calls the tools we need, and hands back a plain Python dict
that the rest of the pipeline (llm_client.py / interactive_demo.py) can
use directly — no protocol details leak into the rest of the codebase.

Usage:
    from mcp_client import get_local_context
    context = get_local_context()
    # {
    #   "date": "2026-07-24", "time": "14:32:10", "day_of_week": "Friday",
    #   "is_weekend": False, "time_of_day_bucket": 1,
    #   "time_of_day_name": "Afternoon", "season": "Summer", "month": 7
    # }

Requirements:
    pip install mcp
"""

import asyncio
import json
import os

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


SERVER_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "mcp_local_server.py")


async def _fetch_local_context() -> dict:
    """Launches mcp_local_server.py as a subprocess and calls all 3 tools."""
    server_params = StdioServerParameters(
        command="python",
        args=[SERVER_SCRIPT],
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            dt_result     = await session.call_tool("get_current_datetime", {})
            bucket_result = await session.call_tool("get_time_of_day_bucket", {})
            season_result = await session.call_tool("get_season", {})

            # FastMCP tools returning a dict are auto-serialized to a
            # single text content block containing JSON — parse it back.
            dt_data     = json.loads(dt_result.content[0].text)
            bucket_data = json.loads(bucket_result.content[0].text)
            season_data = json.loads(season_result.content[0].text)

            return {
                "date":               dt_data["date"],
                "time":               dt_data["time"],
                "day_of_week":        dt_data["day_of_week"],
                "is_weekend":         dt_data["is_weekend"],
                "time_of_day_bucket": bucket_data["bucket"],
                "time_of_day_name":   bucket_data["bucket_name"],
                "season":             season_data["season"],
                "month":              season_data["month"],
            }


def get_local_context() -> dict:
    """
    Synchronous wrapper — call this from anywhere in the pipeline
    (llm_client.py, interactive_demo.py) without dealing with asyncio.

    Returns a plain dict of real local context. Falls back to a clearly
    marked error dict if the MCP server can't be reached, rather than
    crashing the whole pipeline.
    """
    try:
        return asyncio.run(_fetch_local_context())
    except Exception as e:
        return {
            "error": f"Could not reach MCP local server: {e}",
            "date": None, "time": None, "day_of_week": None,
            "is_weekend": None, "time_of_day_bucket": None,
            "time_of_day_name": None, "season": None, "month": None,
        }


if __name__ == "__main__":
    print("Fetching local context via MCP server...")
    ctx = get_local_context()
    for k, v in ctx.items():
        print(f"  {k:20}: {v}")
