"""
UMORDA — Local Data MCP Server
File: mcp_local_server.py

Exposes REAL local machine data as MCP tools, so the LLM extractor no
longer has to guess things like `time_of_day` or seasonal context from
vague user language — it gets ground-truth values instead.

Tools exposed:
  1. get_current_datetime()   -> real date, time, day of week, weekend flag
  2. get_time_of_day_bucket() -> 0-3 bucket matching energy_env.py's
                                  time_of_day scale (0=morning,1=afternoon,
                                  2=evening,3=night), derived from the
                                  REAL system clock
  3. get_season()             -> current season + month, useful for
                                  agriculture's planting-window logic

Run standalone to test:
    python mcp_local_server.py

This starts the server on stdio, ready for an MCP client (see
mcp_client.py) to connect and call these tools.

Requirements:
    pip install mcp
"""

from datetime import datetime
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("UMORDA Local Data Server")


# ─────────────────────────────────────────────────────────────────────────
# Tool 1 — Real current date/time
# ─────────────────────────────────────────────────────────────────────────
@mcp.tool()
def get_current_datetime() -> dict:
    """
    Returns the real current local date and time from the system clock.

    Returns
    -------
    dict with keys:
        iso_datetime : full ISO-format timestamp
        date         : YYYY-MM-DD
        time         : HH:MM:SS (24-hour)
        day_of_week  : e.g. "Monday"
        is_weekend   : bool (True for Saturday/Sunday)
    """
    now = datetime.now()
    return {
        "iso_datetime": now.isoformat(),
        "date":         now.strftime("%Y-%m-%d"),
        "time":         now.strftime("%H:%M:%S"),
        "day_of_week":  now.strftime("%A"),
        "is_weekend":   now.weekday() >= 5,
    }


# ─────────────────────────────────────────────────────────────────────────
# Tool 2 — Time-of-day bucket (matches energy_env.py's time_of_day scale)
# ─────────────────────────────────────────────────────────────────────────
@mcp.tool()
def get_time_of_day_bucket() -> dict:
    """
    Maps the REAL current hour to the same 0-3 scale used by
    EnergyEnv's solar_scheduling task:
        0 = Morning   (06:00-11:59)
        1 = Afternoon (12:00-16:59)
        2 = Evening   (17:00-20:59)
        3 = Night     (21:00-05:59)

    Returns
    -------
    dict with keys:
        hour        : real current hour (0-23)
        bucket      : int 0-3
        bucket_name : "Morning" / "Afternoon" / "Evening" / "Night"
    """
    hour = datetime.now().hour

    if 6 <= hour < 12:
        bucket, name = 0, "Morning"
    elif 12 <= hour < 17:
        bucket, name = 1, "Afternoon"
    elif 17 <= hour < 21:
        bucket, name = 2, "Evening"
    else:
        bucket, name = 3, "Night"

    return {
        "hour":        hour,
        "bucket":      bucket,
        "bucket_name": name,
    }


# ─────────────────────────────────────────────────────────────────────────
# Tool 3 — Season (useful for agriculture's planting-window logic)
# ─────────────────────────────────────────────────────────────────────────
@mcp.tool()
def get_season() -> dict:
    """
    Returns the REAL current month and season (Northern Hemisphere
    convention — adjust SEASON_MAP below if your deployment target is
    Southern Hemisphere).

    Returns
    -------
    dict with keys:
        month  : int 1-12
        season : "Spring" / "Summer" / "Autumn" / "Winter"
    """
    month = datetime.now().month

    if month in (3, 4, 5):
        season = "Spring"
    elif month in (6, 7, 8):
        season = "Summer"
    elif month in (9, 10, 11):
        season = "Autumn"
    else:
        season = "Winter"

    return {
        "month":  month,
        "season": season,
    }


if __name__ == "__main__":
    # Runs the server over stdio — an MCP client (mcp_client.py) launches
    # this as a subprocess and talks to it over stdin/stdout.
    mcp.run()
