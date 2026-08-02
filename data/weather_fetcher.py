"""
data/weather_fetcher.py
========================
Fetches real historical and current weather data for Bangladesh
from Open-Meteo (free, no API key needed) and stores it in a
local SQLite database via SQLAlchemy.

Coordinates used: Dhaka, Bangladesh (23.8103, 90.4125)
-- adjust LAT/LON below for your specific farm location

Run directly to populate the database:
    python data/weather_fetcher.py

Or import and call:
    from data.weather_fetcher import WeatherFetcher
    fetcher = WeatherFetcher()
    fetcher.fetch_and_store(days=365)
"""

import os
import json
import requests
from datetime import datetime, timedelta, date

from sqlalchemy import (
    create_engine, Column, Integer, Float, String,
    Date, DateTime, Boolean, text
)
from sqlalchemy.orm import DeclarativeBase, Session


# ── Bangladesh coordinates ────────────────────────────────────────────────────
LAT = 23.8103   # Dhaka -- change to your farm location
LON = 90.4125
LOCATION_NAME = "Dhaka, Bangladesh"

# ── Database path ─────────────────────────────────────────────────────────────
DB_DIR  = os.path.join(os.path.dirname(__file__), "..", "data")
DB_PATH = os.path.join(DB_DIR, "weather.db")

# ── Open-Meteo API ────────────────────────────────────────────────────────────
METEO_HISTORICAL = "https://archive-api.open-meteo.com/v1/archive"
METEO_FORECAST   = "https://api.open-meteo.com/v1/forecast"

DAILY_VARS = [
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "rain_sum",
    "windspeed_10m_max",
    "et0_fao_evapotranspiration",   # evapotranspiration -- critical for irrigation
    "sunshine_duration",
]

HOURLY_VARS = [
    "soil_moisture_0_to_1cm",
    "soil_moisture_1_to_3cm",
    "relativehumidity_2m",
]


# ═════════════════════════════════════════════════════════════════════════════
# SQLAlchemy Models
# ═════════════════════════════════════════════════════════════════════════════
class Base(DeclarativeBase):
    pass


class DailyWeather(Base):
    """One row per calendar day of weather data."""
    __tablename__ = "daily_weather"

    id                      = Column(Integer, primary_key=True, autoincrement=True)
    date                    = Column(Date,    nullable=False, unique=True, index=True)
    location                = Column(String,  default=LOCATION_NAME)

    # Temperature (celsius)
    temp_max                = Column(Float)
    temp_min                = Column(Float)
    temp_mean               = Column(Float)   # computed as (max+min)/2

    # Precipitation (mm)
    precipitation           = Column(Float)   # total precipitation
    rain                    = Column(Float)   # rain specifically

    # Wind
    windspeed_max           = Column(Float)   # km/h

    # Evapotranspiration (mm/day) -- how much water crops lose
    evapotranspiration      = Column(Float)

    # Sunshine (seconds)
    sunshine_duration       = Column(Float)

    # Soil moisture (m³/m³, top layer)
    soil_moisture_surface   = Column(Float)
    soil_moisture_shallow   = Column(Float)

    # Humidity (%)
    humidity_mean           = Column(Float)

    # Derived agriculture metrics
    rainfall_trend          = Column(Integer)  # -2 to +2 (computed)
    drought_index           = Column(Float)    # precipitation - evapotranspiration
    is_monsoon              = Column(Boolean)  # June-October in Bangladesh

    # Metadata
    source                  = Column(String,  default="open-meteo")
    fetched_at              = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return (f"<DailyWeather date={self.date} "
                f"rain={self.rain}mm temp={self.temp_mean}°C>")


class WeatherSummary(Base):
    """Monthly aggregated statistics for quick lookup."""
    __tablename__ = "weather_summary"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    year            = Column(Integer, nullable=False)
    month           = Column(Integer, nullable=False)
    avg_temp        = Column(Float)
    total_rain      = Column(Float)
    avg_humidity    = Column(Float)
    avg_soil_moist  = Column(Float)
    avg_et0         = Column(Float)
    monsoon_days    = Column(Integer)
    drought_days    = Column(Integer)   # days where rain < et0

    def __repr__(self):
        return f"<WeatherSummary {self.year}-{self.month:02d}>"


# ═════════════════════════════════════════════════════════════════════════════
# WeatherFetcher class
# ═════════════════════════════════════════════════════════════════════════════
class WeatherFetcher:

    def __init__(self, db_path: str = DB_PATH):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.engine  = create_engine(f"sqlite:///{db_path}", echo=False)
        Base.metadata.create_all(self.engine)
        self.db_path = db_path
        print(f"  Database: {db_path}")

    # ── Fetch from Open-Meteo ─────────────────────────────────────────────────
    def _fetch_historical(self, start_date: str, end_date: str) -> dict:
        params = {
            "latitude":    LAT,
            "longitude":   LON,
            "start_date":  start_date,
            "end_date":    end_date,
            "daily":       ",".join(DAILY_VARS),
            "timezone":    "Asia/Dhaka",
        }
        try:
            resp = requests.get(METEO_HISTORICAL, params=params, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            print(f"  [WARNING] Historical fetch failed: {e}")
            return {}

    def _fetch_forecast(self, days: int = 7) -> dict:
        params = {
            "latitude":    LAT,
            "longitude":   LON,
            "daily":       ",".join(DAILY_VARS),
            "forecast_days": days,
            "timezone":    "Asia/Dhaka",
        }
        try:
            resp = requests.get(METEO_FORECAST, params=params, timeout=15)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            print(f"  [WARNING] Forecast fetch failed: {e}")
            return {}

    # ── Compute derived fields ────────────────────────────────────────────────
    @staticmethod
    def _rainfall_trend(rain_mm: float, et0: float) -> int:
        """
        Compute rainfall trend on -2 to +2 scale.
        Compares rainfall to evapotranspiration demand.
        """
        if rain_mm is None or et0 is None:
            return 0
        surplus = rain_mm - et0
        if surplus >= 15:   return  2   # heavy rain
        if surplus >= 5:    return  1   # light rain
        if surplus >= -3:   return  0   # balanced
        if surplus >= -10:  return -1   # dry
        return -2                       # drought

    @staticmethod
    def _is_monsoon(d: date) -> bool:
        """Bangladesh monsoon: June (6) to October (10)."""
        return 6 <= d.month <= 10

    # ── Parse and store API response ──────────────────────────────────────────
    def _parse_and_store(self, data: dict):
        if not data or "daily" not in data:
            print("  [WARNING] No data to parse.")
            return 0

        daily   = data["daily"]
        dates   = daily.get("time", [])
        stored  = 0

        with Session(self.engine) as session:
            for i, date_str in enumerate(dates):
                d = date.fromisoformat(date_str)

                # Skip if already stored
                existing = session.query(DailyWeather).filter_by(date=d).first()
                if existing:
                    continue

                def get(key, idx=i):
                    val = daily.get(key, [])
                    return val[idx] if idx < len(val) and val[idx] is not None else None

                t_max = get("temperature_2m_max")
                t_min = get("temperature_2m_min")
                rain  = get("rain_sum")
                et0   = get("et0_fao_evapotranspiration")
                prec  = get("precipitation_sum")

                row = DailyWeather(
                    date               = d,
                    temp_max           = t_max,
                    temp_min           = t_min,
                    temp_mean          = round((t_max + t_min) / 2, 2) if t_max and t_min else None,
                    precipitation      = prec,
                    rain               = rain,
                    windspeed_max      = get("windspeed_10m_max"),
                    evapotranspiration = et0,
                    sunshine_duration  = get("sunshine_duration"),
                    rainfall_trend     = self._rainfall_trend(rain, et0),
                    drought_index      = round((rain or 0) - (et0 or 0), 2),
                    is_monsoon         = self._is_monsoon(d),
                )
                session.add(row)
                stored += 1

            session.commit()

        return stored

    # ── Public API ────────────────────────────────────────────────────────────
    def fetch_and_store(self, days: int = 365):
        """
        Fetch the last `days` days of historical weather and store in DB.
        Also fetches the 7-day forecast and stores it.
        """
        end   = date.today() - timedelta(days=1)   # yesterday (historical is not real-time)
        start = end - timedelta(days=days)

        print(f"\n  Fetching {days} days of historical weather for {LOCATION_NAME}...")
        print(f"  Period: {start} to {end}")

        hist_data = self._fetch_historical(str(start), str(end))
        n_hist    = self._parse_and_store(hist_data)
        print(f"  Stored {n_hist} new historical records")

        print(f"\n  Fetching 7-day forecast...")
        fore_data = self._fetch_forecast(days=7)
        n_fore    = self._parse_and_store(fore_data)
        print(f"  Stored {n_fore} forecast records")

        self._update_summaries()
        print(f"\n  Database ready: {self.db_path}")
        return n_hist + n_fore

    def fetch_today(self) -> dict:
        """Fetch today's forecast and return as dict."""
        data = self._fetch_forecast(days=1)
        if not data or "daily" not in data:
            return {}
        self._parse_and_store(data)
        return self.get_latest()

    def get_latest(self) -> dict:
        """Return the most recent row as a plain dict."""
        with Session(self.engine) as session:
            row = (session.query(DailyWeather)
                   .order_by(DailyWeather.date.desc())
                   .first())
            if not row:
                return {}
            return {
                "date":             str(row.date),
                "temp_max":         row.temp_max,
                "temp_min":         row.temp_min,
                "temp_mean":        row.temp_mean,
                "rain":             row.rain,
                "precipitation":    row.precipitation,
                "evapotranspiration": row.evapotranspiration,
                "rainfall_trend":   row.rainfall_trend,
                "drought_index":    row.drought_index,
                "soil_moisture":    row.soil_moisture_surface,
                "humidity":         row.humidity_mean,
                "is_monsoon":       row.is_monsoon,
                "windspeed":        row.windspeed_max,
            }

    def get_by_date(self, query_date: date) -> dict:
        """Return weather for a specific date."""
        with Session(self.engine) as session:
            row = session.query(DailyWeather).filter_by(date=query_date).first()
            if not row:
                return {}
            return {
                "date":           str(row.date),
                "temp_mean":      row.temp_mean,
                "rain":           row.rain,
                "rainfall_trend": row.rainfall_trend,
                "drought_index":  row.drought_index,
                "is_monsoon":     row.is_monsoon,
                "evapotranspiration": row.evapotranspiration,
            }

    def get_date_range(self, start: date, end: date) -> list:
        """Return list of weather dicts for a date range."""
        with Session(self.engine) as session:
            rows = (session.query(DailyWeather)
                    .filter(DailyWeather.date >= start)
                    .filter(DailyWeather.date <= end)
                    .order_by(DailyWeather.date)
                    .all())
            return [
                {
                    "date":           str(r.date),
                    "temp_mean":      r.temp_mean,
                    "rain":           r.rain,
                    "rainfall_trend": r.rainfall_trend,
                    "drought_index":  r.drought_index,
                    "is_monsoon":     r.is_monsoon,
                    "evapotranspiration": r.evapotranspiration,
                }
                for r in rows
            ]

    def total_records(self) -> int:
        with Session(self.engine) as session:
            return session.query(DailyWeather).count()

    def _update_summaries(self):
        """Compute and store monthly summaries."""
        with Session(self.engine) as session:
            rows = session.query(DailyWeather).all()
            monthly = {}
            for r in rows:
                key = (r.date.year, r.date.month)
                if key not in monthly:
                    monthly[key] = {
                        "temps": [], "rains": [], "et0s": [],
                        "monsoon": 0, "drought": 0
                    }
                if r.temp_mean:  monthly[key]["temps"].append(r.temp_mean)
                if r.rain:       monthly[key]["rains"].append(r.rain)
                if r.evapotranspiration: monthly[key]["et0s"].append(r.evapotranspiration)
                if r.is_monsoon: monthly[key]["monsoon"] += 1
                if r.drought_index and r.drought_index < 0:
                    monthly[key]["drought"] += 1

            for (yr, mo), v in monthly.items():
                existing = (session.query(WeatherSummary)
                            .filter_by(year=yr, month=mo).first())
                if not existing:
                    existing = WeatherSummary(year=yr, month=mo)
                    session.add(existing)
                existing.avg_temp       = round(sum(v["temps"]) / max(len(v["temps"]), 1), 2)
                existing.total_rain     = round(sum(v["rains"]), 2)
                existing.avg_et0        = round(sum(v["et0s"]) / max(len(v["et0s"]), 1), 2)
                existing.monsoon_days   = v["monsoon"]
                existing.drought_days   = v["drought"]

            session.commit()


# ── Run directly to populate DB ───────────────────────────────────────────────
if __name__ == "__main__":
    fetcher = WeatherFetcher()
    n = fetcher.fetch_and_store(days=365)
    print(f"\n  Total records in DB: {fetcher.total_records()}")
    latest = fetcher.get_latest()
    print(f"  Latest record: {latest}")