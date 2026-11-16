import datetime
from html.parser import HTMLParser
import math
import re
import time
from zoneinfo import ZoneInfo
import requests

# Known PAGASA AWS / Synoptic stations in and around Negros Island Region & Central Visayas
NIR_AWS_STATIONS = [
    {"keywords": ["dumaguete", "sibulan"], "name": "Dumaguete / Sibulan AWS", "lat": 9.3333, "lon": 123.3000},
    {"keywords": ["la granja", "la carlota"], "name": "La Granja, La Carlota AWS", "lat": 10.4050, "lon": 122.9780},
    {"keywords": ["bacolod", "silay"], "name": "Bacolod / Silay AWS", "lat": 10.7764, "lon": 123.0150},
    {"keywords": ["kabankalan"], "name": "Kabankalan AWS", "lat": 9.9833, "lon": 122.8167},
    {"keywords": ["mactan", "cebu"], "name": "Mactan, Cebu AWS", "lat": 10.3125, "lon": 123.9794},
    {"keywords": ["bohol", "tagbilaran"], "name": "Bohol Island AWS", "lat": 9.6500, "lon": 123.8500},
    {"keywords": ["bohol", "tagbilaran", "panglao"], "name": "Panglao / Tagbilaran, Bohol AWS", "lat": 9.5767, "lon": 123.7742},
]

_CACHE = {"timestamp": 0.0, "rows": []}
CACHE_TTL_SECONDS = 900  # Cache table for 15 minutes so 58-town batch runs make only 1 HTTP request

class _TableParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.in_td = False
        self.rows = []
        self.current_row = []
        self.current_cell = []

    def handle_starttag(self, tag, attrs):
        if tag in ("td", "th"):
            self.in_td = True
            self.current_cell = []
        elif tag == "tr":
            self.current_row = []

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self.in_td:
            self.in_td = False
            self.current_row.append(" ".join("".join(self.current_cell).split()))
        elif tag == "tr" and self.current_row:
            self.rows.append(self.current_row)

    def handle_data(self, data):
        if self.in_td:
            self.current_cell.append(data)

def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def _extract_float(text: str):
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    return float(match.group(0)) if match else None

def _is_recent(last_updated_str: str, max_age_hours: float = 2.5) -> bool:
    """Verifies that a PAGASA station reading is fresh (not stale from days ago)."""
    try:
        # Format example: "November 16, 2026, 11:30 am"
        cleaned = last_updated_str.strip()
        dt = datetime.datetime.strptime(cleaned, "%B %d, %Y, %I:%M %p")
        now_mnl = datetime.datetime.now(ZoneInfo("Asia/Manila")).replace(tzinfo=None)
        age_hours = abs((now_mnl - dt).total_seconds()) / 3600.0
        return age_hours <= max_age_hours
    except Exception:
        return False

def _fetch_pagasa_table() -> list[list[str]]:
    now = time.time()
    if _CACHE["rows"] and (now - _CACHE["timestamp"]) < CACHE_TTL_SECONDS:
        return _CACHE["rows"]

    url = "".join(["https://", "bagong.pagasa.", "dost.gov.ph", "/automated-weather-station/"])
    resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
    resp.raise_for_status()

    parser = _TableParser()
    parser.feed(resp.text)
    _CACHE["timestamp"] = now
    _CACHE["rows"] = parser.rows
    return parser.rows

def get_pagasa_ground_boost(lat: float, lon: float) -> dict | None:
    """
    Finds the nearest fresh (<2.5h old) PAGASA AWS station around NIR/Central Visayas.
    Returns dict with station name, distance_km, precip_mm, and humidity, or None if unavailable.
    """
    try:
        rows = _fetch_pagasa_table()
    except Exception:
        return None

    active_candidates = []
    for row in rows:
        # Expected columns: Site ID | Site Name | Temp | Humidity | Wind | Dir | Precip | Pressure | Solar | Last Updated
        if len(row) < 10:
            continue
        site_name = row[1].lower()
        last_updated = row[9]

        if not _is_recent(last_updated):
            continue

        for stn in NIR_AWS_STATIONS:
            if any(kw in site_name for kw in stn["keywords"]):
                dist_km = _haversine_km(lat, lon, stn["lat"], stn["lon"])
                precip = _extract_float(row[6])
                humidity = _extract_float(row[3])
                temp = _extract_float(row[2])

                active_candidates.append({
                    "station": row[1],
                    "distance_km": round(dist_km, 1),
                    "precip_mm": precip if precip is not None else 0.0,
                    "humidity": humidity,
                    "temp": temp,
                    "last_updated": last_updated
                })

    if not active_candidates:
        return None

    # Pick the closest active station (within 120 km of target town)
    nearest = min(active_candidates, key=lambda x: x["distance_km"])
    if nearest["distance_km"] > 120.0:
        return None

    return nearest
