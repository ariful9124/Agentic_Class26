import math
from typing import Dict, List

I64_ROUTE: List[dict] = [
    {"city": "Chesapeake, VA", "lat": 36.8190, "lon": -76.2749},
    {"city": "Richmond, VA", "lat": 37.5407, "lon": -77.4360},
    {"city": "Charlottesville, VA", "lat": 38.0293, "lon": -78.4767},
    {"city": "Lexington, VA", "lat": 37.7840, "lon": -79.4430},
    {"city": "Staunton, VA", "lat": 38.1496, "lon": -79.0717},
    {"city": "Charleston, WV", "lat": 38.3498, "lon": -81.6326},
    {"city": "Lexington, KY", "lat": 38.0406, "lon": -84.5037},
    {"city": "Louisville, KY", "lat": 38.2527, "lon": -85.7585},
    {"city": "Evansville, IN", "lat": 37.9748, "lon": -87.5558},
    {"city": "St. Louis, MO", "lat": 38.6270, "lon": -90.1994},
    {"city": "Wentzville, MO", "lat": 38.8139, "lon": -90.8462},
]

CITY_INDEX = {c["city"].lower(): c for c in I64_ROUTE}


def get_coords(city_name: str) -> Dict[str, float]:
    key = city_name.strip().lower()
    if key not in CITY_INDEX:
        raise ValueError(
            f"City '{city_name}' not found in the I-64 list. "
            f"Call list_i64_cities to see valid names."
        )
    c = CITY_INDEX[key]
    return {"lat": c["lat"], "lon": c["lon"]}


def haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 3958.8
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))
