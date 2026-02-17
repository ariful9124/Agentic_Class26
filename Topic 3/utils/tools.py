# tools.py
import math
from typing import Dict, Any, List

from langchain_core.tools import tool

from utils.data_i64 import I64_ROUTE, get_coords, haversine_miles


# ---- I-64 tools ----
@tool
def list_i64_cities() -> Dict[str, Any]:
    """Return the valid city names in the I-64 route list."""
    return {"ok": True, "cities": [c["city"] for c in I64_ROUTE]}


@tool
def get_coords_tool(city_name: str) -> Dict[str, Any]:
    """Get latitude/longitude for a city in the I-64 route list."""
    try:
        coords = get_coords(city_name)
        return {"ok": True, "city": city_name, **coords}
    except Exception as e:
        return {"ok": False, "error": str(e), "city": city_name}


@tool
def distance_between_cities_tool(city_a: str, city_b: str) -> Dict[str, Any]:
    """Compute Haversine distance (miles) between two I-64 cities."""
    try:
        a = get_coords(city_a)
        b = get_coords(city_b)
        dist = haversine_miles(a["lat"], a["lon"], b["lat"], b["lon"])
        return {"ok": True, "city_a": city_a, "city_b": city_b, "distance_miles": dist}
    except Exception as e:
        return {"ok": False, "error": str(e), "city_a": city_a, "city_b": city_b}


# ---- Letter + sin + calculator tools ----
@tool
def count_letter_tool(text: str, letter: str) -> Dict[str, Any]:
    """Count occurrences of a single letter in text (case-insensitive)."""
    if not letter or len(letter) != 1:
        return {"ok": False, "error": "letter must be exactly one character."}
    n = sum(1 for ch in text.lower() if ch == letter.lower())
    return {"ok": True, "text": text, "letter": letter, "count": n}


@tool
def sin_tool(x: float) -> Dict[str, Any]:
    """Compute sin(x) where x is in radians."""
    return {"ok": True, "x": x, "sin_x": math.sin(x)}


@tool
def calculator_tool(a: float, b: float, op_name: str) -> Dict[str, Any]:
    """
    Simple calculator for this project.
    op_name in: "add", "sub", "mul", "div"
    """
    try:
        if op_name == "add":
            v = a + b
        elif op_name == "sub":
            v = a - b
        elif op_name == "mul":
            v = a * b
        elif op_name == "div":
            if b == 0:
                return {"ok": False, "error": "division by zero"}
            v = a / b
        else:
            return {"ok": False, "error": f"unknown op_name: {op_name}"}
        return {"ok": True, "a": a, "b": b, "op_name": op_name, "value": v}
    except Exception as e:
        return {"ok": False, "error": str(e), "a": a, "b": b, "op_name": op_name}


TOOLS: List = [
    list_i64_cities,
    get_coords_tool,
    distance_between_cities_tool,
    count_letter_tool,
    sin_tool,
    calculator_tool,
]
