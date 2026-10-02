"""
Tool: locate_nearby_clinics
Real implementation -- free, no billing required:
1. Nominatim (OpenStreetMap) geocodes the user's location string into
   latitude/longitude.
2. Overpass API (OpenStreetMap) searches for nearby hospitals/clinics
   tagged amenity=hospital or amenity=clinic within a radius.
3. A single batched LLM call classifies which of the returned
   facility names are actually appropriate for burn/emergency/general
   medical care -- replacing an earlier keyword-blocklist approach,
   which proved unreliable against real-world OSM data (abbreviations,
   misspellings, and facilities mistagged as amenity=hospital that are
   really alternative-medicine or non-medical businesses all slipped
   through a hand-maintained keyword list). The LLM step generalizes
   far better without manual list maintenance.

   This classification call runs on Gemini (google-genai), not Groq,
   deliberately: it is a small, self-contained, single-turn, no-tool,
   no-conversation-history call, so it doesn't need the tool-calling
   integration the main agent loop is built around, and routing it to
   a second provider keeps it off the main agent's Groq rate-limit
   budget. On any Gemini failure this still fails open (see
   _classify_facilities_with_llm) -- it never blocks clinic results.

Both OSM services are free public services with no signup or billing
needed -- citable in a research paper as "OpenStreetMap contributors"
(OSM is a legitimate, widely used open geographic dataset in published
healthcare-access research).

Courtesy note: both OSM services request a descriptive User-Agent
header and reasonable request rates -- we set a clear User-Agent and
this tool is only called occasionally (once per triage decision),
which is within acceptable use.
"""

import os
import json
import requests
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
OVERPASS_URL = "https://overpass-api.de/api/interpreter"
USER_AGENT = "BurnScope-Capstone-Project/1.0 (student research project)"

_gemini_client = None
CLASSIFY_MODEL = "gemini-3.6-flash"

CLASSIFY_SYSTEM_INSTRUCTION = (
    "You are a strict JSON-only classifier for a burn-wound triage app. Respond with ONLY a JSON "
    "object, no other text, no markdown fences."
)


def _get_gemini_client():
    global _gemini_client
    if _gemini_client is None:
        _gemini_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    return _gemini_client


def _geocode_location(location: str):
    """
    Converts a place name/address string into (lat, lon) using Nominatim.
    Returns None if no match is found.
    """
    params = {"q": location, "format": "json", "limit": 1}
    headers = {"User-Agent": USER_AGENT}

    response = requests.get(NOMINATIM_URL, params=params, headers=headers, timeout=10)
    response.raise_for_status()
    results = response.json()

    if not results:
        return None

    return float(results[0]["lat"]), float(results[0]["lon"])


def _search_nearby_facilities(lat: float, lon: float, radius_km: float, max_retries: int = 3):
    """
    Queries the Overpass API for hospitals and clinics within radius_km
    of (lat, lon). No specialty filtering is done here -- that now
    happens via the LLM classification step, since OSM's structured
    tags are too inconsistently filled in to filter reliably alone.
    Returns a list of raw OSM elements.

    The public Overpass instance is a free, shared, best-effort service
    with no uptime guarantee and frequently times out under load -- we
    retry a few times with a short backoff before giving up, since a
    second or third attempt often succeeds where the first didn't.
    """
    import time

    radius_m = int(radius_km * 1000)
    query = f"""
    [out:json][timeout:15];
    (
      node["amenity"="hospital"](around:{radius_m},{lat},{lon});
      way["amenity"="hospital"](around:{radius_m},{lat},{lon});
      node["amenity"="clinic"](around:{radius_m},{lat},{lon});
      way["amenity"="clinic"](around:{radius_m},{lat},{lon});
    );
    out center 30;
    """
    headers = {"User-Agent": USER_AGENT}

    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(OVERPASS_URL, data={"data": query}, headers=headers, timeout=20)
            response.raise_for_status()
            return response.json().get("elements", [])
        except requests.RequestException as e:
            last_error = e
            print(f"  [clinic_locator] Overpass attempt {attempt}/{max_retries} failed ({e}); "
                  f"{'retrying...' if attempt < max_retries else 'giving up.'}")
            if attempt < max_retries:
                time.sleep(2 * attempt)  # 2s, then 4s backoff

    raise last_error


def _classify_facilities_with_llm(names: list) -> list:
    """
    Sends ALL candidate facility names in ONE batched Gemini call and asks
    which are appropriate for burn/emergency/general medical care.

    Args:
        names: list of facility name strings, in order.

    Returns:
        list of bool, same length and order as `names`. On any failure
        (API error, malformed response, length mismatch), fails open --
        returns all True -- so a classification hiccup never silently
        drops real, possibly-relevant results from the user.
    """
    if not names:
        return []

    numbered_list = "\n".join(f"{i}: {name}" for i, name in enumerate(names))
    prompt = f"""For EACH numbered facility below, decide if it is
an appropriate place to seek care for a burn injury -- i.e. a general
hospital, emergency clinic, general practitioner, or nursing home.

Mark as NOT appropriate: alternative medicine (homeopathy, ayurveda,
naturopathy), narrow specialty clinics unrelated to burns/emergencies
(eye/dental/veterinary/gynecology/psychiatry/physiotherapy/cosmetic),
yoga studios, spas, wellness centers, or anything clearly non-medical.
When genuinely uncertain, default to APPROPRIATE (do not over-exclude).

Facilities:
{numbered_list}

Respond with ONLY a JSON object, in this exact shape:
{{"appropriate": [true, false, ...]}}
The list must have exactly {len(names)} boolean values, in the same
order as the numbered facilities above."""

    try:
        client = _get_gemini_client()
        response = client.models.generate_content(
            model=CLASSIFY_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0,
                system_instruction=CLASSIFY_SYSTEM_INSTRUCTION,
                max_output_tokens=2000,
            ),
        )
        content = (response.text or "").strip()

        # Models sometimes wrap JSON in markdown fences despite instructions -- strip if present.
        if content.startswith("```"):
            content = content.strip("`")
            if content.lower().startswith("json"):
                content = content[4:].strip()

        parsed = json.loads(content)
        results = parsed.get("appropriate", [])

        if len(results) != len(names):
            print(f"  [clinic_locator] LLM classification length mismatch "
                  f"({len(results)} vs {len(names)}), failing open (keeping all).")
            return [True] * len(names)

        return [bool(r) for r in results]

    except Exception as e:
        print(f"  [clinic_locator] LLM classification failed ({e}), failing open (keeping all).")
        return [True] * len(names)


def _haversine_km(lat1, lon1, lat2, lon2):
    """Approximate straight-line distance in km between two lat/lon points."""
    from math import radians, sin, cos, sqrt, atan2
    R = 6371.0
    d_lat = radians(lat2 - lat1)
    d_lon = radians(lon2 - lon1)
    a = sin(d_lat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(d_lon / 2) ** 2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


MIN_OPTIONS = 4   # keep widening the radius (5 -> 15 -> 30 km) until at least this many are found


def _find_clinics(lat, lon, radius_km):
    """Appropriate facilities within radius_km, hospitals first then nearest."""
    elements = _search_nearby_facilities(lat, lon, radius_km)

    raw_facilities = []
    for el in elements:
        tags = el.get("tags", {})
        name = tags.get("name", "Unnamed facility")

        el_lat = el.get("lat") or el.get("center", {}).get("lat")
        el_lon = el.get("lon") or el.get("center", {}).get("lon")
        if el_lat is None or el_lon is None:
            continue

        distance = _haversine_km(lat, lon, el_lat, el_lon)

        address_parts = [
            tags.get("addr:housenumber", ""),
            tags.get("addr:street", ""),
            tags.get("addr:city", ""),
        ]
        address = " ".join(p for p in address_parts if p) or "Address not available"

        raw_facilities.append({
            "name": name,
            "address": address,
            "distance_km": round(distance, 2),
            "is_hospital": tags.get("amenity") == "hospital",
        })

    # De-duplicate BEFORE the LLM call, so we don't waste tokens
    # classifying the same real-world place twice.
    seen = set()
    deduped = []
    for c in raw_facilities:
        key = (c["name"].strip().lower(), round(c["distance_km"], 1))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(c)

    names = [c["name"] for c in deduped]
    appropriate_flags = _classify_facilities_with_llm(names)

    clinics = [c for c, is_ok in zip(deduped, appropriate_flags) if is_ok]

    # Nearest first, so the chat and the report agree on "the closest few".
    clinics.sort(key=lambda c: c["distance_km"])
    return clinics


def locate_nearby_clinics(location: str, radius_km: float = 5.0) -> dict:
    """
    Args:
        location: user's approximate location (city, address, etc.)
        radius_km: search radius in kilometers.

    Returns:
        dict with key:
            clinics: list of {name, address, distance_km}
        On failure (location not found, no results, network error),
        returns an empty list with a "note" field explaining why,
        rather than raising -- so the agent can gracefully tell the
        user instead of crashing.
    """
    try:
        coords = _geocode_location(location)
        if coords is None:
            return {"clinics": [], "note": f"Could not find a location matching '{location}'."}
        lat, lon = coords

        # Widen the search when the first radius finds too few options, so the user
        # (and the report) get the closest few choices rather than a single one.
        clinics, used_radius = [], radius_km
        for used_radius in dict.fromkeys([radius_km, max(radius_km, 15.0), max(radius_km, 30.0)]):
            clinics = _find_clinics(lat, lon, used_radius)
            if len(clinics) >= MIN_OPTIONS:
                break

        for c in clinics:
            del c["is_hospital"]

        result = {"clinics": clinics[:10]}
        if used_radius != radius_km:
            result["note"] = f"Few options within {radius_km:g} km, so the search was widened to {used_radius:g} km."
        return result

    except requests.RequestException as e:
        return {"clinics": [], "note": f"Location lookup service unavailable: {e}"}