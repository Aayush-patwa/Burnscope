"""Nearby facilities using Nominatim geocoding and Geoapify Places.

SETUP: add GEOAPIFY_API_KEY to your project-root .env and restart Streamlit.
No Overpass request is made. Gemini remains an optional name filter.
Direct diagnostic: python -m tools.clinic_locator "Tilak Nagar, Indore, India"
Provider docs: https://apidocs.geoapify.com/docs/places/
Display Geoapify / OpenStreetMap attribution in app and reports.
Keeps locate_nearby_clinics(location, radius_km=5.0) compatible with BurnScope.
Facility listings are not verification of burn expertise, opening hours or availability.
"""

import os
import json
import math
import hashlib
import re
import time
import threading
from collections import OrderedDict
from copy import deepcopy

import requests
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

NOMINATIM_URL = os.getenv("BURNSCOPE_NOMINATIM_URL", "https://nominatim.openstreetmap.org/search")
GEOAPIFY_PLACES_URL = "https://api.geoapify.com/v2/places"
# General care candidates, not every specialised practice (eye/dental etc.).
GEOAPIFY_CATEGORIES = ",".join((
    "healthcare.hospital", "healthcare.clinic_or_praxis.general",
    "healthcare.clinic_or_praxis.trauma",
))
PROVIDER = "geoapify"
ATTRIBUTION = "Powered by Geoapify; data: OpenStreetMap contributors (ODbL)."
_CACHE = OrderedDict()
_CACHE_LOCK = threading.RLock()
_GEOCODE_LOCK = threading.Lock()
_LAST_GEOCODE = 0.0
CACHE_LIMIT = 256
SEARCH_BUDGET_SECONDS = 60.0  # soft budget; requests timeouts are not strict wall-clock deadlines


class ClinicLookupError(requests.RequestException):
    def __init__(self, stage, message, code="service_unavailable"):
        super().__init__(message)
        self.stage = stage
        self.code = code


def _cached(key):
    with _CACHE_LOCK:
        item = _CACHE.get(key)
        if item and item[0] > time.monotonic():
            _CACHE.move_to_end(key)
            return deepcopy(item[1])
        _CACHE.pop(key, None)
    return None


def _remember(key, value, seconds):
    with _CACHE_LOCK:
        _CACHE[key] = (time.monotonic() + seconds, deepcopy(value))
        _CACHE.move_to_end(key)
        while len(_CACHE) > CACHE_LIMIT:
            _CACHE.popitem(last=False)


def _remaining(deadline):
    remaining = deadline - time.monotonic()
    if remaining < 2:
        raise ClinicLookupError("geoapify", "The facility-search time budget was exhausted. Please retry.", "timeout")
    return remaining
USER_AGENT = os.getenv("BURNSCOPE_OSM_USER_AGENT", "BurnScope-Capstone-Project/1.1 (student research project)")

_gemini_client = None
CLASSIFY_MODEL = "gemini-3.6-flash"

# A small fallback for explicit unrelated specialisms. This is a heuristic,
# not verification of clinical capability. Keep ambiguous/general names.
_UNRELATED_NAME = re.compile(
    r"\b(?:homeo\w*|homoeo\w*|ayurved\w*|naturopath\w*|yoga|spa|wellness|"
    r"dental|dentist\w*|orthodont\w*|veterinar\w*|eye|ophthalm\w*|optical)\b",
    re.IGNORECASE,
)
_NARROW_NAME = re.compile(
    r"\b(?:maternity|gynae\w*|gyne\w*|fertility|ivf|psychiatr\w*|physiotherap\w*)\b",
    re.IGNORECASE,
)
_GENERAL_NAME = re.compile(
    r"\b(?:general|multi[\s-]?special(?:ity|ty)|emergency|trauma|burns?)\b",
    re.IGNORECASE,
)


def _name_is_candidate(name: str) -> bool:
    """Exclude clear alternative/unrelated names even when Gemini is unavailable.
    A general/multispecialty service can coexist with maternity or another department;
    a name alone still cannot establish burn-care suitability or availability.
    """
    if _UNRELATED_NAME.search(name):
        return False
    return not _NARROW_NAME.search(name) or bool(_GENERAL_NAME.search(name))

CLASSIFY_SYSTEM_INSTRUCTION = (
    "You are a strict JSON-only classifier for a burn-wound triage app. Respond with ONLY a JSON "
    "object, no other text, no markdown fences."
)


def _get_gemini_client():
    global _gemini_client
    if _gemini_client is None:
        _gemini_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"],
                                      http_options=types.HttpOptions(timeout=12000))
    return _gemini_client


def _geocode_location(location: str):
    """Cache successful geocodes; serialize public Nominatim calls to <=1/sec/process."""
    global _LAST_GEOCODE
    key = ("geo", " ".join(location.casefold().split()))
    hit = _cached(key)
    if hit is not None:
        return hit
    with _GEOCODE_LOCK:
        hit = _cached(key)
        if hit is not None:
            return hit
        wait = 1.1 - (time.monotonic() - _LAST_GEOCODE)
        if wait > 0:
            time.sleep(wait)
        _LAST_GEOCODE = time.monotonic()
        try:
            response = requests.get(NOMINATIM_URL,
                params={"q": location, "format": "json", "limit": 1},
                headers={"User-Agent": USER_AGENT}, timeout=(5, 15))
            response.raise_for_status()
            results = response.json()
            if not isinstance(results, list):
                raise ValueError("Invalid geocoding response")
            if not results:
                return None
            first = results[0]
            result = {"lat": float(first["lat"]), "lon": float(first["lon"]),
                      "display_name": first.get("display_name", location)}
            if not (-90 <= result["lat"] <= 90 and -180 <= result["lon"] <= 180):
                raise ValueError("Invalid coordinates")
        except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
            raise ClinicLookupError("geocoding", "Could not resolve the requested place.") from exc
        _remember(key, result, 86400)
        return result


def _search_nearby_facilities(lat, lon, radius_km, deadline=None):
    """GeoJSON features nearest the given point, inside the requested circle.
    The key goes only to Geoapify, never to the LLM, log, cache value or returned result.
    Do not expose requests exceptions: their URL can contain apiKey.
    """
    api_key = os.getenv("GEOAPIFY_API_KEY", "").strip()
    if not api_key:
        raise ClinicLookupError("configuration", "GEOAPIFY_API_KEY is missing. Add it to .env and restart Streamlit.", "missing_api_key")
    deadline = deadline or time.monotonic() + SEARCH_BUDGET_SECONDS
    key_id = hashlib.sha256(api_key.encode()).hexdigest()[:16]
    key = ("geoapify", round(lat, 6), round(lon, 6), float(radius_km), key_id)
    hit = _cached(key)
    if hit is not None:
        return hit
    remaining = _remaining(deadline)
    params = {"categories": GEOAPIFY_CATEGORIES,
              "filter": f"circle:{lon},{lat},{int(radius_km * 1000)}",
              "bias": f"proximity:{lon},{lat}", "limit": 60,
              "lang": "en", "apiKey": api_key}
    try:
        response = requests.get(GEOAPIFY_PLACES_URL, params=params,
            headers={"User-Agent": USER_AGENT},
            timeout=(min(5, remaining / 4), min(20, remaining * 0.7)))
    except requests.Timeout:
        raise ClinicLookupError("geoapify", "Geoapify did not respond in time. Please retry shortly.", "timeout") from None
    except requests.RequestException:
        raise ClinicLookupError("geoapify", "Could not connect to Geoapify. Check the network and retry.", "connection_error") from None
    status = response.status_code
    if status in (401, 403):
        raise ClinicLookupError("configuration", "Geoapify rejected the API key or its restrictions. Check the key and allowed access in your Geoapify project.", "api_key_rejected")
    if status == 429:
        raise ClinicLookupError("geoapify", "Geoapify's request limit was reached. Wait before retrying and check your account quota.", "rate_limited")
    if status >= 500:
        raise ClinicLookupError("geoapify", f"Geoapify returned HTTP {status}. Please retry shortly.", "provider_error")
    if status != 200:
        raise ClinicLookupError("geoapify", f"Geoapify returned HTTP {status}. Check the Places API configuration.", "request_rejected")
    try:
        data = response.json()
        if not isinstance(data, dict) or data.get("type") != "FeatureCollection" or not isinstance(data.get("features"), list):
            raise ValueError()
    except (ValueError, TypeError):
        raise ClinicLookupError("geoapify", "Geoapify returned an unreadable or unexpected response. Please retry.", "invalid_response") from None
    features = data["features"]
    _remember(key, features, 600)
    print(f"  [clinic_locator] Geoapify success: radius={radius_km:g} km, candidates={len(features)}")
    return features


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
    if not os.environ.get("GEMINI_API_KEY") or os.getenv("BURNSCOPE_CLINIC_FILTER", "on").lower() == "off":
        return [True] * len(names)
    key = ("filter", tuple(names), CLASSIFY_MODEL)
    hit = _cached(key)
    if hit is not None:
        return hit

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
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
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

        if len(results) != len(names) or any(type(flag) is not bool for flag in results):
            print(f"  [clinic_locator] LLM filter response invalid "
                  f"({len(results)} vs {len(names)}), failing open (keeping all).")
            return [True] * len(names)

        _remember(key, results, 86400)
        return results

    except Exception as e:
        print(f"  [clinic_locator] Optional Gemini filter failed ({type(e).__name__}); keeping candidates already screened by name rules.")
        return [True] * len(names)


def _haversine_km(lat1, lon1, lat2, lon2):
    """Approximate straight-line distance in km between two lat/lon points."""
    from math import radians, sin, cos, sqrt, atan2
    R = 6371.0
    d_lat = radians(lat2 - lat1)
    d_lon = radians(lon2 - lon1)
    a = sin(d_lat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(d_lon / 2) ** 2
    a = min(1.0, max(0.0, a))
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


MIN_OPTIONS = 4   # keep widening the radius (5 -> 15 -> 30 km) until at least this many are found


def _find_clinics(lat, lon, radius_km, deadline=None):
    features = _search_nearby_facilities(lat, lon, radius_km, deadline=deadline)
    candidates = []
    seen_ids = set()
    for feature in features:
        if not isinstance(feature, dict):
            continue
        props = feature.get("properties") or {}
        if not isinstance(props, dict):
            continue
        name = props.get("name")
        if not isinstance(name, str) or not name.strip():
            continue  # Do not invent a name for an unnamed object.
        if not _name_is_candidate(name):
            continue
        el_lat, el_lon = props.get("lat"), props.get("lon")
        if el_lat is None or el_lon is None:
            geometry = feature.get("geometry") or {}
            coords = geometry.get("coordinates") if isinstance(geometry, dict) else None
            if not isinstance(geometry, dict) or geometry.get("type") != "Point" or not isinstance(coords, (list, tuple)) or len(coords) < 2:
                continue
            el_lon, el_lat = coords[:2]  # GeoJSON order is longitude, latitude.
        try:
            el_lat, el_lon = float(el_lat), float(el_lon)
            if not (-90 <= el_lat <= 90 and -180 <= el_lon <= 180):
                continue
            distance = _haversine_km(lat, lon, el_lat, el_lon)
        except (TypeError, ValueError):
            continue
        if distance > radius_km:
            continue
        place_id = props.get("place_id")
        if place_id and place_id in seen_ids:
            continue
        if place_id:
            seen_ids.add(place_id)
        address = props.get("formatted") or props.get("address_line2") or " ".join(str(props[k]) for k in
            ("housenumber", "street", "suburb", "city", "postcode") if props.get(k))
        candidates.append({"name": name.strip(), "address": str(address or "Address not available"),
                           "distance_km": distance, "latitude": el_lat, "longitude": el_lon,
                           "place_id": place_id, "provider": PROVIDER,
                           "attribution": ATTRIBUTION})
    candidates.sort(key=lambda c: c["distance_km"])
    unique = []
    for c in candidates:
        if any(c["name"].casefold() == p["name"].casefold() and
               _haversine_km(c["latitude"], c["longitude"], p["latitude"], p["longitude"]) < 0.15
               for p in unique):
            continue
        unique.append(c)
    flags = _classify_facilities_with_llm([c["name"] for c in unique])
    return [dict(c, distance_km=round(c["distance_km"], 2)) for c, keep in zip(unique, flags) if keep]


def _failure(stage, note, origin=None, code="service_unavailable"):
    return {"clinics": [], "status": "unavailable", "failed_stage": stage,
            "origin": origin, "note": note, "error_code": code, "provider": PROVIDER,
            "instruction": "No verified clinic results are available. Do not invent facility names, distances, burn units or opening hours. State the lookup failed and offer to retry."}


def locate_nearby_clinics(location: str, radius_km: float = 5.0) -> dict:
    """Start within 5 km even if the agent asks for 20; widen only on valid sparse results.
    Returns existing 'clinics' field plus status, resolved origin, radius and provenance.
    No cached failures and no stale results from a different location are returned.
    """
    if not isinstance(location, str) or not location.strip():
        return _failure("input", "Please provide a city, neighbourhood or postal code.")
    try:
        if isinstance(radius_km, bool):
            raise ValueError()
        radius_km = float(radius_km)
        if not math.isfinite(radius_km) or not 0 < radius_km <= 30:
            raise ValueError()
    except (ValueError, TypeError):
        return _failure("input", "The search radius must be greater than 0 and at most 30 km.")
    if not os.getenv("GEOAPIFY_API_KEY", "").strip():
        return _failure("configuration", "GEOAPIFY_API_KEY is missing. Add it to .env and restart Streamlit.", code="missing_api_key")
    try:
        origin = _geocode_location(location.strip())
    except ClinicLookupError as exc:
        return _failure(exc.stage, "The place lookup service is unavailable. Please retry shortly.")
    if origin is None:
        return {"clinics": [], "status": "location_not_found", "failed_stage": "geocoding",
                "note": "No matching place was found. Please include the city and country.",
                "instruction": "Ask for a clearer location. Do not invent facilities."}
    deadline = time.monotonic() + SEARCH_BUDGET_SECONDS
    clinics, used_radius, partial = [], min(5.0, radius_km), False
    for radius in dict.fromkeys((min(5.0, radius_km), 15.0, 30.0)):
        try:
            found = _find_clinics(origin["lat"], origin["lon"], radius, deadline=deadline)
        except ClinicLookupError as exc:
            if not clinics:
                return _failure(exc.stage, str(exc), origin, exc.code)
            partial = True
            break  # Preserve a successful smaller-radius list if expansion fails.
        # Retain real candidates across sparse expansions before sorting again.
        combined = clinics + found
        by_id = {}
        for entry in combined:
            identity = entry.get("place_id") or (entry["name"].casefold(), entry.get("latitude"), entry.get("longitude"))
            by_id[identity] = entry
        clinics = sorted(by_id.values(), key=lambda c: c["distance_km"])
        used_radius = radius
        if len(clinics) >= MIN_OPTIONS:
            break
    for clinic in clinics:
        clinic["origin_name"] = origin["display_name"]
    result = {"clinics": clinics[:10], "status": "partial" if partial else "ok" if clinics else "no_results",
              "origin": origin, "search_radius_km": used_radius,
              "distance_type": "straight_line_km", "source": ATTRIBUTION, "provider": PROVIDER,
              "attribution": ATTRIBUTION,
              "instruction": "State the resolved origin.display_name before distances. List only returned facilities. They are map listings, not verified burn specialists. Include attribution: Powered by Geoapify (https://www.geoapify.com/); OpenStreetMap contributors (https://www.openstreetmap.org/copyright). Do not claim opening hours, bed availability or burn units."}
    if partial:
        result["note"] = "Showing the successful smaller-area search; a wider search failed. The list may be incomplete."
    elif not clinics:
        result["note"] = "The search completed but found no named matching facilities. This does not prove no care is available."
    elif used_radius > 5:
        result["note"] = f"The search was widened to {used_radius:g} km because fewer than four matches were found nearby."
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test nearby-clinic lookup without loading image models or the database.")
    parser.add_argument("location", help="Neighbourhood, city and country")
    parser.add_argument("--radius", type=float, default=5.0)
    args = parser.parse_args()
    print(json.dumps(locate_nearby_clinics(args.location, args.radius), indent=2, ensure_ascii=False))
