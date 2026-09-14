"""Google Solar API lookups: does this roof already have solar panels?

The Solar API has no range query. `buildingInsights:findClosest` returns exactly one
building -- the one whose centroid is nearest the query point -- with no radius, bbox
or batch mode, so a neighbourhood costs one call per house. (`dataLayers:get` does take
a radius, but it returns flux/DSM/RGB rasters and says nothing about installed panels.)
This walks `data/house/manifest.csv`, caches one JSON per polygon id, and can fold the
results back into the manifest as extra columns.

Usage:
    python get_solar_insights.py --update-manifest
    python get_solar_insights.py --limit 5              # try a handful first
    python get_solar_insights.py "1600 Amphitheatre Pkwy, Mountain View, CA"
    python get_solar_insights.py --lat 37.309 --lon -121.9898

Requires GOOGLE_MAPS_API_KEY with the Solar API enabled on the key.
"""

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path

import requests

from get_building_footprint import geocode_address

SOLAR_URL = "https://solar.googleapis.com/v1/buildingInsights:findClosest"

# Shared 600 QPM budget across Building Insights and Data Layers. 0.1s between calls
# keeps a single-threaded run just under it without needing a real rate limiter.
CALL_INTERVAL_S = 0.1

# findClosest snaps to *Google's* nearest building, which need not be the OSM footprint
# the centroid came from. Suburban lots put neighbouring houses ~20m apart, so anything
# past this is more likely the house next door than a projection disagreement.
MAX_MATCH_DISTANCE_M = 15.0

MANIFEST_DEFAULT = "data/house/manifest.csv"
CACHE_DIR_DEFAULT = "data/house/solar"

# The manifest's own columns, written by the notebooks. Solar columns follow them.
BASE_COLUMNS = [
    "id", "image", "has_image", "address", "area_sqft", "area_m2",
    "width_px", "height_px", "center_lat", "center_lon", "osm_type", "osm_id",
]

# Appended to the manifest by --update-manifest, in this order.
SOLAR_COLUMNS = [
    "solar_status",
    "has_panels",
    "panels_capture_date",
    "solar_match_m",
    "solar_imagery_date",
    "solar_imagery_quality",
    "max_panel_count",
    "solar_building_id",
]


def fetch_building_insights(
    lat: float,
    lon: float,
    api_key: str,
    required_quality: str = "HIGH",
    detected_arrays: bool = True,
    attempts: int = 4,
    timeout: int = 30,
) -> dict | None:
    """One findClosest call. Returns the response, or None if the building isn't covered.

    A 404 is a normal outcome, not an error: it means no building meeting
    `required_quality` was found near the point. Coverage is patchy outside the US, EU
    and Japan, and DETECTED_ARRAYS in particular only exists where Google has
    high-resolution aerial imagery -- so a 404 at HIGH may well succeed at MEDIUM,
    minus the array detection."""
    params = {
        "location.latitude": lat,
        "location.longitude": lon,
        "requiredQuality": required_quality,
        "key": api_key,
    }
    if detected_arrays:
        params["additionalInsights"] = "DETECTED_ARRAYS"

    retryable = {429, 500, 502, 503, 504}
    delay = 2.0

    for attempt in range(attempts):
        last = attempt == attempts - 1
        try:
            resp = requests.get(SOLAR_URL, params=params, timeout=timeout)
            if resp.status_code == 404:
                return None
            if resp.status_code in retryable:
                if last:
                    raise RuntimeError(
                        f"Solar API gave up after {attempts} attempts "
                        f"(last status {resp.status_code}): {resp.text[:200]}"
                    )
                time.sleep(delay)
                delay *= 2
                continue
            if resp.status_code == 403:
                raise RuntimeError(
                    "Solar API returned 403. Enable the Solar API on the project and "
                    f"check the key's API restrictions: {resp.text[:200]}"
                )
            resp.raise_for_status()
            return resp.json()
        except (requests.Timeout, requests.ConnectionError) as exc:
            if last:
                raise RuntimeError(f"Solar API unreachable after {attempts} attempts") from exc
            time.sleep(delay)
            delay *= 2

    raise AssertionError("unreachable")


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres. Used only at building scale, where the choice of
    earth radius is far below the noise in either centroid."""
    r = 6371008.8
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def format_date(date: dict | None) -> str:
    """Google's Date message ({year, month, day}) as YYYY-MM-DD."""
    if not date or "year" not in date:
        return ""
    return f"{date['year']:04d}-{date.get('month', 0):02d}-{date.get('day', 0):02d}"


def summarize(insights: dict | None, lat: float, lon: float, max_match_m: float) -> dict:
    """Flatten a findClosest response into the manifest columns.

    `solar_status` carries why a row has no panel verdict, so an empty `has_panels` can
    be told apart from a confident "no": NOT_FOUND (no coverage), MISMATCH (the returned
    building is too far from our footprint to be the same one), DATA_UNAVAILABLE (Google
    covers the building but not with imagery it will run detection on)."""
    empty = dict.fromkeys(SOLAR_COLUMNS, "")

    if insights is None:
        return {**empty, "solar_status": "NOT_FOUND"}

    center = insights.get("center", {})
    distance = haversine_m(lat, lon, center.get("latitude", lat), center.get("longitude", lon))

    row = {
        **empty,
        "solar_match_m": f"{distance:.1f}",
        "solar_imagery_date": format_date(insights.get("imageryDate")),
        "solar_imagery_quality": insights.get("imageryQuality", ""),
        "max_panel_count": insights.get("solarPotential", {}).get("maxArrayPanelsCount", ""),
        # "buildings/ChIJ..." -- the only handle for spotting two footprints that
        # collapsed onto the same Google building.
        "solar_building_id": insights.get("name", "").rsplit("/", 1)[-1],
    }

    if distance > max_match_m:
        return {**row, "solar_status": "MISMATCH"}

    arrays = insights.get("detectedArrays")
    if not arrays:
        return {**row, "solar_status": "NO_DETECTION_DATA"}

    # DETECTION_STATUS_ARRAYS_DETECTED -> ARRAYS_DETECTED
    status = arrays.get("detectionStatus", "").replace("DETECTION_STATUS_", "")
    row["solar_status"] = status or "UNSPECIFIED"
    row["panels_capture_date"] = format_date(arrays.get("latestCaptureDate"))
    if status == "ARRAYS_DETECTED":
        row["has_panels"] = "1"
    elif status == "NO_ARRAYS_DETECTED":
        row["has_panels"] = "0"
    return row


def load_or_fetch(
    house_id: str,
    lat: float,
    lon: float,
    api_key: str,
    cache_dir: Path,
    required_quality: str = "HIGH",
    detected_arrays: bool = True,
    refresh: bool = False,
) -> tuple[dict | None, bool]:
    """Cached findClosest, one JSON per polygon id. Returns (insights, was_cached).

    Keyed on the polygon id rather than the address so the join back to the manifest is
    trivial -- with the same caveat as everywhere else in this project: the id is
    content-addressed, so a re-surveyed footprint gets a new id and re-fetches.

    A 404 is cached too, as {"error": "NOT_FOUND"}. Coverage gaps are a property of the
    area, not a transient failure, and re-asking every run would spend a paid call to
    learn the same thing."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{house_id}.json"

    if path.exists() and not refresh:
        cached = json.loads(path.read_text())
        return (None if cached.get("error") == "NOT_FOUND" else cached), True

    insights = fetch_building_insights(
        lat, lon, api_key, required_quality, detected_arrays
    )
    path.write_text(json.dumps(insights if insights is not None else {"error": "NOT_FOUND"}, indent=2))
    return insights, False


def read_manifest(path: Path) -> tuple[list[str], list[dict]]:
    if not path.exists():
        return [], []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or []), list(reader)


def write_manifest(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    """Rewrite the manifest via a temp file, so an interrupted write can't truncate the
    only lookup table mapping opaque image ids back to addresses."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, restval="", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def manifest_row(house_id: str, **fields) -> dict:
    """A manifest row for a polygon id. Every geometry column is derived from the id
    itself, so callers only supply what the id cannot encode: address, area, OSM
    provenance. `has_image` is a placeholder -- `merge_manifest` sets it from disk."""
    from satellite_cache import parse_polygon_id, polygon_id_bbox

    x0, y0, x1, y1 = parse_polygon_id(house_id)
    min_lon, min_lat, max_lon, max_lat = polygon_id_bbox(house_id)
    row = {
        "id": house_id,
        "image": f"{house_id}_raw.png",
        "has_image": 0,
        "address": "",
        "area_sqft": "",
        "area_m2": "",
        "width_px": x1 - x0,
        "height_px": y1 - y0,
        "center_lat": round((min_lat + max_lat) / 2, 7),
        "center_lon": round((min_lon + max_lon) / 2, 7),
        "osm_type": "",
        "osm_id": "",
    }
    row.update(fields)
    return row


def merge_manifest(
    manifest_path: Path,
    candidates: list[dict] = (),
    *,
    solar_dir: Path,
    api_key: str | None = None,
    image_dir: Path | None = None,
    fetch: bool = True,
    refresh: bool = False,
    limit: int | None = None,
    max_match_m: float = MAX_MATCH_DISTANCE_M,
    required_quality: str = "HIGH",
    detected_arrays: bool = True,
    log=lambda msg: print(msg, file=sys.stderr),
) -> tuple[list[dict], int, int]:
    """Merge rows into the manifest, fill blank solar columns, write. Returns
    (all rows, added, filled).

    Merging rather than rebuilding is the point: the manifest accumulates state no
    single run can re-derive -- solar columns, hand corrections, and rows belonging to
    other addresses or radii. `candidates` not already present are appended; rows
    already there keep their values.

    Two exceptions to "keep their values". `has_image` is re-read from `image_dir`,
    since it describes the filesystem and goes stale as soon as imagery is fetched.
    And blank solar columns are filled -- blank only, so a manual correction survives.
    `refresh=True` re-fetches and overwrites every row's solar columns instead.

    Cached lookups are free; a house with no cached response costs an API call, and
    none is made unless `fetch` and `api_key` are both set."""
    prior_columns, prior_rows = read_manifest(manifest_path)
    existing = {r["id"]: r for r in prior_rows}

    rows = list(existing.values())
    added = [dict(c) for c in candidates if c["id"] not in existing]
    rows.extend(added)

    if image_dir is not None:
        for r in rows:
            r["has_image"] = int((image_dir / r["image"]).exists())

    log(f"{len(existing)} row(s) already in manifest, {len(added)} added")

    todo = [r for r in rows if refresh or not r.get("solar_status")]
    if limit:
        todo = todo[:limit]
    uncached = [r for r in todo if not (solar_dir / f"{r['id']}.json").exists()]
    log(f"{len(todo)} row(s) needing solar info, "
        f"{len(uncached)} of them an API call")

    filled = 0
    for r in todo:
        cached = (solar_dir / f"{r['id']}.json").exists()
        if not cached and not (fetch and api_key):
            continue  # would cost a call we are not authorised or willing to make
        lat, lon = float(r["center_lat"]), float(r["center_lon"])
        insights, was_cached = load_or_fetch(
            r["id"], lat, lon, api_key, solar_dir,
            required_quality, detected_arrays, refresh,
        )
        if not was_cached:
            time.sleep(CALL_INTERVAL_S)
        r.update(summarize(insights, lat, lon, max_match_m))
        filled += 1

    log(f"solar columns filled for {filled} row(s)")

    # Preserve any column an earlier run or another tool added that we do not know about
    extra = [c for c in prior_columns if c not in BASE_COLUMNS + SOLAR_COLUMNS]
    columns = BASE_COLUMNS + extra + SOLAR_COLUMNS

    # fetched first, then by address; unaddressed buildings sort last within each group
    rows.sort(key=lambda r: (-int(r.get("has_image") or 0), r.get("address") or "~", r["id"]))
    write_manifest(manifest_path, columns, rows)

    return rows, len(added), filled


def process_manifest(
    manifest_path: Path,
    cache_dir: Path,
    api_key: str,
    required_quality: str = "HIGH",
    detected_arrays: bool = True,
    max_match_m: float = MAX_MATCH_DISTANCE_M,
    limit: int | None = None,
    refresh: bool = False,
    update_manifest: bool = False,
) -> list[dict]:
    """CLI path: fill solar columns for rows already in the manifest.

    `--update-manifest` is what writes; without it this is a dry run that reports what
    it found and leaves the file alone. The solar JSON cache is populated either way,
    so a later write costs nothing."""
    _, before = read_manifest(manifest_path)
    print(f"{len(before)} house(s) from {manifest_path}", file=sys.stderr)

    if not update_manifest:
        # Fill the cache without touching the file, then report from a throwaway copy.
        rows = [dict(r) for r in before]
        todo = [r for r in rows if refresh or not r.get("solar_status")][:limit or None]
        for n, r in enumerate(todo, 1):
            lat, lon = float(r["center_lat"]), float(r["center_lon"])
            insights, was_cached = load_or_fetch(
                r["id"], lat, lon, api_key, cache_dir,
                required_quality, detected_arrays, refresh,
            )
            if not was_cached:
                time.sleep(CALL_INTERVAL_S)
            r.update(summarize(insights, lat, lon, max_match_m))
            print(f"  [{n}/{len(todo)}] {r['id']} {r['solar_status']:<18} "
                  f"{r.get('address', '')}", file=sys.stderr)
        print("Dry run -- pass --update-manifest to write these columns", file=sys.stderr)
    else:
        rows, _, _ = merge_manifest(
            manifest_path,
            solar_dir=cache_dir,
            api_key=api_key,
            refresh=refresh,
            limit=limit,
            max_match_m=max_match_m,
            required_quality=required_quality,
            detected_arrays=detected_arrays,
        )
        print(f"Updated {manifest_path} with {len(SOLAR_COLUMNS)} solar columns",
              file=sys.stderr)

    detected = sum(1 for r in rows if r.get("has_panels") == "1")
    known = sum(1 for r in rows if r.get("has_panels") in ("0", "1"))
    print(f"{detected}/{known} with panels detected "
          f"({len(rows) - known} without a verdict)", file=sys.stderr)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("address", nargs="?", help="Look up a single address instead of the manifest")
    parser.add_argument("--lat", type=float, help="Latitude (alternative to address)")
    parser.add_argument("--lon", type=float, help="Longitude (alternative to address)")
    parser.add_argument("--manifest", default=MANIFEST_DEFAULT)
    parser.add_argument("--cache-dir", default=CACHE_DIR_DEFAULT)
    parser.add_argument(
        "--update-manifest",
        action="store_true",
        help="Write the results back into the manifest as extra columns",
    )
    parser.add_argument(
        "--required-quality",
        choices=["HIGH", "MEDIUM", "BASE"],
        default="HIGH",
        help="Minimum imagery quality. Panel detection needs HIGH; lower values widen "
        "coverage but return no detectedArrays (default: HIGH)",
    )
    parser.add_argument(
        "--max-match-distance",
        type=float,
        default=MAX_MATCH_DISTANCE_M,
        help="Reject a result whose building centroid is further than this from our "
        f"footprint's, in meters (default: {MAX_MATCH_DISTANCE_M})",
    )
    parser.add_argument("--limit", type=int, help="Only process the first N houses")
    parser.add_argument("--refresh", action="store_true", help="Re-fetch cached responses")
    parser.add_argument(
        "--no-detected-arrays",
        action="store_true",
        help="Skip the DETECTED_ARRAYS insight and return solar potential only",
    )
    args = parser.parse_args()

    api_key = os.environ.get("GOOGLE_MAPS_API_KEY")
    if not api_key:
        parser.error("GOOGLE_MAPS_API_KEY is not set")

    detected_arrays = not args.no_detected_arrays

    if args.address or (args.lat is not None and args.lon is not None):
        if args.lat is not None and args.lon is not None:
            lat, lon = args.lat, args.lon
        else:
            lat, lon = geocode_address(args.address)
            print(f"Geocoded {args.address!r} -> ({lat}, {lon})", file=sys.stderr)

        insights = fetch_building_insights(
            lat, lon, api_key, args.required_quality, detected_arrays
        )
        if insights is None:
            print(
                f"No building covered at ({lat}, {lon}) at {args.required_quality} quality",
                file=sys.stderr,
            )
            sys.exit(1)
        print(json.dumps(insights, indent=2))
        summary = summarize(insights, lat, lon, args.max_match_distance)
        print(
            f"status={summary['solar_status']} "
            f"has_panels={summary['has_panels'] or '?'} "
            f"captured={summary['panels_capture_date'] or '?'} "
            f"match={summary['solar_match_m']}m",
            file=sys.stderr,
        )
        return

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        parser.error(
            f"{manifest_path} not found -- run notebooks/batch_image_fetch.ipynb first, "
            "or pass an address / --lat --lon for a single lookup"
        )

    process_manifest(
        manifest_path,
        Path(args.cache_dir),
        api_key,
        required_quality=args.required_quality,
        detected_arrays=detected_arrays,
        max_match_m=args.max_match_distance,
        limit=args.limit,
        refresh=args.refresh,
        update_manifest=args.update_manifest,
    )


if __name__ == "__main__":
    main()
