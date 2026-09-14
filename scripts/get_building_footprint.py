"""Address -> geocode -> Overpass query -> nearest building footprint.

Usage:
    python get_building_footprint.py "1600 Amphitheatre Pkwy, Mountain View, CA"
    python get_building_footprint.py "..." --radius 75 --out data/footprint.geojson
    python get_building_footprint.py "..." --geocoder google   # requires GOOGLE_MAPS_API_KEY
"""

import argparse
import json
import math
import os
import sys
import time

import requests
from shapely.geometry import LineString, Point, Polygon, box, mapping
from shapely.ops import unary_union

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
GOOGLE_GEOCODE_URL = "https://maps.googleapis.com/maps/api/geocode/json"
# Overridable, because the public instance goes down outright often enough that the
# retry loop below cannot save a run -- see the message it raises when it gives up.
OVERPASS_URL = os.environ.get(
    "OVERPASS_URL", "https://overpass-api.de/api/interpreter"
)

# Nominatim's usage policy requires a descriptive User-Agent identifying the app.
USER_AGENT = "DeepSat/0.1 (building-footprint-script)"


def geocode_nominatim(address: str) -> tuple[float, float]:
    resp = requests.get(
        NOMINATIM_URL,
        params={"q": address, "format": "json", "limit": 1},
        headers={"User-Agent": USER_AGENT},
        timeout=10,
    )
    resp.raise_for_status()
    results = resp.json()
    if not results:
        raise ValueError(f"No geocoding result for address: {address!r}")
    return float(results[0]["lat"]), float(results[0]["lon"])


def geocode_google(address: str, api_key: str) -> tuple[float, float]:
    resp = requests.get(
        GOOGLE_GEOCODE_URL,
        params={"address": address, "key": api_key},
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json()
    if data.get("status") != "OK" or not data.get("results"):
        raise ValueError(f"Google geocoding failed for {address!r}: {data.get('status')}")
    loc = data["results"][0]["geometry"]["location"]
    return loc["lat"], loc["lng"]


def geocode_address(address: str, geocoder: str = "nominatim") -> tuple[float, float]:
    if geocoder == "google":
        api_key = os.environ.get("GOOGLE_MAPS_API_KEY")
        if not api_key:
            raise EnvironmentError("GOOGLE_MAPS_API_KEY is not set")
        return geocode_google(address, api_key)
    return geocode_nominatim(address)


def run_overpass_query(query: str, attempts: int = 4, timeout: int = 90) -> list[dict]:
    """POST an Overpass QL query, retrying on the failures its public instance throws
    routinely.

    overpass-api.de is a free shared service and sheds load with 429 (rate limited),
    503 (slot unavailable) and 504 (gateway timeout) far more often than most APIs.
    Those are transient and worth retrying with backoff; a 400 means the query itself
    is malformed and retrying would just be rude, so it raises immediately."""
    retryable = {429, 502, 503, 504}
    exhausted = (
        f"Overpass gave up after {attempts} attempts. Its public instance is often "
        "briefly overloaded -- wait a minute and retry, or point OVERPASS_URL at "
        "another mirror such as https://overpass.kumi.systems/api/interpreter"
    )
    delay = 2.0

    for attempt in range(attempts):
        last = attempt == attempts - 1
        try:
            resp = requests.post(
                OVERPASS_URL,
                data={"data": query},
                headers={"User-Agent": USER_AGENT},
                timeout=timeout,
            )
            if resp.status_code in retryable:
                if last:
                    raise RuntimeError(f"{exhausted} (last status {resp.status_code})")
                time.sleep(delay)
                delay *= 2
                continue
            resp.raise_for_status()
            return resp.json().get("elements", [])
        except (requests.Timeout, requests.ConnectionError) as exc:
            if last:
                raise RuntimeError(exhausted) from exc
            time.sleep(delay)
            delay *= 2

    raise AssertionError("unreachable")


def query_overpass_buildings(lat: float, lon: float, radius: int) -> list[dict]:
    return run_overpass_query(f"""
    [out:json];
    way["building"](around:{radius},{lat},{lon});
    out geom;
    """)


def elements_to_polygons(elements: list[dict]) -> list[tuple[dict, Polygon]]:
    polygons = []
    for el in elements:
        geometry = el.get("geometry")
        if not geometry or len(geometry) < 4:
            continue
        coords = [(pt["lon"], pt["lat"]) for pt in geometry]
        polygons.append((el, Polygon(coords)))
    return polygons


SQFT_PER_SQM = 10.763910416709722


def polygon_area_m2(polygon: Polygon) -> float:
    """Ground area of a lat/lon polygon in square metres.

    Shapely's `.area` on lat/lon coordinates is in square degrees, which is meaningless
    as an area. This projects onto a local equirectangular plane centred on the polygon
    -- exact enough well below a percent at building scale -- using the WGS84 series for
    metres per degree at that latitude.

    Note this is the *footprint* area: the ground outline. It is not a house's living
    area, which counts every storey and excludes the garage."""
    lat0 = math.radians(polygon.centroid.y)

    m_per_deg_lat = (
        111132.92 - 559.82 * math.cos(2 * lat0) + 1.175 * math.cos(4 * lat0)
        - 0.0023 * math.cos(6 * lat0)
    )
    m_per_deg_lon = (
        111412.84 * math.cos(lat0) - 93.5 * math.cos(3 * lat0)
        + 0.118 * math.cos(5 * lat0)
    )

    lon0, lat0_deg = polygon.centroid.x, polygon.centroid.y
    projected = Polygon(
        [
            ((x - lon0) * m_per_deg_lon, (y - lat0_deg) * m_per_deg_lat)
            for x, y in polygon.exterior.coords
        ]
    )
    return projected.area


def polygon_area_sqft(polygon: Polygon) -> float:
    """Ground footprint area in square feet. See `polygon_area_m2` on what this measures."""
    return polygon_area_m2(polygon) * SQFT_PER_SQM


def format_address(tags: dict) -> str:
    """OSM address tags as a single line, e.g. "1095 HAPPY VALLEY AVE, SAN JOSE, CA".
    Returns "" when the building has no `addr:housenumber` + `addr:street`. Both parts
    are required: a lone city or postcode does not identify a house."""
    street = " ".join(
        p for p in (tags.get("addr:housenumber"), tags.get("addr:street")) if p
    )
    if not street:
        return ""
    return ", ".join(
        p for p in (street, tags.get("addr:city"), tags.get("addr:postcode")) if p
    )


def query_overpass_roads(lat: float, lon: float, radius: int) -> list[dict]:
    return run_overpass_query(f"""
    [out:json];
    way["highway"](around:{radius},{lat},{lon});
    out geom;
    """)


def elements_to_lines(elements: list[dict]) -> list[tuple[dict, LineString]]:
    lines = []
    for el in elements:
        geometry = el.get("geometry")
        if not geometry or len(geometry) < 2:
            continue
        coords = [(pt["lon"], pt["lat"]) for pt in geometry]
        lines.append((el, LineString(coords)))
    return lines


def polygons_bounding_box(polygons: list[tuple[dict, Polygon]]) -> tuple[float, float, float, float]:
    return unary_union([poly for _, poly in polygons]).bounds


def clip_lines_to_bbox(
    lines: list[tuple[dict, LineString]], bounds: tuple[float, float, float, float]
) -> list[tuple[dict, LineString]]:
    bbox = box(*bounds)
    clipped = []
    for el, line in lines:
        intersection = line.intersection(bbox)
        if intersection.is_empty:
            continue
        geoms = intersection.geoms if hasattr(intersection, "geoms") else [intersection]
        for geom in geoms:
            if geom.geom_type == "LineString" and not geom.is_empty:
                clipped.append((el, geom))
    return clipped


def nearest_building(
    polygons: list[tuple[dict, Polygon]], lat: float, lon: float
) -> tuple[dict, Polygon]:
    if not polygons:
        raise ValueError("No building footprints found near this location")
    point = Point(lon, lat)
    return min(polygons, key=lambda item: item[1].distance(point))


def to_geojson_feature(element: dict, polygon: Polygon) -> dict:
    return {
        "type": "Feature",
        "properties": {"osm_id": element.get("id"), "tags": element.get("tags", {})},
        "geometry": mapping(polygon),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("address", help="Street address to look up")
    parser.add_argument(
        "--geocoder", choices=["nominatim", "google"], default="nominatim"
    )
    parser.add_argument(
        "--radius", type=int, default=50, help="Overpass search radius in meters"
    )
    parser.add_argument(
        "--out", help="Path to write GeoJSON output (default: print to stdout)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Write every building found in the radius, not just the nearest",
    )
    args = parser.parse_args()

    lat, lon = geocode_address(args.address, args.geocoder)
    print(f"Geocoded {args.address!r} -> ({lat}, {lon})", file=sys.stderr)

    elements = query_overpass_buildings(lat, lon, args.radius)
    polygons = elements_to_polygons(elements)
    print(f"Found {len(polygons)} building(s) within {args.radius}m", file=sys.stderr)

    if args.all:
        features = [to_geojson_feature(el, poly) for el, poly in polygons]
    else:
        el, poly = nearest_building(polygons, lat, lon)
        features = [to_geojson_feature(el, poly)]

    geojson = {"type": "FeatureCollection", "features": features}
    output = json.dumps(geojson, indent=2)

    if args.out:
        with open(args.out, "w") as f:
            f.write(output)
        print(f"Wrote {len(features)} feature(s) to {args.out}", file=sys.stderr)
    else:
        print(output)


if __name__ == "__main__":
    main()
