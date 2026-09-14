"""Batch satellite image fetch: the script form of notebooks/batch_image_fetch.ipynb.

Collects building footprints around an address, filters them to plausible houses, saves
them as GeoJSON, fetches one satellite crop per house, and merges the result into
`manifest.csv` along with whether each roof already has solar panels.

Two things this adds over the notebook. A radius above `--sub-radius` (500 m) is swept
as a lattice of smaller Overpass circles instead of one wide query, and `--max-houses`
caps the run: regions are visited nearest-centre first and collection stops as soon as
the cap is reached, so a large radius can be explored a slice at a time.

Everything expensive is cached and re-runs are cheap: tiles under `data/tiles/`, solar
responses under `data/house/solar/`, crops skipped when already on disk, and the
manifest merged rather than rewritten. An interrupted run loses only the manifest
merge -- re-run it and the caches make the repeat close to free.

Usage:
    python scripts/batch_fetch.py "1095 HAPPY VALLEY AVE, SAN JOSE, CA" --radius 200
    python scripts/batch_fetch.py "..." --radius 1200 --max-houses 400
    python scripts/batch_fetch.py "..." --radius 1200 --dry-run   # cost preview only

Collecting and filtering footprints needs no API key -- that is Overpass and Nominatim,
both free -- so `--dry-run` and `--no-images --no-solar` work without one. Imagery needs
GOOGLE_MAPS_API_KEY with the Maps Static API enabled; solar needs the Solar API on the
same key, and spends nothing for a house already cached in `data/house/solar/`.
"""

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path

from shapely.geometry import Polygon

from get_building_footprint import (
    elements_to_polygons,
    format_address,
    geocode_address,
    polygon_area_m2,
    polygon_area_sqft,
    query_overpass_buildings,
)
from get_solar_insights import (
    haversine_m,
    manifest_row,
    merge_manifest,
    read_manifest,
)
from satellite_cache import (
    GOOGLE_ATTRIBUTION,
    SatelliteTileCache,
    parse_polygon_id,
    polygon_id,
    polygon_id_bbox,
)

# Overpass answers one `around:` circle per query. Wide circles do work -- 3 km returns
# ~28k buildings in about 12s -- but the public instance sheds load under pressure and a
# shed 3 km query costs the whole sweep, where a shed 500 m one costs a retry.
MAX_SUB_RADIUS_M = 500

# Query circles sit on a square lattice whose cells they circumscribe. Shrinking the
# cell 5% is margin for the flat-earth metre-to-degree conversion in `offset_latlon`,
# which is off by a few tenths of a percent here; without it a cell corner could land
# just outside its own circle and leave an unqueried sliver.
SPACING_SAFETY = 0.95

MIN_SQFT_DEFAULT = 800
MAX_SQFT_DEFAULT = 6000
MAX_HOUSES_DEFAULT = 500

HOUSE_DIR_DEFAULT = "data/house"
TILES_DIR_DEFAULT = "data/tiles"

# Building Insights free tier, for the warning in `preview`. One call per house, so a
# wide enough sweep is the thing that exhausts it.
SOLAR_FREE_CALLS_PER_MONTH = 10_000


def _log(msg: str) -> None:
    print(msg, file=sys.stderr)


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def offset_latlon(lat: float, lon: float, east_m: float, north_m: float) -> tuple[float, float]:
    """Move a point by metres east and north, on a local flat-earth approximation.

    Matches `satellite_cache.bbox_around_point`: good to a few tenths of a percent,
    which at the few-kilometre scale this places query centres at is metres."""
    dlat = north_m / 111_320.0
    dlon = east_m / (111_320.0 * max(math.cos(math.radians(lat)), 1e-6))
    return lat + dlat, lon + dlon


def plan_subregions(
    lat: float, lon: float, radius_m: float, sub_radius_m: float = MAX_SUB_RADIUS_M
) -> list[tuple[float, float, float]]:
    """Cover the disc of `radius_m` with query circles no bigger than `sub_radius_m`.

    A circle of radius r circumscribes a square of side r*sqrt(2), so circles on a
    lattice of that side cover every cell completely and therefore the whole plane. A
    cell is kept when its square intersects the requested disc, which is the smallest
    set that still guarantees full coverage.

    Returned nearest-centre first, so truncating the sweep at `--max-houses` yields a
    roughly centre-out area rather than an arbitrary edge of one."""
    if radius_m <= sub_radius_m:
        return [(lat, lon, radius_m)]

    spacing = sub_radius_m * math.sqrt(2) * SPACING_SAFETY
    # +0.5 because a cell whose centre sits at radius_m + spacing/2 still overlaps the
    # disc; ceil alone drops that outer ring when radius_m is a multiple of spacing.
    n = math.ceil(radius_m / spacing + 0.5)

    cells = []
    for iy in range(-n, n + 1):
        for ix in range(-n, n + 1):
            dx, dy = ix * spacing, iy * spacing
            # distance from the region centre to the nearest point of this cell's square
            nearest = math.hypot(
                max(0.0, abs(dx) - spacing / 2), max(0.0, abs(dy) - spacing / 2)
            )
            if nearest > radius_m:
                continue
            sub_lat, sub_lon = offset_latlon(lat, lon, dx, dy)
            cells.append((math.hypot(dx, dy), sub_lat, sub_lon))

    cells.sort()
    return [(sub_lat, sub_lon, sub_radius_m) for _, sub_lat, sub_lon in cells]


def filter_houses(
    polygons: list[tuple[dict, Polygon]],
    *,
    min_sqft: float = MIN_SQFT_DEFAULT,
    max_sqft: float = MAX_SQFT_DEFAULT,
    require_address: bool = True,
) -> tuple[list[dict], int, int]:
    """Keep the footprints that plausibly are houses. Returns (houses, dropped_size,
    dropped_address).

    Bulk-imported OSM data mixes houses with sheds, garages and apartment blocks, so
    trim to a size range before spending anything on imagery. `require_address` also
    drops footprints with no `addr:housenumber` + `addr:street`: those are usually
    outbuildings, they cannot be joined to external data, and an outbuilding's centroid
    sits close enough to the house it belongs to that a solar lookup would credit it
    with the neighbour's roof.

    The size is *footprint* area -- the ground outline, not a listing's square footage,
    which counts every storey and excludes the garage."""
    houses = []
    dropped_size = 0
    dropped_address = 0

    for element, poly in polygons:
        tags = element.get("tags", {})

        sqft = polygon_area_sqft(poly)
        if not (min_sqft <= sqft <= max_sqft):
            dropped_size += 1
            continue

        # both parts are required -- a lone city or postcode does not identify a house
        has_address = bool(tags.get("addr:housenumber") and tags.get("addr:street"))
        if require_address and not has_address:
            dropped_address += 1
            continue

        houses.append({
            "id": polygon_id(poly),
            "osm_id": element.get("id"),
            "osm_type": element.get("type", "way"),
            "area_sqft": round(sqft, 1),
            "area_m2": round(polygon_area_m2(poly), 1),
            "tags": tags,
            "polygon": poly,
        })

    return houses, dropped_size, dropped_address


def collect_houses(
    lat: float,
    lon: float,
    radius_m: float,
    *,
    min_sqft: float = MIN_SQFT_DEFAULT,
    max_sqft: float = MAX_SQFT_DEFAULT,
    require_address: bool = True,
    max_houses: int | None = None,
    sub_radius_m: float = MAX_SUB_RADIUS_M,
    log=_log,
) -> list[dict]:
    """Every house within `radius_m` of a point, at most `max_houses` of them.

    Sweeps `plan_subregions` in order and stops early once the cap is reached, so an
    ambitious radius does not have to be queried in full to get a usable slice.

    A house is kept when any of its footprint's vertices falls inside the requested
    radius, which is what Overpass's own `around:` means -- so a split sweep returns the
    same set the one wide query it replaces would have. Distance to the centroid is
    still what `dist_m` records and what orders the result, since that is the sensible
    reading of how far away a house is.

    Ids are `polygon_id`s, so the same building seen from two overlapping circles
    dedupes for free. Two *different* footprints quantising to one id would collide
    and lose an image; that is vanishingly unlikely at ~0.12 m and only warned about
    here rather than raised, so a long sweep is not lost to one bad footprint."""
    regions = plan_subregions(lat, lon, radius_m, sub_radius_m)
    if len(regions) > 1:
        log(f"radius {radius_m:g} m > {sub_radius_m:g} m: sweeping "
            f"{len(regions)} circles of {sub_radius_m:g} m")

    by_id: dict[str, dict] = {}
    collisions: list[tuple[str, object, object]] = []
    queried = 0

    for n, (sub_lat, sub_lon, sub_r) in enumerate(regions, 1):
        elements = query_overpass_buildings(sub_lat, sub_lon, int(round(sub_r)))
        polygons = elements_to_polygons(elements)
        kept, _, _ = filter_houses(
            polygons,
            min_sqft=min_sqft,
            max_sqft=max_sqft,
            require_address=require_address,
        )
        queried = n

        fresh = []
        for house in kept:
            nearest = min(
                haversine_m(lat, lon, y, x)
                for x, y in house["polygon"].exterior.coords
            )
            if nearest > radius_m:
                continue
            centroid = house["polygon"].centroid
            dist = haversine_m(lat, lon, centroid.y, centroid.x)
            seen = by_id.get(house["id"])
            if seen is not None:
                if seen["osm_id"] != house["osm_id"]:
                    collisions.append((house["id"], seen["osm_id"], house["osm_id"]))
                continue
            house["dist_m"] = round(dist, 1)
            fresh.append(house)

        # nearest first, so a cap that lands mid-region still takes the inner houses
        fresh.sort(key=lambda h: h["dist_m"])
        if max_houses is not None:
            fresh = fresh[: max_houses - len(by_id)]
        for house in fresh:
            by_id[house["id"]] = house

        log(f"  region {n}/{len(regions)} r={sub_r:g}m: {len(polygons)} buildings, "
            f"{len(kept)} pass filter, {len(fresh)} new (total {len(by_id)})")

        if max_houses is not None and len(by_id) >= max_houses:
            log(f"  reached --max-houses {max_houses} after {n} of {len(regions)} region(s)")
            break

    if collisions:
        log(f"WARNING: {len(collisions)} polygon id collision(s) -- distinct footprints "
            f"sharing one id, all but the first dropped:")
        for house_id, first, second in collisions[:5]:
            log(f"    {house_id}: kept OSM {first}, dropped {second}")

    houses = sorted(by_id.values(), key=lambda h: h["dist_m"])
    log(f"collected {len(houses)} house(s) within {radius_m:g} m "
        f"from {queried} of {len(regions)} region(s)")
    return houses


def write_geojson(path: Path, houses: list[dict], properties: dict, log=_log) -> None:
    """The collected footprints as GeoJSON, so they open in any GIS tool.

    A provenance record of what one run looked at -- `manifest.csv` is the table that
    accumulates state across runs."""
    feature_collection = {
        "type": "FeatureCollection",
        "properties": properties,
        "features": [
            {
                "type": "Feature",
                "properties": {k: v for k, v in h.items() if k != "polygon"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[x, y] for x, y in h["polygon"].exterior.coords]],
                },
            }
            for h in houses
        ],
    }
    with open(path, "w") as f:
        json.dump(feature_collection, f, indent=2)
    log(f"wrote {len(houses)} houses to {path}")


def preview(
    cache: SatelliteTileCache,
    houses: list[dict],
    *,
    image_dir: Path,
    manifest_path: Path,
    solar_dir: Path,
    log=_log,
) -> dict:
    """What this run would cost, before it spends anything.

    Tiles are shared: houses on a block sit in the same tile, so tile downloads scale
    with the area covered rather than the number of houses. Solar does not share --
    findClosest has no range query, so it is one call per house that has no cached
    response and no verdict in the manifest yet."""
    needed = set()
    for house in houses:
        i0, j0, i1, j1 = cache.tile_indices_for_bbox(polygon_id_bbox(house["id"]))
        needed.update((i, j) for i in range(i0, i1 + 1) for j in range(j0, j1 + 1))
    uncached_tiles = [t for t in needed if not cache.tile_path(*t).exists()]

    missing_images = [h for h in houses if not (image_dir / f"{h['id']}_raw.png").exists()]

    _, prior_rows = read_manifest(manifest_path)
    prior = {r["id"]: r for r in prior_rows}
    new_rows = [h for h in houses if h["id"] not in prior]
    solar_todo = [h for h in houses if not prior.get(h["id"], {}).get("solar_status")]
    solar_calls = [h for h in solar_todo if not (solar_dir / f"{h['id']}.json").exists()]

    log("")
    log(f"{len(houses)} house(s):")
    log(f"  crops to fetch    : {len(missing_images):>6}  ({len(houses) - len(missing_images)} already on disk)")
    log(f"  tiles touched     : {len(needed):>6}")
    log(f"  tile API calls    : {len(uncached_tiles):>6}  (uncached tiles; the rest are free)")
    log(f"  manifest rows new : {len(new_rows):>6}  ({len(houses) - len(new_rows)} already present)")
    log(f"  solar API calls   : {len(solar_calls):>6}  (of {len(solar_todo)} row(s) needing a verdict)")

    if len(solar_calls) > SOLAR_FREE_CALLS_PER_MONTH // 5:
        log(f"  NOTE: Building Insights bills past {SOLAR_FREE_CALLS_PER_MONTH:,} "
            f"calls/month; this run uses {len(solar_calls):,} of them")

    return {
        "tiles": len(needed),
        "tile_calls": len(uncached_tiles),
        "images_to_fetch": len(missing_images),
        "new_rows": len(new_rows),
        "solar_calls": len(solar_calls),
    }


def fetch_images(
    cache: SatelliteTileCache,
    houses: list[dict],
    image_dir: Path,
    *,
    progress_every: int = 100,
    log=_log,
) -> tuple[int, int, list[tuple[str, str]]]:
    """One crop per house as `{id}_raw.png`, skipping ids already on disk.

    The crop is the polygon's bounding box, and because the id encodes that box in world
    pixels every image comes out exactly the `w x h` its id claims -- checked here, since
    a mismatch means the tile grid and the id zoom have drifted apart.

    The separator is an underscore rather than a dot so the filename has a single suffix:
    `Path.stem` then yields `{id}_raw` cleanly, where `{id}.raw.png` would leave a stray
    `.raw` behind."""
    written = 0
    skipped = 0
    failures: list[tuple[str, str]] = []

    for n, house in enumerate(houses, 1):
        house_id = house["id"]
        out_path = image_dir / f"{house_id}_raw.png"
        if out_path.exists():
            skipped += 1
            continue
        try:
            image, _ = cache.get_region(polygon_id_bbox(house_id))
            x0, y0, x1, y1 = parse_polygon_id(house_id)
            if image.size != (x1 - x0, y1 - y0):
                raise ValueError(f"got {image.size}, id encodes {(x1 - x0, y1 - y0)}")
            # via a temp file: a crop truncated by an interrupt would be skipped as
            # "already fetched" by every later run
            tmp_path = out_path.with_suffix(".png.tmp")
            image.save(tmp_path, format="PNG")  # explicit: PIL reads the format off the
            tmp_path.replace(out_path)          # extension, and .tmp means nothing to it
            written += 1
        except Exception as exc:
            failures.append((house_id, repr(exc)))

        if progress_every and n % progress_every == 0:
            log(f"  {n}/{len(houses)} houses, {written} written, "
                f"{cache.downloads} tiles downloaded, {len(failures)} failed")

    return written, skipped, failures


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("address", nargs="?", help="Centre of the region to sweep")
    parser.add_argument("--lat", type=float, help="Centre latitude (alternative to address)")
    parser.add_argument("--lon", type=float, help="Centre longitude (alternative to address)")
    parser.add_argument(
        "--radius", type=float, default=200,
        help="Radius around the centre in metres (default: 200)",
    )
    parser.add_argument(
        "--max-houses", type=int, default=MAX_HOUSES_DEFAULT,
        help=f"Stop once this many houses are collected; 0 for no cap "
             f"(default: {MAX_HOUSES_DEFAULT})",
    )
    parser.add_argument(
        "--sub-radius", type=float, default=MAX_SUB_RADIUS_M,
        help=f"Split a larger radius into circles of at most this size "
             f"(default: {MAX_SUB_RADIUS_M})",
    )
    parser.add_argument("--min-sqft", type=float, default=MIN_SQFT_DEFAULT)
    parser.add_argument("--max-sqft", type=float, default=MAX_SQFT_DEFAULT)
    parser.add_argument(
        "--no-require-address", action="store_true",
        help="Keep footprints with no street address (they get no solar verdict)",
    )
    parser.add_argument("--house-dir", default=HOUSE_DIR_DEFAULT)
    parser.add_argument("--tiles-dir", default=TILES_DIR_DEFAULT)
    parser.add_argument("--manifest", help="Default: <house-dir>/manifest.csv")
    parser.add_argument("--geojson", help="Default: <house-dir>/<address>_r<radius>.json")
    parser.add_argument("--no-images", action="store_true", help="Skip the imagery step")
    parser.add_argument(
        "--no-solar", action="store_true",
        help="Fill solar columns from cache only, spending nothing",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Collect footprints and print the cost preview, then stop",
    )
    parser.add_argument("--geocoder", choices=["nominatim", "google"], default="nominatim")
    args = parser.parse_args()

    if args.address is None and (args.lat is None or args.lon is None):
        parser.error("give an address, or both --lat and --lon")
    if args.radius <= 0:
        parser.error("--radius must be positive")
    if args.sub_radius <= 0:
        parser.error("--sub-radius must be positive")

    api_key = os.environ.get("GOOGLE_MAPS_API_KEY")
    fetch_imagery = not (args.no_images or args.dry_run)
    if not api_key:
        if fetch_imagery:
            parser.error(
                "GOOGLE_MAPS_API_KEY is not set -- needed for imagery. Pass --no-images "
                "to build the manifest without it, or --dry-run for a cost preview"
            )
        if not args.no_solar:
            _log("GOOGLE_MAPS_API_KEY is not set -- solar columns fill from cache only")

    if args.lat is not None and args.lon is not None:
        lat, lon = args.lat, args.lon
        label = args.address or f"{lat}_{lon}"
    else:
        lat, lon = geocode_address(args.address, args.geocoder)
        label = args.address
        _log(f"geocoded {args.address!r} -> ({lat}, {lon})")

    house_dir = Path(args.house_dir)
    image_dir = house_dir / "images"
    solar_dir = house_dir / "solar"
    image_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.manifest) if args.manifest else house_dir / "manifest.csv"

    houses = collect_houses(
        lat, lon, args.radius,
        min_sqft=args.min_sqft,
        max_sqft=args.max_sqft,
        require_address=not args.no_require_address,
        max_houses=args.max_houses or None,
        sub_radius_m=args.sub_radius,
    )
    if not houses:
        _log("no houses passed the filter -- widen --radius, or --min-sqft/--max-sqft")
        sys.exit(1)

    cache = SatelliteTileCache(cache_dir=args.tiles_dir, api_key=api_key)
    preview(
        cache, houses,
        image_dir=image_dir,
        manifest_path=manifest_path,
        solar_dir=solar_dir,
        log=_log,
    )
    # nothing above this line has written anything, so --dry-run leaves no trace --
    # including not overwriting the GeoJSON a previous full run or the notebook left
    if args.dry_run:
        _log("\n--dry-run: stopping before any billed call")
        return

    geojson_path = (
        Path(args.geojson) if args.geojson
        else house_dir / f"{slugify(label)}_r{args.radius:g}.json"
    )
    write_geojson(geojson_path, houses, {
        "address": label,
        "center": [lat, lon],
        "radius_m": args.radius,
        "sub_radius_m": args.sub_radius,
        "max_houses": args.max_houses,
        "houses": len(houses),
        "min_sqft": args.min_sqft,
        "max_sqft": args.max_sqft,
        "require_address": not args.no_require_address,
    })

    if fetch_imagery:
        _log("")
        written, skipped, failures = fetch_images(cache, houses, image_dir)
        _log(f"crops: {written} written, {skipped} already present, {len(failures)} failed")
        _log(f"tiles: {cache.downloads} downloaded, {cache.hits} from cache")
        for house_id, err in failures[:10]:
            _log(f"  FAILED {house_id} {err}")
    else:
        _log("\nskipping imagery (--no-images)")

    _log("")
    rows, added, filled = merge_manifest(
        manifest_path,
        [
            manifest_row(
                h["id"],
                address=format_address(h.get("tags", {})),
                area_sqft=h.get("area_sqft", ""),
                area_m2=h.get("area_m2", ""),
                osm_type=h.get("osm_type", ""),
                osm_id=h.get("osm_id", ""),
            )
            for h in houses
        ],
        solar_dir=solar_dir,
        api_key=api_key,
        image_dir=image_dir,
        fetch=not args.no_solar,
        log=_log,
    )

    with_image = sum(int(r["has_image"] or 0) for r in rows)
    with_panels = sum(1 for r in rows if r.get("has_panels") == "1")
    known_panels = sum(1 for r in rows if r.get("has_panels") in ("0", "1"))
    print(f"\n{manifest_path}: {len(rows)} rows ({added} added, {filled} solar filled)")
    print(f"  with image  : {with_image}")
    print(f"  missing     : {len(rows) - with_image}")
    print(f"  addressed   : {sum(1 for r in rows if r['address'])}")
    print(f"  with panels : {with_panels} of {known_panels} with a solar verdict")
    print(f"  imagery     : {GOOGLE_ATTRIBUTION}")


if __name__ == "__main__":
    main()
