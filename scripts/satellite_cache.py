"""Cached, tiled access to Google Static Maps satellite imagery.

A single Static Maps call is capped at 640x640 and is billed, so fetching a large
region -- or re-fetching one already seen -- has to be avoided. This module fetches
on a *fixed global grid* at a fixed zoom, so any region maps deterministically onto a
known set of tiles: a request checks the cache, downloads only the tiles it is missing,
then stitches and crops. A requested region may freely span tile boundaries.

Usage:
    export GOOGLE_MAPS_API_KEY=...
    python satellite_cache.py "1600 Amphitheatre Pkwy, Mountain View, CA" --radius 80
    python satellite_cache.py --bbox=-122.086,37.421,-122.083,37.424 --out data/region.png

Note the `=` in --bbox: a western longitude starts with '-', which argparse would
otherwise read as an option flag.
"""

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

from PIL import Image

from get_building_footprint import geocode_address
from get_satellite_image import (
    MAX_LATITUDE,
    WORLD_TILE_SIZE,
    download_satellite_image,
    latlon_to_world,
    world_to_latlon,
)

ZOOM = 20
SCALE = 1
MAPTYPE = "satellite"

TILE_PX = 600  # usable tile size after cropping
WATERMARK_PX = 20  # Google logo strip burned into the bottom of every response
REQUEST_W = TILE_PX  # 600
REQUEST_H = TILE_PX + WATERMARK_PX  # 620 -- both within Google's 640 cap at scale=1

MAX_TILES_DEFAULT = 256  # ~16x16 ~ 1.14 km^2; guards against an accidental huge bbox

# Zoom that polygon ids are quantised at. Pinned separately from ZOOM on purpose:
# changing the tile zoom must not silently rewrite every building id.
ID_ZOOM = 20

# Cropping removes Google's credit, which the Maps Platform terms require be preserved.
# Callers displaying imagery should surface this once per figure.
GOOGLE_ATTRIBUTION = "Imagery © Google"


def world_px_size(zoom: int = ZOOM) -> int:
    """Width/height of the whole world in pixels at `zoom`."""
    return WORLD_TILE_SIZE * 2**zoom


def max_tile_index(zoom: int = ZOOM) -> int:
    """Largest tile index that can actually be fetched in alignment.

    The world does not divide evenly into TILE_PX (at zoom 20,
    268435456 / 600 = 447392.43), so the final index `floor(world/TILE_PX)` is both
    partial *and* has a request center outside the Mercator square, which Google would
    silently clamp into a misaligned image. Stop one short of it."""
    return world_px_size(zoom) // TILE_PX - 1


def latlon_to_world_px(lat: float, lon: float, zoom: int = ZOOM) -> tuple[float, float]:
    """World pixel coordinates at `zoom`, clamped into the world square.

    The clamp absorbs float noise: at the exact edges the projection can return -0.0,
    which would make floor(y / TILE_PX) yield tile index -1."""
    x, y = latlon_to_world(lat, lon)
    limit = world_px_size(zoom)
    scale = 2**zoom
    return (
        min(max(x * scale, 0.0), limit),
        min(max(y * scale, 0.0), limit),
    )


def world_px_to_latlon(x: float, y: float, zoom: int = ZOOM) -> tuple[float, float]:
    """Inverse of `latlon_to_world_px`."""
    scale = 2**zoom
    return world_to_latlon(x / scale, y / scale)


def tile_request_center(i: int, j: int, zoom: int = ZOOM) -> tuple[float, float]:
    """(lat, lon) to request so the *cropped* tile covers exactly
    [i*TILE_PX, (i+1)*TILE_PX) x [j*TILE_PX, (j+1)*TILE_PX).

    Only the bottom of the response is cropped, so the retained area is not centered on
    the requested point -- it sits WATERMARK_PX/2 north of it. An image of height
    REQUEST_H centered at cy covers [cy - REQUEST_H/2, cy + REQUEST_H/2); cropping
    WATERMARK_PX off the bottom leaves [cy - REQUEST_H/2, cy + REQUEST_H/2 -
    WATERMARK_PX). Setting the top edge to j*TILE_PX gives cy below, and the bottom
    then lands exactly on (j+1)*TILE_PX since REQUEST_H - WATERMARK_PX == TILE_PX."""
    cx = i * TILE_PX + TILE_PX / 2
    cy = j * TILE_PX + REQUEST_H / 2
    return world_px_to_latlon(cx, cy, zoom)


def polygon_world_bounds(polygon, zoom: int = ID_ZOOM) -> tuple[int, int, int, int]:
    """Integer world-pixel bounding box of a shapely polygon, as (x0, y0, x1, y1).

    (x0, y0) is the north-west corner -- the same corner tiles are keyed on. Note the
    y axis inverts relative to latitude: the polygon's *max* latitude is its *minimum*
    world y.

    The two corners round in opposite directions so the integer box always *contains*
    the polygon. Flooring all four (as `int()` would) pulls the south-east corner
    inward and clips up to a pixel off the east and south edges of every footprint."""
    min_lon, min_lat, max_lon, max_lat = polygon.bounds
    x0, y0 = latlon_to_world_px(max_lat, min_lon, zoom)  # north-west
    x1, y1 = latlon_to_world_px(min_lat, max_lon, zoom)  # south-east
    return math.floor(x0), math.floor(y0), math.ceil(x1), math.ceil(y1)


def polygon_id(polygon, zoom: int = ID_ZOOM) -> str:
    """Deterministic id for a footprint: "b{x0}_{y0}_{w}x{h}" in world pixels.

    Derived from geometry rather than the OSM element id, which is not permanent --
    mappers routinely delete and redraw buildings (bulk re-imports especially), and the
    replacement gets a fresh id. The id is self-describing: `parse_polygon_id` recovers
    the bounding box from it with no lookup, so tile coverage and crop rectangles can be
    computed straight from the key.

    This is a content-addressed key, *not* a stable identity. It is quantised to
    ID_ZOOM pixels (~0.12 m), so any edit that moves a bounding-box extreme yields a
    different id -- a 30 cm shift changes it ~99% of the time. Matching a building
    across re-mapping needs spatial comparison (centroid distance or IoU), not this."""
    x0, y0, x1, y1 = polygon_world_bounds(polygon, zoom)
    return f"b{x0}_{y0}_{x1 - x0}x{y1 - y0}"


def parse_polygon_id(polygon_id_str: str) -> tuple[int, int, int, int]:
    """Inverse of `polygon_id`: recover (x0, y0, x1, y1) world pixels."""
    try:
        if not polygon_id_str.startswith("b"):
            raise ValueError("missing 'b' prefix")
        x0_s, y0_s, extent = polygon_id_str[1:].split("_")
        w_s, h_s = extent.split("x")
        x0, y0, w, h = int(x0_s), int(y0_s), int(w_s), int(h_s)
    except (ValueError, AttributeError) as exc:
        raise ValueError(
            f"{polygon_id_str!r} is not a polygon id of the form b<x0>_<y0>_<w>x<h>"
        ) from exc
    return x0, y0, x0 + w, y0 + h


def polygon_id_bbox(polygon_id_str: str, zoom: int = ID_ZOOM):
    """Bounding box of a polygon id as (min_lon, min_lat, max_lon, max_lat), ready to
    hand to `get_region`."""
    x0, y0, x1, y1 = parse_polygon_id(polygon_id_str)
    max_lat, min_lon = world_px_to_latlon(x0, y0, zoom)  # north-west
    min_lat, max_lon = world_px_to_latlon(x1, y1, zoom)  # south-east
    return min_lon, min_lat, max_lon, max_lat


class SatelliteTileCache:
    """Local tile cache over the Static Maps API, keyed on a fixed grid."""

    def __init__(
        self,
        cache_dir: str | Path = "data/tiles",
        api_key: str | None = None,
        zoom: int = ZOOM,
        maptype: str = MAPTYPE,
        request_delay: float = 0.1,
    ):
        self.cache_dir = Path(cache_dir)
        self.zoom = zoom
        self.maptype = maptype
        self.request_delay = request_delay
        self._api_key = api_key
        self.downloads = 0
        self.hits = 0

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.cache_dir / "index.json"
        self.index = self._load_index()

    # -- index ---------------------------------------------------------------

    def _index_metadata(self) -> dict:
        return {
            "zoom": self.zoom,
            "scale": SCALE,
            "tile_px": TILE_PX,
            "watermark_px": WATERMARK_PX,
            "maptype": self.maptype,
        }

    def _load_index(self) -> dict:
        if not self.index_path.exists():
            return {**self._index_metadata(), "tiles": {}}

        with open(self.index_path) as f:
            index = json.load(f)

        # Tiles built with different constants are not interchangeable -- mixing them in
        # one directory would misregister imagery, so fail loudly rather than silently.
        expected = self._index_metadata()
        mismatched = {
            k: (index.get(k), v) for k, v in expected.items() if index.get(k) != v
        }
        if mismatched:
            raise ValueError(
                f"Cache at {self.cache_dir} was built with different settings "
                f"(got vs expected: {mismatched}). Use a different cache_dir."
            )
        index.setdefault("tiles", {})
        return index

    def _save_index(self) -> None:
        with open(self.index_path, "w") as f:
            json.dump(self.index, f, indent=2)

    # -- tiles ---------------------------------------------------------------

    def tile_path(self, i: int, j: int) -> Path:
        return self.cache_dir / f"x{i}_y{j}.png"

    def _validate_indices(self, i: int, j: int) -> None:
        limit = max_tile_index(self.zoom)
        if not (0 <= i <= limit and 0 <= j <= limit):
            raise ValueError(
                f"Tile ({i}, {j}) out of range at zoom {self.zoom}; "
                f"valid indices are 0..{limit}"
            )

    def ensure_tile(self, i: int, j: int) -> Path:
        """Path to tile (i, j), downloading and cropping it if not already cached."""
        self._validate_indices(i, j)
        path = self.tile_path(i, j)
        if path.exists():
            self.hits += 1
            return path

        if not self._api_key:
            self._api_key = os.environ.get("GOOGLE_MAPS_API_KEY")
        if not self._api_key:
            raise EnvironmentError("GOOGLE_MAPS_API_KEY is not set")

        lat, lon = tile_request_center(i, j, self.zoom)
        raw = self._download_with_retry(lat, lon)

        image = Image.open(BytesIO(raw)).convert("RGB")
        if image.size != (REQUEST_W, REQUEST_H):
            raise RuntimeError(
                f"Expected a {REQUEST_W}x{REQUEST_H} response, got {image.size}"
            )
        tile = image.crop((0, 0, TILE_PX, TILE_PX))
        tile.save(path)

        self.index["tiles"][f"{i}_{j}"] = {
            "file": path.name,
            "downloaded": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "center_lat": lat,
            "center_lon": lon,
        }
        self._save_index()
        self.downloads += 1
        return path

    def _download_with_retry(self, lat: float, lon: float, attempts: int = 3) -> bytes:
        for attempt in range(attempts):
            try:
                return download_satellite_image(
                    lat,
                    lon,
                    self.zoom,
                    self._api_key,
                    size=f"{REQUEST_W}x{REQUEST_H}",
                    maptype=self.maptype,
                    scale=SCALE,
                )
            except Exception:
                if attempt == attempts - 1:
                    raise
                time.sleep(2**attempt)
            finally:
                time.sleep(self.request_delay)
        raise AssertionError("unreachable")

    # -- regions -------------------------------------------------------------

    def _bbox_to_world_rect(self, bbox) -> tuple[float, float, float, float]:
        """(min_lon, min_lat, max_lon, max_lat) -> (x0, y0, x1, y1) in world pixels.

        bbox ordering matches shapely's `.bounds`, so `polygons_bounding_box` feeds in
        directly. Latitude inverts: max_lat is the *minimum* world y."""
        min_lon, min_lat, max_lon, max_lat = bbox
        if min_lon > max_lon:
            raise ValueError(
                f"bbox crosses the antimeridian (min_lon {min_lon} > max_lon {max_lon}); "
                "the tile grid does not wrap"
            )
        if min_lat > max_lat:
            raise ValueError(f"bbox inverted in latitude: {min_lat} > {max_lat}")

        x0, y0 = latlon_to_world_px(max_lat, min_lon, self.zoom)  # north-west
        x1, y1 = latlon_to_world_px(min_lat, max_lon, self.zoom)  # south-east
        return x0, y0, x1, y1

    def tile_indices_for_bbox(self, bbox) -> tuple[int, int, int, int]:
        """Inclusive tile range (i0, j0, i1, j1) covering `bbox`."""
        x0, y0, x1, y1 = self._bbox_to_world_rect(bbox)
        i0 = math.floor(x0 / TILE_PX)
        j0 = math.floor(y0 / TILE_PX)
        i1 = math.ceil(x1 / TILE_PX) - 1
        j1 = math.ceil(y1 / TILE_PX) - 1
        # A degenerate (zero-width) bbox still needs the one tile containing it.
        i1 = max(i0, i1)
        j1 = max(j0, j1)
        self._validate_indices(i0, j0)
        self._validate_indices(i1, j1)
        return i0, j0, i1, j1

    def get_region(self, bbox, max_tiles: int = MAX_TILES_DEFAULT):
        """Stitch the imagery covering `bbox`, downloading only missing tiles.

        Returns (PIL.Image cropped exactly to bbox, (x0, y0) world-pixel origin). Pass
        the origin to `region_to_pixel` to place lat/lon features on the result."""
        x0, y0, x1, y1 = self._bbox_to_world_rect(bbox)
        i0, j0, i1, j1 = self.tile_indices_for_bbox(bbox)

        n_tiles = (i1 - i0 + 1) * (j1 - j0 + 1)
        if n_tiles > max_tiles:
            raise ValueError(
                f"Region needs {n_tiles} tiles, over the {max_tiles} limit. Each "
                "uncached tile is a billed API call -- narrow the bbox, or pass a "
                "larger max_tiles if this is intended."
            )

        width = max(1, round(x1 - x0))
        height = max(1, round(y1 - y0))
        canvas = Image.new("RGB", (width, height))

        for j in range(j0, j1 + 1):
            for i in range(i0, i1 + 1):
                tile = Image.open(self.ensure_tile(i, j))
                # paste clips tiles that overhang the canvas edges
                canvas.paste(tile, (round(i * TILE_PX - x0), round(j * TILE_PX - y0)))

        return canvas, (x0, y0)

    def get_region_for_point(
        self, lat: float, lon: float, radius_m: float, **kwargs
    ) -> tuple[Image.Image, tuple[float, float]]:
        """`get_region` over the square of half-width `radius_m` around a point."""
        return self.get_region(bbox_around_point(lat, lon, radius_m), **kwargs)

    def region_to_pixel(
        self, lat: float, lon: float, origin: tuple[float, float]
    ) -> tuple[float, float]:
        """Pixel coordinates of (lat, lon) within a mosaic returned by `get_region`."""
        x, y = latlon_to_world_px(lat, lon, self.zoom)
        return x - origin[0], y - origin[1]


def meters_per_pixel(lat: float, zoom: int = ZOOM) -> float:
    """Ground resolution at `lat`, in metres per pixel."""
    from get_satellite_image import EARTH_METERS_PER_PIXEL_AT_ZOOM_0

    return EARTH_METERS_PER_PIXEL_AT_ZOOM_0 * math.cos(math.radians(lat)) / 2**zoom


def bbox_around_point(lat: float, lon: float, radius_m: float):
    """Square bbox of half-width `radius_m` around a point, as
    (min_lon, min_lat, max_lon, max_lat)."""
    lat = max(-MAX_LATITUDE, min(MAX_LATITUDE, lat))
    dlat = radius_m / 111_320.0
    dlon = radius_m / (111_320.0 * max(math.cos(math.radians(lat)), 1e-6))
    return lon - dlon, lat - dlat, lon + dlon, lat + dlat


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("address", nargs="?", help="Street address to look up")
    parser.add_argument("--lat", type=float, help="Latitude (alternative to address)")
    parser.add_argument("--lon", type=float, help="Longitude (alternative to address)")
    parser.add_argument(
        "--radius", type=float, default=80, help="Half-width of the region in meters"
    )
    parser.add_argument(
        "--bbox",
        help="Explicit region as min_lon,min_lat,max_lon,max_lat. Use --bbox=... "
        "(with the equals sign) so a negative longitude isn't read as a flag",
    )
    parser.add_argument("--cache-dir", default="data/tiles")
    parser.add_argument("--max-tiles", type=int, default=MAX_TILES_DEFAULT)
    parser.add_argument("--out", default="data/region.png", help="Output image path")
    args = parser.parse_args()

    if not os.environ.get("GOOGLE_MAPS_API_KEY"):
        parser.error("GOOGLE_MAPS_API_KEY is not set")

    if args.bbox:
        try:
            bbox = tuple(float(v) for v in args.bbox.split(","))
        except ValueError:
            parser.error(f"could not parse --bbox={args.bbox!r} as 4 numbers")
        if len(bbox) != 4:
            parser.error("--bbox needs 4 comma-separated values")
    else:
        if args.lat is not None and args.lon is not None:
            lat, lon = args.lat, args.lon
        elif args.address:
            lat, lon = geocode_address(args.address)
        else:
            parser.error("Provide an address, --lat/--lon, or --bbox")
        bbox = bbox_around_point(lat, lon, args.radius)

    cache = SatelliteTileCache(cache_dir=args.cache_dir)
    i0, j0, i1, j1 = cache.tile_indices_for_bbox(bbox)
    print(
        f"Region spans tiles x{i0}..{i1} y{j0}..{j1} "
        f"({(i1 - i0 + 1) * (j1 - j0 + 1)} total)",
        file=sys.stderr,
    )

    image, _ = cache.get_region(bbox, max_tiles=args.max_tiles)
    image.save(args.out)
    print(
        f"Saved {image.size[0]}x{image.size[1]} mosaic to {args.out} "
        f"({cache.downloads} downloaded, {cache.hits} from cache). {GOOGLE_ATTRIBUTION}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
