"""Address/lat-lon -> Google Static Maps satellite image.

Requires a GOOGLE_MAPS_API_KEY with the Maps Static API enabled.

Usage:
    export GOOGLE_MAPS_API_KEY=...
    python get_satellite_image.py "1600 Amphitheatre Pkwy, Mountain View, CA"
    python get_satellite_image.py "..." --radius 60 --size 640x640 --out data/satellite.png
    python get_satellite_image.py --lat 37.422 --lon -122.084 --radius 60
"""

import argparse
import math
import os
import sys

import requests

from get_building_footprint import geocode_address

STATIC_MAP_URL = "https://maps.googleapis.com/maps/api/staticmap"

# Meters per pixel at the equator at zoom 0, for the Web Mercator projection
# Google Static Maps uses (256px tiles covering the full ~40,075km circumference).
EARTH_METERS_PER_PIXEL_AT_ZOOM_0 = 156543.03392

# The whole earth occupies a WORLD_TILE_SIZE square at zoom 0. Origin (0, 0) is the
# top-left -- longitude -180, latitude +MAX_LATITUDE -- with y increasing southward.
WORLD_TILE_SIZE = 256

# Mercator's y diverges at the poles, so the projection only covers latitudes up to the
# one where y reaches the edge of the square. Defining it as the inverse projection
# evaluated at y=0 keeps latlon_to_world and world_to_latlon consistent by construction.
MAX_LATITUDE = math.degrees(math.atan(math.sinh(math.pi)))  # 85.05112877980659


def zoom_for_radius(lat: float, radius_m: float, image_px: int, max_zoom: int = 21) -> int:
    """Largest zoom level whose image still covers a `2*radius_m` wide area."""
    meters_per_pixel_needed = (2 * radius_m) / image_px
    zoom = math.log2(
        EARTH_METERS_PER_PIXEL_AT_ZOOM_0 * math.cos(math.radians(lat)) / meters_per_pixel_needed
    )
    return max(0, min(max_zoom, math.floor(zoom)))


def latlon_to_world(lat: float, lon: float) -> tuple[float, float]:
    """Project (lat, lon) to Google world coordinates in [0, WORLD_TILE_SIZE]^2.

    Longitude is cyclic, so it wraps into [-180, 180) -- note this makes +180
    identical to -180 (x = 0), with x = WORLD_TILE_SIZE a limit approached but
    never attained. Latitude is clamped to +/-MAX_LATITUDE, which also keeps
    1 - sin(lat) >= 0.0037 so the log below can never blow up."""
    lon = ((lon + 180.0) % 360.0) - 180.0
    lat = max(-MAX_LATITUDE, min(MAX_LATITUDE, lat))

    siny = math.sin(math.radians(lat))
    x = WORLD_TILE_SIZE * (0.5 + lon / 360.0)
    y = WORLD_TILE_SIZE * (0.5 - math.log((1 + siny) / (1 - siny)) / (4 * math.pi))
    return x, y


def world_to_latlon(x: float, y: float) -> tuple[float, float]:
    """Inverse of `latlon_to_world`. Total for any finite input, so it needs no
    clamping: y = 0 gives +MAX_LATITUDE and y = WORLD_TILE_SIZE gives -MAX_LATITUDE."""
    lon = (x / WORLD_TILE_SIZE - 0.5) * 360.0
    lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / WORLD_TILE_SIZE))))
    return lat, lon


def latlon_to_pixel(
    lat: float,
    lon: float,
    center_lat: float,
    center_lon: float,
    zoom: int,
    image_size: tuple[int, int],
    scale: int = 1,
) -> tuple[float, float]:
    """Project (lat, lon) to pixel coordinates in a Static Maps image
    of `image_size` (the requested, unscaled WIDTHxHEIGHT) centered on
    (center_lat, center_lon) at the given zoom, using the same Web
    Mercator tiling Google Static Maps renders with. `scale` should match
    the `scale` the image was requested with (the returned bitmap is
    `image_size * scale` pixels)."""
    px, py = latlon_to_world(lat, lon)
    cx, cy = latlon_to_world(center_lat, center_lon)
    pixels_per_tile = 2**zoom

    width, height = image_size
    img_x = (px - cx) * pixels_per_tile + width / 2
    img_y = (py - cy) * pixels_per_tile + height / 2
    return img_x * scale, img_y * scale


def download_satellite_image(
    lat: float,
    lon: float,
    zoom: int,
    api_key: str,
    size: str = "640x640",
    maptype: str = "satellite",
    scale: int = 2,
) -> bytes:
    resp = requests.get(
        STATIC_MAP_URL,
        params={
            "center": f"{lat},{lon}",
            "zoom": zoom,
            "size": size,
            "maptype": maptype,
            "scale": scale,
            "key": api_key,
        },
        timeout=15,
    )
    resp.raise_for_status()

    # Google can answer 200 with a text/plain error body; caching that as a tile would
    # poison it persistently, so reject anything that isn't actually an image.
    content_type = resp.headers.get("Content-Type", "")
    if not content_type.startswith("image/"):
        raise RuntimeError(
            f"Static Maps returned {content_type!r}, not an image: {resp.text[:200]!r}"
        )
    return resp.content


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("address", nargs="?", help="Street address to look up")
    parser.add_argument("--lat", type=float, help="Latitude (alternative to address)")
    parser.add_argument("--lon", type=float, help="Longitude (alternative to address)")
    parser.add_argument(
        "--radius", type=float, default=60, help="Coverage radius in meters"
    )
    parser.add_argument(
        "--size", default="640x640", help="Image size in pixels, WIDTHxHEIGHT"
    )
    parser.add_argument("--zoom", type=int, help="Override the auto-computed zoom level")
    parser.add_argument(
        "--maptype", default="satellite", choices=["satellite", "hybrid"]
    )
    parser.add_argument("--out", default="data/satellite.png", help="Output image path")
    args = parser.parse_args()

    if args.lat is not None and args.lon is not None:
        lat, lon = args.lat, args.lon
    elif args.address:
        lat, lon = geocode_address(args.address)
    else:
        parser.error("Provide an address or both --lat and --lon")

    api_key = os.environ.get("GOOGLE_MAPS_API_KEY")
    if not api_key:
        raise EnvironmentError("GOOGLE_MAPS_API_KEY is not set")

    width_px = int(args.size.lower().split("x")[0])
    zoom = (
        args.zoom
        if args.zoom is not None
        else zoom_for_radius(lat, args.radius, width_px)
    )

    image_bytes = download_satellite_image(
        lat, lon, zoom, api_key, size=args.size, maptype=args.maptype
    )

    with open(args.out, "wb") as f:
        f.write(image_bytes)
    print(f"Saved satellite image (zoom={zoom}) to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
