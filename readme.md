# SolarFetch

SolarFetch is a solar panel installation rate analysis tool. Given a street
address and radius, it locates properties within the radius, pulls their 
building polygons from OpenStreetMap, matching satellite imagery from Google, 
and solar panel installation information from Google Solar API, and prepares 
per-house image crops for computer vision models that analyze roofs and solar panel coverage.

## Installation

Requires **Python 3.10+**.

```bash
pip install -r env/requirements.txt
```

Imagery needs a Google Maps API key with the **Maps Static API** and 
**Google Solar API** (https://developers.google.com/maps/documentation/solar) enabled:

```bash
export GOOGLE_MAPS_API_KEY="your-key-here"     # add to ~/.bashrc to persist
```

Geocoding and footprints work without a key; they use the free Nominatim and
Overpass services. Only the satellite imagery steps are billed.

## Quick start

Usage:
```bash
    python scripts/batch_fetch.py "1095 HAPPY VALLEY AVE, SAN JOSE, CA" --radius 200
    python scripts/batch_fetch.py "..." --radius 1200 --max-houses 400
    python scripts/batch_fetch.py "..." --radius 1200 --dry-run   # cost preview only
```

Collecting and filtering footprints needs no API key, 
so `--dry-run` and `--no-images --no-solar` work without one. Imagery needs
GOOGLE_MAPS_API_KEY with the Maps Static API enabled; solar needs the Solar API on the
same key, and spends nothing for a house already cached in `data/house/solar/`

## File structure:

