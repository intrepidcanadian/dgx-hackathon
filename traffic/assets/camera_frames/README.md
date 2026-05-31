# Toronto Traffic Camera Frames — Hackathon Proof Snapshot

These are **real still frames** captured from the City of Toronto's public
traffic-camera feed, stored as evidence that the pipeline reads live municipal
camera imagery (the source the gemma3:4b VLM analyzes per sweep).

## Provenance
- **Source:** City of Toronto Open Data — Traffic Cameras
  (CKAN resource `824d2986-2fe0-4513-bdfb-e37e2499e7a9`). Each camera's live
  JPEG lives at the `IMAGEURL` in `traffic/data/raw/traffic_cameras.csv`, e.g.
  `https://opendata.toronto.ca/transportation/tmc/rescucameraimages/CameraImages/loc8001.jpg`.
- **Captured:** see `fetched_at_utc` in `manifest.json` (one snapshot moment).
- **Count:** 336 cameras, one frame each (`loc<REC_ID>.jpg`).

## How they're used in the live system
The dashboard and VLM orchestrator (`scripts/15_vlm_orchestrator.py`) normally
fetch each frame **live** from the `IMAGEURL` on every sweep — the images are
not persisted in production. This folder is a one-time captured snapshot kept
purely as hackathon proof of the real data source.

## Files
- `loc<REC_ID>.jpg` — one frame per camera, keyed by the camera's `REC_ID`.
- `manifest.json` — per-camera record: `rec_id`, `loc_key` (MAINROAD & CROSSROAD),
  `lat`/`lon`, `image_url`, `fetched_at_utc`, HTTP `status`, and byte size.
