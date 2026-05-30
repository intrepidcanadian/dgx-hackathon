#!/bin/bash
# Download Toronto shelter occupancy data (2021-2025) from CKAN API

DATA_DIR="$(dirname "$0")/../data/raw"
mkdir -p "$DATA_DIR"

BASE="https://ckan0.cf.opendata.inter.prod-toronto.ca/datastore/dump"

download() {
  local name="$1" rid="$2"
  local out="$DATA_DIR/${name}.csv"
  if [ -f "$out" ]; then
    echo "Already exists: $out"
  else
    echo "Downloading $name..."
    curl -sL "${BASE}/${rid}?format=csv" -o "$out"
    echo "  -> $(wc -l < "$out") rows saved to $out"
  fi
}

download shelter_occupancy_2021 da8854b8-e570-4de2-b051-a906b62fe7f8
download shelter_occupancy_2022 1cc46acb-c6d3-4537-93ef-3ebad039275c
download shelter_occupancy_2023 62786156-b463-4c04-b286-c23f32c726ab
download shelter_occupancy_2024 fc409fd7-0348-49d7-bba9-70ac1a8c727c
download shelter_occupancy_2025 5dc4fbfc-0951-45e8-ae30-962af9dcaf7c

echo ""
echo "Done. Files in $DATA_DIR:"
ls -lh "$DATA_DIR"/*.csv
