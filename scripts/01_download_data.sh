#!/bin/bash
# Download Toronto shelter occupancy data (2021-2025) from CKAN API

DATA_DIR="$(dirname "$0")/../data/raw"
mkdir -p "$DATA_DIR"

BASE="https://ckan0.cf.opendata.inter.prod-toronto.ca/datastore/dump"

declare -A RESOURCES=(
  ["shelter_occupancy_2021"]="da8854b8-e570-4de2-b051-a906b62fe7f8"
  ["shelter_occupancy_2022"]="1cc46acb-c6d3-4537-93ef-3ebad039275c"
  ["shelter_occupancy_2023"]="62786156-b463-4c04-b286-c23f32c726ab"
  ["shelter_occupancy_2024"]="fc409fd7-0348-49d7-bba9-70ac1a8c727c"
  ["shelter_occupancy_2025"]="5dc4fbfc-0951-45e8-ae30-962af9dcaf7c"
)

for name in "${!RESOURCES[@]}"; do
  rid="${RESOURCES[$name]}"
  out="$DATA_DIR/${name}.csv"
  if [ -f "$out" ]; then
    echo "Already exists: $out"
  else
    echo "Downloading $name..."
    curl -sL "${BASE}/${rid}?format=csv" -o "$out"
    echo "  -> $(wc -l < "$out") rows saved to $out"
  fi
done

echo ""
echo "Done. Files in $DATA_DIR:"
ls -lh "$DATA_DIR"/*.csv
