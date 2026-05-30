#!/bin/bash
# Download Toronto shelter occupancy data (2021-2026) and weather data

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
download shelter_occupancy_2026 42714176-4f05-44e6-b157-2b57f29b856a

# Download Environment Canada daily weather for Toronto Pearson (station 51459)
# Covers 2021-2026 to match shelter data
WEATHER_BASE="https://climate.weather.gc.ca/climate_data/bulk_data_e.html"
WEATHER_OUT="$DATA_DIR/weather_toronto.csv"

if [ -f "$WEATHER_OUT" ]; then
  echo "Already exists: $WEATHER_OUT"
else
  echo "Downloading Toronto weather data (2021-2026)..."
  first=1
  for year in 2021 2022 2023 2024 2025 2026; do
    echo "  Fetching $year..."
    tmp="$DATA_DIR/weather_${year}.tmp"
    curl -sL "${WEATHER_BASE}?format=csv&stationID=51459&Year=${year}&Month=1&Day=1&timeframe=2" -o "$tmp"
    if [ $first -eq 1 ]; then
      cat "$tmp" > "$WEATHER_OUT"
      first=0
    else
      tail -n +2 "$tmp" >> "$WEATHER_OUT"
    fi
    rm -f "$tmp"
  done
  echo "  -> $(wc -l < "$WEATHER_OUT") rows saved to $WEATHER_OUT"
fi

echo ""
echo "Done. Files in $DATA_DIR:"
ls -lh "$DATA_DIR"/*.csv
