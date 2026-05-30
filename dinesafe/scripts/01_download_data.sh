#!/bin/bash
# Download DineSafe inspection data + enrichment sources
set -e

RAW_DIR="$(dirname "$0")/../data/raw"
mkdir -p "$RAW_DIR"
cd "$RAW_DIR"

echo "=== 1. DineSafe Historical (2001-2022) ==="
if [ ! -f "dinesafe_historical.zip" ]; then
    curl -L -o dinesafe_historical.zip \
        "https://ckan0.cf.opendata.inter.prod-toronto.ca/dataset/dinesafe/resource/c0a5f6b0-534a-47c3-867d-d4b5cc84a656/download/Dinesafe.zip"
    unzip -o dinesafe_historical.zip -d dinesafe_hist/
fi
echo "Done"

echo ""
echo "=== 2. DineSafe Current (CSV ~23MB) ==="
curl -L -o dinesafe_current.csv \
    "https://ckan0.cf.opendata.inter.prod-toronto.ca/dataset/dinesafe/resource/af0f5b8a-4b73-4a50-8781-65e949792b40/download/Dinesafe.csv"
echo "Done"

echo ""
echo "=== 3. Fire Incidents ==="
curl -L -o fire_incidents.csv \
    "https://ckan0.cf.opendata.inter.prod-toronto.ca/dataset/fire-incidents/resource/fa5c7de5-10f8-41cf-883a-9b30a67c7b56/download/Fire%20Incidents%20Data.csv"
echo "Done"

echo ""
echo "=== 4. 311 Service Requests (2023-2026) ==="
for year in 2023 2024 2025 2026; do
    if [ ! -f "311_${year}.zip" ]; then
        echo "  Downloading $year..."
        case $year in
            2023) rid="079766f3-815d-4257-8731-5ff6b0c84c13" ;;
            2024) rid="f46b640d-d465-4f8b-9db5-5000a08295cd" ;;
            2025) rid="f3db05ab-2588-4159-89f7-56c74d1d8201" ;;
            2026) rid="99b7f283-7345-4f5a-a126-d078ed4f3419" ;;
        esac
        curl -L -o "311_${year}.zip" \
            "https://ckan0.cf.opendata.inter.prod-toronto.ca/dataset/311-service-requests-customer-initiated/resource/${rid}/download/311-service-requests-${year}.zip"
    fi
done
echo "Done"

echo ""
echo "=== 5. Weather (Toronto Pearson, hourly) ==="
# Reuse from housing project if available
if [ -f "../../housing/data/raw/weather_hourly_toronto.csv" ]; then
    echo "Linking from housing project..."
    ln -sf "$(realpath ../../housing/data/raw/weather_hourly_toronto.csv)" weather_hourly.csv
else
    echo "Will be downloaded by prepare script"
fi
echo "Done"

echo ""
echo "All downloads complete!"
ls -lh "$RAW_DIR"
