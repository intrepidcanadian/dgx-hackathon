#!/usr/bin/env python3
"""Bulk-ingest neighbourhood documents into txt2kg.

Run AFTER:
  1. txt2kg is running on Spark (./start.sh)
  2. 10_prepare_kg_documents.py has been run

Usage:
  python3 11_ingest_kg.py [--host HOST] [--model MODEL] [--limit N]
"""

import json
import time
import argparse
import requests
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "kg_documents"

TXT2KG_API = "http://localhost:3001/api"


def extract_triples(text, host, model):
    """Send text to txt2kg extract-triples API."""
    url = f"{host}/api/extract-triples"
    payload = {
        "text": text,
        "llmProvider": "ollama",
        "ollamaModel": model,
        "ollamaBaseUrl": "http://ollama:11434/v1",
    }
    resp = requests.post(url, json=payload, timeout=600)
    resp.raise_for_status()
    return resp.json()


def store_triples(triples, host):
    """Store extracted triples in the graph database."""
    url = f"{host}/api/graph-db/triples"
    payload = {"triples": triples}
    resp = requests.post(url, json=payload, timeout=60)
    resp.raise_for_status()
    return resp.json()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="http://localhost:3001",
                        help="txt2kg host URL")
    parser.add_argument("--model", default="llama3.1:8b",
                        help="Ollama model for triple extraction")
    parser.add_argument("--limit", type=int, default=0,
                        help="Max documents to process (0 = all)")
    parser.add_argument("--multi-source-only", action="store_true",
                        help="Only process documents with 2+ data sources")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be processed without calling API")
    args = parser.parse_args()

    bulk_file = DATA_DIR / "bulk_documents.json"
    if not bulk_file.exists():
        print(f"Error: {bulk_file} not found. Run 10_prepare_kg_documents.py first.")
        return

    with open(bulk_file) as f:
        documents = json.load(f)

    if args.multi_source_only:
        documents = [d for d in documents if
                     sum([d["metadata"]["has_dinesafe"],
                          d["metadata"]["has_311"],
                          d["metadata"]["has_health_hazard"]]) >= 2]
        print(f"Filtered to {len(documents)} multi-source documents")

    if args.limit > 0:
        documents = documents[:args.limit]

    print(f"Processing {len(documents)} documents")
    print(f"Host: {args.host}")
    print(f"Model: {args.model}")
    print()

    if args.dry_run:
        for i, doc in enumerate(documents):
            lat, lon = doc["metadata"]["lat"], doc["metadata"]["lon"]
            sources = []
            if doc["metadata"]["has_dinesafe"]:
                sources.append("DineSafe")
            if doc["metadata"]["has_311"]:
                sources.append("311")
            if doc["metadata"]["has_health_hazard"]:
                sources.append("HealthHazard")
            print(f"  [{i+1}] ({lat}, {lon}) — {', '.join(sources)} — {len(doc['text'])} chars")
        return

    total_triples = 0
    errors = 0

    for i, doc in enumerate(documents):
        lat, lon = doc["metadata"]["lat"], doc["metadata"]["lon"]
        print(f"[{i+1}/{len(documents)}] ({lat}, {lon})...", end=" ", flush=True)

        try:
            start = time.time()
            result = extract_triples(doc["text"], args.host, args.model)
            n_triples = result.get("count", 0)
            elapsed = time.time() - start

            if n_triples > 0 and result.get("triples"):
                store_triples(result["triples"], args.host)

            total_triples += n_triples
            print(f"{n_triples} triples ({elapsed:.1f}s)")

        except Exception as e:
            errors += 1
            print(f"ERROR: {e}")
            time.sleep(2)

    print(f"\nDone: {total_triples} triples from {len(documents)} documents ({errors} errors)")
    print(f"View the knowledge graph at {args.host}")


if __name__ == "__main__":
    main()
