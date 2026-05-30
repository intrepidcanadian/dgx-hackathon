#!/usr/bin/env python3
"""Actionable Toronto traffic intelligence for Hermes Agent.

Not just "what's happening" but "what should you do about it."

Outputs:
  1. ROUTE ADVICE — avoid X, take Y instead
  2. TIMING — "congestion clearing in ~20 min" based on historical patterns
  3. TRANSIT — "Line 1 running normally, consider TTC" or "Line 2 delayed, stay on road"
  4. PREDICTION — "DVP will hit gridlock by 5:30 PM based on current buildup"
  5. CONTEXT — WHY is it congested (construction, incident, rush hour, event)

Pulls live data:
  - 336 traffic cameras → VLM congestion classification (GPU)
  - TTC subway delays → transit alternative viability
  - Utility cuts → explains construction-related congestion
  - Road restrictions → active closures
  - Historical model → is this normal for this time?

Usage:
  python3 09_hermes_actionable_monitor.py --cameras 25 --model gemma3:4b
  python3 09_hermes_actionable_monitor.py --commute "downtown to north york"

Hermes scheduling:
  "Every weekday at 7:30 AM and 4:30 PM, run the traffic monitor
   and tell me how to get from downtown to North York"
"""

import argparse
import json
import time
import base64
import re
import sys
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

DATA_DIR = Path(__file__).parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
STATE_DIR = DATA_DIR / "monitor_state"
ENRICH_DIR = DATA_DIR / "enrichment"
MODEL_DIR = Path(__file__).parent.parent / "models"
STATE_DIR.mkdir(parents=True, exist_ok=True)

CKAN_API = "https://ckan0.cf.opendata.inter.prod-toronto.ca/api/3/action/datastore_search"

# Toronto corridor graph — major alternative routes
CORRIDORS = {
    "GARDINER": {
        "direction": "E-W", "type": "Expressway",
        "alternatives": ["LAKE SHORE BLVD", "KING ST", "QUEEN ST", "FRONT ST"],
        "connects": "Downtown ↔ West End / Airport",
    },
    "DVP": {
        "direction": "N-S", "type": "Expressway",
        "alternatives": ["BAYVIEW AVE", "DON MILLS RD", "LESLIE ST"],
        "connects": "Downtown ↔ North York / 401",
    },
    "DON VALLEY": {
        "direction": "N-S", "type": "Expressway",
        "alternatives": ["BAYVIEW AVE", "DON MILLS RD"],
        "connects": "Downtown ↔ North York",
    },
    "YONGE ST": {
        "direction": "N-S", "type": "Arterial",
        "alternatives": ["BAYVIEW AVE", "AVENUE RD", "BATHURST ST"],
        "connects": "Downtown ↔ Midtown ↔ North York",
        "transit": "Line 1 Yonge-University",
    },
    "BLOOR ST": {
        "direction": "E-W", "type": "Arterial",
        "alternatives": ["DUPONT ST", "DAVENPORT RD", "DANFORTH AVE"],
        "connects": "West End ↔ East End",
        "transit": "Line 2 Bloor-Danforth",
    },
    "EGLINTON AVE": {
        "direction": "E-W", "type": "Arterial",
        "alternatives": ["ST CLAIR AVE", "LAWRENCE AVE"],
        "connects": "West ↔ East through Midtown",
        "transit": "Line 5 Eglinton (LRT)",
    },
    "LAKE SHORE": {
        "direction": "E-W", "type": "Arterial",
        "alternatives": ["GARDINER XY", "QUEEN ST", "KING ST"],
        "connects": "Waterfront corridor",
    },
    "SPADINA AVE": {
        "direction": "N-S", "type": "Arterial",
        "alternatives": ["BATHURST ST", "UNIVERSITY AVE"],
        "connects": "Harbourfront ↔ Bloor ↔ Uptown",
        "transit": "509/510 Streetcar, Line 1/2",
    },
    "KING ST": {
        "direction": "E-W", "type": "Arterial",
        "alternatives": ["QUEEN ST", "FRONT ST", "ADELAIDE ST"],
        "connects": "Liberty Village ↔ Financial District ↔ Distillery",
        "transit": "504 King Streetcar",
    },
    "QUEEN ST": {
        "direction": "E-W", "type": "Arterial",
        "alternatives": ["KING ST", "DUNDAS ST"],
        "connects": "Parkdale ↔ Downtown ↔ Beaches",
        "transit": "501 Queen Streetcar",
    },
    "FINCH AVE": {
        "direction": "E-W", "type": "Arterial",
        "alternatives": ["STEELES AVE", "SHEPPARD AVE"],
        "connects": "North boundary corridor",
    },
}

# TTC subway lines and which roads they parallel
SUBWAY_LINES = {
    "YU": {"name": "Line 1 Yonge-University", "roads": ["YONGE", "UNIVERSITY", "SPADINA"]},
    "BD": {"name": "Line 2 Bloor-Danforth", "roads": ["BLOOR", "DANFORTH"]},
    "SRT": {"name": "Line 3 Scarborough", "roads": ["MCCOWAN", "KENNEDY"]},
    "SM": {"name": "Line 4 Sheppard", "roads": ["SHEPPARD"]},
}

PRIORITY_ROADS = [
    "GARDINER", "DVP", "DON VALLEY", "ALLEN", "F G GARDINER",
    "401", "427", "404", "LAKE SHORE",
    "YONGE", "BLOOR", "DUNDAS", "QUEEN", "KING", "SPADINA",
    "UNIVERSITY", "BAY", "EGLINTON", "FINCH",
]

FAST_PROMPT = """Analyze this Toronto traffic camera. Return ONLY valid JSON:
{"level":<0-3>,"flow":"<Free|Steady|Slow|StopGo|Gridlock>","vehicles":<count>,"queue":"<None|Short|Medium|Long>","issue":"<none|construction|incident|transit_delay>"}
0=empty/free, 1=normal, 2=heavy, 3=gridlock"""


def fetch_image(url, timeout=8):
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        return base64.b64encode(r.content).decode("utf-8") if len(r.content) > 500 else None
    except Exception:
        return None


def query_vlm(img, ollama_url, model):
    r = requests.post(f"{ollama_url}/api/generate", json={
        "model": model, "prompt": FAST_PROMPT, "images": [img],
        "stream": False, "options": {"temperature": 0.0, "num_predict": 150},
    }, timeout=120)
    r.raise_for_status()
    m = re.search(r'\{[^}]+\}', r.json().get("response", ""))
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    return {"level": -1, "flow": "Unknown", "vehicles": -1}


def get_ttc_delays_today():
    """Check for TTC subway delays in the last 2 hours."""
    delays = {}
    try:
        r = requests.get(CKAN_API, params={
            "id": "6088e14f-e46e-4f5c-9daa-dea1359ad396",
            "limit": 50,
            "sort": "_id desc",
        }, timeout=10)
        records = r.json()["result"]["records"]
        today = datetime.now().strftime("%Y-%m-%d")
        for rec in records:
            date_str = str(rec.get("Date", rec.get("date", "")))
            if today in date_str or (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d") in date_str:
                line = str(rec.get("Line", rec.get("line", "Unknown")))
                delay_min = float(rec.get("Min Delay", rec.get("min_delay", 0)) or 0)
                if line not in delays:
                    delays[line] = {"total_delay_min": 0, "incidents": 0}
                delays[line]["total_delay_min"] += delay_min
                delays[line]["incidents"] += 1
    except Exception:
        pass
    return delays


def get_active_disruptions():
    """Check for active utility cuts and road restrictions."""
    disruptions = []
    try:
        r = requests.get(CKAN_API, params={
            "id": "ebdec599-4522-4473-b276-fa07d8638248",
            "limit": 100,
            "filters": json.dumps({"PERMIT_STATUS": "PERMIT ISSUED"}),
        }, timeout=10)
        records = r.json()["result"]["records"]
        today = datetime.now()
        for rec in records:
            from_date = pd.to_datetime(rec.get("PROPOSED_FROM_DATE"), errors="coerce")
            to_date = pd.to_datetime(rec.get("PROPOSED_TO_DATE"), errors="coerce")
            if from_date and to_date and from_date <= today <= to_date:
                desc = rec.get("DISPLAY_DESC", "Unknown location")
                disruptions.append({"type": "utility_cut", "location": desc})
    except Exception:
        pass
    return disruptions[:20]


def fetch_todays_events():
    """Pull real events happening today from Toronto Open Data.

    Source: Liquor Licence Special Events (municipally significant).
    Returns list of events with location, category, and affected roads.
    """
    today = datetime.now().date()
    events = []
    try:
        records = []
        offset = 0
        while True:
            r = requests.get(CKAN_API, params={
                "id": "e9f77756-2baf-46ba-b2c6-4050e2fba755",
                "limit": 5000, "offset": offset,
            }, timeout=30)
            data = r.json()["result"]
            records.extend(data["records"])
            if len(data["records"]) < 5000:
                break
            offset += 5000

        for rec in records:
            start = pd.to_datetime(rec.get("STARTING_DATE"), errors="coerce")
            end = pd.to_datetime(rec.get("ENDING_DATE"), errors="coerce")
            if pd.isna(start):
                continue
            end = end if pd.notna(end) else start
            if not (start.date() <= today <= end.date()):
                continue

            lat, lon = None, None
            geo = rec.get("geometry")
            if isinstance(geo, str):
                try:
                    import json as _json
                    g = _json.loads(geo)
                    if g.get("type") == "Point":
                        lon, lat = g["coordinates"]
                except Exception:
                    pass
            elif isinstance(geo, dict) and geo.get("type") == "Point":
                lon, lat = geo["coordinates"]

            name = str(rec.get("ESTABLISHMENT", "")).lower()
            large_kw = ["festival", "pride", "caribana", "marathon", "parade",
                        "fair", "exhibition", "cne", "taste of", "nuit blanche"]
            if any(kw in name for kw in large_kw):
                category = "large"
            elif any(kw in name for kw in ["market", "concert", "gala"]):
                category = "medium"
            else:
                category = "small"

            # Match to nearby corridors
            affected_corridor = None
            address = str(rec.get("ADDRESS", "")).upper()
            for ck in CORRIDORS:
                if ck in address:
                    affected_corridor = ck
                    break

            events.append({
                "name": rec.get("ESTABLISHMENT", "Unknown"),
                "address": rec.get("ADDRESS", ""),
                "ward": rec.get("WARD_NAME", ""),
                "lat": lat, "lon": lon,
                "category": category,
                "affected_corridor": affected_corridor,
            })
    except Exception as e:
        print(f"  Event fetch warning: {e}", file=sys.stderr)

    return events


def fetch_live_road_restrictions():
    """Pull active road restrictions from Toronto live feed."""
    restrictions = []
    try:
        import io
        url = "https://secure.toronto.ca/opendata/cart/road_restrictions/v3?format=csv"
        r = requests.get(url, timeout=20)
        lines = r.text.strip().split("\n")
        csv_text = "\n".join(lines[1:])
        df = pd.read_csv(io.StringIO(csv_text))

        # Major roads only
        major = df[df["RoadClass"].isin([
            "Major Arterial Road", "Expressway", "Expressway Ramp",
        ])]

        for _, row in major.iterrows():
            road = str(row.get("Road", "")).upper()
            affected_corridor = None
            for ck in CORRIDORS:
                if ck in road:
                    affected_corridor = ck
                    break

            restrictions.append({
                "road": row.get("Road", ""),
                "name": str(row.get("Name", ""))[:60],
                "type": row.get("WorkEventType", ""),
                "road_class": row.get("RoadClass", ""),
                "affected_corridor": affected_corridor,
            })
    except Exception as e:
        print(f"  Restriction fetch warning: {e}", file=sys.stderr)

    return restrictions


def get_historical_baseline():
    """Load model's expected congestion for current time."""
    try:
        meta_file = MODEL_DIR / "model_metadata.json"
        if meta_file.exists():
            with open(meta_file) as f:
                return json.load(f)
    except Exception:
        pass
    return None


def load_previous_state():
    state_file = STATE_DIR / "last_state.json"
    if state_file.exists():
        try:
            with open(state_file) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_state(results):
    state = {r.get("location", "?"): {
        "level": r.get("level", -1), "flow": r.get("flow", ""),
        "timestamp": r.get("timestamp", ""),
    } for r in results}
    with open(STATE_DIR / "last_state.json", "w") as f:
        json.dump(state, f)
    return state


def match_corridor(road_name):
    """Match a camera's road to a known corridor."""
    road_upper = str(road_name).upper()
    for corridor_key in CORRIDORS:
        if corridor_key in road_upper:
            return corridor_key
    return None


def generate_route_advice(corridor_congestion, ttc_delays):
    """Generate specific route recommendations."""
    advice = []

    for corridor, data in sorted(corridor_congestion.items(),
                                  key=lambda x: x[1]["avg_level"], reverse=True):
        if data["avg_level"] < 1.5:
            continue

        info = CORRIDORS.get(corridor, {})
        alternatives = info.get("alternatives", [])
        transit_line = info.get("transit", None)
        direction = info.get("direction", "")
        connects = info.get("connects", "")

        severity = "🔴 AVOID" if data["avg_level"] >= 2.5 else "🟠 SLOW"

        rec = f"{severity} **{corridor}** ({connects})\n"

        # Suggest alternatives
        if alternatives:
            rec += f"  → Try: {', '.join(alternatives[:2])}\n"

        # Check if transit is viable
        if transit_line:
            line_key = None
            for lk, lv in SUBWAY_LINES.items():
                if any(r in corridor for r in lv["roads"]):
                    line_key = lk
                    break

            if line_key and line_key in ttc_delays:
                delay = ttc_delays[line_key]
                if delay["total_delay_min"] > 15:
                    rec += f"  ⚠️ {transit_line} also delayed ({delay['total_delay_min']:.0f} min today)\n"
                else:
                    rec += f"  🚇 {transit_line} running OK — consider transit\n"
            elif transit_line:
                rec += f"  🚇 {transit_line} — consider transit\n"

        # Issue context
        issues = [r for r in data.get("results", []) if r.get("issue", "none") != "none"]
        if issues:
            issue_types = set(r["issue"] for r in issues)
            rec += f"  📋 Cause: {', '.join(issue_types)}\n"

        advice.append(rec)

    return advice


def generate_timing_advice(corridor_congestion):
    """Predict when congestion will ease based on time of day patterns."""
    now = datetime.now()
    hour = now.hour

    timing = []
    if 7 <= hour <= 9:
        timing.append("🕐 Morning rush typically clears by 9:30-10:00 AM")
        if hour >= 9:
            timing.append("🕐 Rush winding down — conditions improving in ~30 min")
    elif 16 <= hour <= 18:
        timing.append("🕐 Evening rush typically clears by 6:30-7:00 PM")
        if hour >= 18:
            timing.append("🕐 Rush winding down — conditions improving in ~30 min")
    elif 11 <= hour <= 14:
        timing.append("🕐 Midday — current conditions likely stable for 1-2 hours")
    elif hour >= 20 or hour < 6:
        timing.append("🕐 Off-peak — roads should stay clear")
    elif 9 <= hour <= 11:
        timing.append("🕐 Post-morning rush — if still congested, may be incident/construction")
    elif 14 <= hour <= 16:
        timing.append("🕐 Pre-rush — expect congestion building from 4 PM onward")

    return timing


def format_actionable_report(results, corridor_congestion, ttc_delays,
                              disruptions, prev_state, timestamp, elapsed,
                              live_events=None, live_restrictions=None):
    """The 'so what' report — what to do, not just what's happening."""
    lines = []
    valid = [r for r in results if r.get("level", -1) >= 0]
    avg = np.mean([r["level"] for r in valid]) if valid else 0

    # === HEADLINE ===
    if avg < 0.5:
        lines.append("🟢 ROADS CLEAR — Go anywhere")
    elif avg < 1.2:
        lines.append("🟡 NORMAL TRAFFIC — Standard routes fine")
    elif avg < 2.0:
        lines.append("🟠 HEAVY TRAFFIC — Check routes below")
    else:
        lines.append("🔴 MAJOR CONGESTION — Reroute needed")
    lines.append(f"📊 {len(valid)} cameras | {timestamp}")
    lines.append("")

    # === ROUTE ADVICE (the "so what") ===
    route_advice = generate_route_advice(corridor_congestion, ttc_delays)
    if route_advice:
        lines.append("🛣 ROUTE ADVICE:")
        lines.extend(route_advice)
    else:
        lines.append("✅ All major corridors flowing normally")
        lines.append("")

    # === TIMING ===
    timing = generate_timing_advice(corridor_congestion)
    if timing:
        for t in timing:
            lines.append(t)
        lines.append("")

    # === TRANSIT STATUS ===
    if ttc_delays:
        lines.append("🚇 TTC STATUS:")
        for line_key, delay in ttc_delays.items():
            line_name = SUBWAY_LINES.get(line_key, {}).get("name", line_key)
            if delay["total_delay_min"] > 15:
                lines.append(f"  ⚠️ {line_name}: {delay['incidents']} delays, {delay['total_delay_min']:.0f} min total")
            elif delay["incidents"] > 0:
                lines.append(f"  🟡 {line_name}: minor delays ({delay['total_delay_min']:.0f} min)")
            else:
                lines.append(f"  ✅ {line_name}: normal service")
    else:
        lines.append("🚇 TTC: No delay data — assume normal service")
    lines.append("")

    # === ACTIVE DISRUPTIONS ===
    if disruptions:
        lines.append(f"🚧 {len(disruptions)} active road works:")
        for d in disruptions[:5]:
            lines.append(f"  • {d['location'][:50]}")
        lines.append("")

    # === LIVE EVENTS ===
    if live_events:
        large = [e for e in live_events if e["category"] == "large"]
        medium = [e for e in live_events if e["category"] == "medium"]
        small = [e for e in live_events if e["category"] == "small"]

        if large or medium:
            lines.append(f"🎪 EVENTS TODAY ({len(live_events)} total):")
            for ev in large[:3]:
                corr_note = f" — affects {ev['affected_corridor']}" if ev.get("affected_corridor") else ""
                lines.append(f"  🎪 {ev['name']}{corr_note}")
                if ev.get("address"):
                    lines.append(f"     📍 {ev['address']}")
            for ev in medium[:3]:
                lines.append(f"  🎵 {ev['name']}")
                if ev.get("address"):
                    lines.append(f"     📍 {ev['address']}")
            if small:
                lines.append(f"  ℹ️ + {len(small)} smaller events across the city")
            lines.append("")
        elif small:
            lines.append(f"ℹ️ {len(small)} small events today (minimal traffic impact)")
            lines.append("")

    # === ROAD RESTRICTIONS ON CORRIDORS ===
    if live_restrictions:
        corridor_rest = {}
        for r in live_restrictions:
            ck = r.get("affected_corridor")
            if ck:
                corridor_rest.setdefault(ck, []).append(r)
        if corridor_rest:
            lines.append(f"🚧 CORRIDOR RESTRICTIONS ({len(live_restrictions)} total on major roads):")
            for ck, rests in sorted(corridor_rest.items(),
                                     key=lambda x: len(x[1]), reverse=True)[:5]:
                info = CORRIDORS.get(ck, {})
                lines.append(f"  {ck}: {len(rests)} restriction(s)")
                if info.get("alternatives"):
                    lines.append(f"    → Alt: {', '.join(info['alternatives'][:2])}")
            lines.append("")

    # === CHANGES SINCE LAST CHECK ===
    if prev_state:
        worsened = []
        improved = []
        for r in valid:
            loc = r.get("location", "?")
            if loc in prev_state:
                prev = prev_state[loc].get("level", -1)
                curr = r.get("level", -1)
                if curr > prev and curr >= 2:
                    worsened.append(f"  🔺 {loc}")
                elif curr < prev and prev >= 2:
                    improved.append(f"  🔻 {loc}")

        if worsened:
            lines.append("⚠️ GETTING WORSE:")
            lines.extend(worsened[:3])
        if improved:
            lines.append("✅ IMPROVING:")
            lines.extend(improved[:3])
        if worsened or improved:
            lines.append("")

    # === QUICK STATS ===
    for lvl, emoji, label in [(3, "🔴", "Gridlock"), (2, "🟠", "Heavy"),
                               (1, "🟡", "Normal"), (0, "🟢", "Clear")]:
        cnt = sum(1 for r in valid if r["level"] == lvl)
        if cnt > 0:
            lines.append(f"  {emoji} {label}: {cnt}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cameras", type=int, default=25)
    parser.add_argument("--priority", action="store_true")
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--model", default="gemma3:4b")
    parser.add_argument("--commute", type=str, default="",
                        help='Route context, e.g. "downtown to north york"')
    args = parser.parse_args()

    cam_file = RAW_DIR / "traffic_cameras.csv"
    if not cam_file.exists():
        print("Error: Run 01_prepare_traffic_data.py first")
        sys.exit(1)

    cams = pd.read_csv(cam_file)
    url_col = [c for c in cams.columns if "image" in c.lower() or "url" in c.lower()][0]

    # Select priority cameras
    mask = cams["MAINROAD"].str.upper().apply(
        lambda x: any(p in str(x) for p in PRIORITY_ROADS))
    selected = cams[mask].head(args.cameras) if args.cameras > 0 else cams[mask]
    if len(selected) < 10:
        selected = cams.head(args.cameras)

    timestamp = datetime.now().strftime("%H:%M %b %d")
    prev_state = load_previous_state()

    # Parallel data pulls — VLM + context
    print(f"Analyzing {len(selected)} cameras...", file=sys.stderr)
    t_start = time.time()

    # Pull context data (fast, no GPU)
    ttc_delays = get_ttc_delays_today()
    disruptions = get_active_disruptions()

    print("Fetching live events and road restrictions...", file=sys.stderr)
    live_events = fetch_todays_events()
    live_restrictions = fetch_live_road_restrictions()
    print(f"  {len(live_events)} events, {len(live_restrictions)} restrictions",
          file=sys.stderr)

    # VLM sweep (GPU)
    results = []
    for _, cam in selected.iterrows():
        main_road = str(cam.get("MAINROAD", ""))
        cross_road = str(cam.get("CROSSROAD", ""))
        location = f"{main_road} & {cross_road}"

        img = fetch_image(cam[url_col])
        if not img:
            continue

        try:
            parsed = query_vlm(img, args.ollama_url, args.model)
            parsed["location"] = location
            parsed["main_road"] = main_road
            parsed["cross_road"] = cross_road
            parsed["timestamp"] = timestamp
            results.append(parsed)
        except Exception:
            pass

    elapsed = time.time() - t_start

    if not results:
        print("No cameras analyzed")
        sys.exit(1)

    # Aggregate by corridor
    corridor_congestion = {}
    for r in results:
        corridor = match_corridor(r.get("main_road", ""))
        if corridor:
            if corridor not in corridor_congestion:
                corridor_congestion[corridor] = {"levels": [], "results": []}
            if r.get("level", -1) >= 0:
                corridor_congestion[corridor]["levels"].append(r["level"])
                corridor_congestion[corridor]["results"].append(r)

    for corridor in corridor_congestion:
        levels = corridor_congestion[corridor]["levels"]
        corridor_congestion[corridor]["avg_level"] = np.mean(levels) if levels else 0

    # Save state
    save_state(results)

    # Generate actionable report
    report = format_actionable_report(
        results, corridor_congestion, ttc_delays,
        disruptions, prev_state, timestamp, elapsed,
        live_events=live_events,
        live_restrictions=live_restrictions,
    )

    # Add commute-specific advice if requested
    if args.commute:
        commute_upper = args.commute.upper()
        report += f"\n\n🧭 YOUR COMMUTE: {args.commute}\n"

        if "NORTH YORK" in commute_upper and "DOWNTOWN" in commute_upper:
            dvp = corridor_congestion.get("DVP", {}).get("avg_level", 0)
            yonge = corridor_congestion.get("YONGE ST", {}).get("avg_level", 0)
            if dvp >= 2:
                report += "  → DVP congested — take Bayview or Don Mills instead\n"
                report += "  → Or take Line 1 Yonge-University subway\n"
            elif yonge >= 2:
                report += "  → Yonge slow — DVP is faster right now\n"
            else:
                report += "  → Both DVP and Yonge flowing — take your usual route\n"

        elif "SCARBOROUGH" in commute_upper:
            report += "  → Check DVP/401 interchange and Kingston Rd\n"
        elif "MISSISSAUGA" in commute_upper or "AIRPORT" in commute_upper:
            gard = corridor_congestion.get("GARDINER", {}).get("avg_level", 0)
            if gard >= 2:
                report += "  → Gardiner congested — consider 401 or Lakeshore\n"
            else:
                report += "  → Gardiner flowing — standard route is fine\n"

    print(report)

    # Save for dashboard
    ts_file = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(STATE_DIR / f"actionable_{ts_file}.json", "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "results": results,
            "corridor_congestion": {k: {"avg_level": v["avg_level"]}
                                     for k, v in corridor_congestion.items()},
            "ttc_delays": ttc_delays,
            "disruptions": disruptions[:10],
            "live_events": len(live_events),
            "live_restrictions": len(live_restrictions),
            "large_events": [e["name"] for e in live_events
                             if e["category"] in ("large", "medium")][:10],
        }, f, default=str)


if __name__ == "__main__":
    main()
