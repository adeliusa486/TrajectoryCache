#!/usr/bin/env python
"""
Download the least-congested (free-flow-onset) window of NGSIM US-101 from the
U.S. DOT open-data portal (Socrata dataset 8ect-6jqj) and write it in the exact
7-column schema the project's ngsim_adapter expects.

Period 0 (approx 07:50-08:05, global_time < 1118847903866) averages ~41 km/h,
markedly faster than the congested I-80 window (~28 km/h), so it exercises the
free-flow regime relevant to the request-radius question.

Output: data/raw/ngsim/ngsim_us101_freeflow.csv
"""
from __future__ import annotations

import csv
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "raw" / "ngsim" / "ngsim_us101_freeflow.csv"
BASE = "https://data.transportation.gov/resource/8ect-6jqj.json"
UA = {"User-Agent": "ngsim-fetch/1.0 (research)"}
COLS = ["vehicle_id", "frame_id", "global_time", "local_y", "v_vel", "direction", "lane_id"]
GT_HI = 1118847903866  # end of US-101 period 0 (free-flow onset)
PAGE = 50000


def fetch(offset):
    soql = (
        f"SELECT {','.join(COLS)} "
        f"WHERE location='us-101' AND global_time < {GT_HI} "
        f"ORDER BY vehicle_id, global_time "
        f"LIMIT {PAGE} OFFSET {offset}"
    )
    url = BASE + "?" + urllib.parse.urlencode({"$query": soql})
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers=UA)
            return json.load(urllib.request.urlopen(req, timeout=120))
        except Exception as e:
            print(f"  retry {attempt+1} (offset {offset}): {e}", flush=True)
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"failed at offset {offset}")


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    n = 0
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f, quoting=csv.QUOTE_ALL)
        w.writerow(COLS)
        offset = 0
        while True:
            rows = fetch(offset)
            if not rows:
                break
            for r in rows:
                w.writerow([r.get(c, "") for c in COLS])
            n += len(rows)
            print(f"  fetched {n} rows ({time.time()-t0:.0f}s)", flush=True)
            if len(rows) < PAGE:
                break
            offset += PAGE
    print(f"saved {n} rows -> {OUT}  ({time.time()-t0:.0f}s)")
    if n == 0:
        sys.exit("no rows fetched")


if __name__ == "__main__":
    main()
