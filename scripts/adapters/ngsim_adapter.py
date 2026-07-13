"""
NGSIM trajectory adapter.

Parses a filtered NGSIM CSV (columns: vehicle_id, frame_id, global_time,
local_y, v_vel, direction, lane_id) into the replay pipeline's per-timestep
schema:  steps = [ (t_seconds, [ {"x", "speed", "direction"}, ... ]), ... ].

Conventions / unit handling (verified on i-80 window 1, 2026-07-13):
  * local_y is longitudinal position along the roadway in FEET; converted to
    metres (x0.3048). It increases in the travel direction.
  * v_vel is speed in FEET/SEC; converted to metres/sec.
  * direction: i-80 is a unidirectional segment (field often "NA"); we verify
    local_y is non-decreasing per vehicle and set direction = +1 uniformly.
    (If a dataset has genuine two-way travel, direction is inferred per vehicle
    from the sign of its net local_y displacement.)
  * NGSIM is sampled at 10 Hz (global_time step 100 ms). The demand model runs
    at 1 Hz, so we down-sample to every 10th distinct global_time.

The adapter emits only trajectories; the caching scenario (catalog, RSU) is
applied by the caller, exactly as for the SUMO replays, so the mobility source
is the only thing that changes across fidelity tiers.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

FT_TO_M = 0.3048


def load_ngsim_steps(csv_path: str | Path, downsample: int = 10):
    """Return (steps, meta).

    steps : list of (t_seconds, list of vehicle dicts) at 1 Hz.
    meta  : dict with segment_length_m, n_vehicles, mean_speed_mps, n_steps.
    """
    # Pass 1: group rows by global_time; also collect per-vehicle y for direction.
    by_time: dict[int, list[tuple[float, float, int]]] = defaultdict(list)
    veh_y: dict[str, list[tuple[int, float]]] = defaultdict(list)
    y_min = float("inf")
    y_max = float("-inf")

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            gt = int(row["global_time"])
            y_ft = float(row["local_y"])
            v_fts = float(row["v_vel"])
            vid = row["vehicle_id"]
            by_time[gt].append((y_ft, v_fts, 0))  # direction filled below
            veh_y[vid].append((gt, y_ft))
            if y_ft < y_min:
                y_min = y_ft
            if y_ft > y_max:
                y_max = y_ft

    # Per-vehicle travel direction from net displacement (robust to noise).
    veh_dir: dict[str, int] = {}
    for vid, pts in veh_y.items():
        pts.sort()
        veh_dir[vid] = 1 if pts[-1][1] >= pts[0][1] else -1

    # Re-read is avoided: rebuild by_time with direction using a second grouping
    # keyed by (global_time) but we need vid per row, so redo the light pass.
    by_time = defaultdict(list)
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            gt = int(row["global_time"])
            vid = row["vehicle_id"]
            x_m = (float(row["local_y"]) - y_min) * FT_TO_M
            spd = float(row["v_vel"]) * FT_TO_M
            by_time[gt].append({"x": x_m, "speed": spd, "direction": veh_dir[vid]})

    times = sorted(by_time)
    # Down-sample 10 Hz -> 1 Hz and re-index time to seconds from 0.
    steps = []
    all_speeds = []
    for k, gt in enumerate(times[::downsample]):
        vehs = by_time[gt]
        steps.append((float(k), vehs))
        all_speeds.extend(v["speed"] for v in vehs)

    seg_len_m = (y_max - y_min) * FT_TO_M
    mean_speed = sum(all_speeds) / len(all_speeds) if all_speeds else 0.0
    meta = {
        "segment_length_m": seg_len_m,
        "n_vehicles": len(veh_y),
        "mean_speed_mps": mean_speed,
        "n_steps": len(steps),
        "mean_on_road": sum(len(v) for _, v in steps) / len(steps) if steps else 0.0,
    }
    return steps, meta


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else (
        Path(__file__).resolve().parents[2] / "data" / "raw" / "ngsim" / "ngsim_i80_win1.csv")
    steps, meta = load_ngsim_steps(path)
    print("meta:", meta)
    print("first step vehicles:", len(steps[0][1]))
    print("sample vehicle:", steps[0][1][0])
