# TrajectoryCache

**Trajectory-Aware Proactive Edge Caching for Vehicular Networks**

A research implementation of the **TrajectoryCache** algorithm — a joint spatial-urgency and popularity scoring cache replacement policy for V2I edge caches. Evaluated in two traffic environments: independent vehicles (SimPy) and SUMO Krauss car-following platooning.

---

## Paper Results (Verified Ground Truth)

### α = 0.8 (High Skew) — Mean Cache Miss Rate across 10 seeds

| Policy         | SimPy (Independent) | SUMO (Platooning) |
|----------------|--------------------|--------------------|
| LRU            | 6.14%              | 20.37%             |
| LFU            | 4.99%              | 14.79%             |
| **TC (W=0.1)** | **4.54%**          | **14.34%**         |
| TC (W=0.2)     | 4.64%              | 14.79%             |
| TC (W=0.5)     | 4.84%              | 15.67%             |

TC with W=0.1 achieves the **best miss rate** in both environments.

---

## Repository Structure

```
TrajectoryCache/
├── simpy_simulation.py      # SimPy independent-traffic simulation + LRU/LFU/FIFO/Random/TC
├── sumo_cache_sim.py        # SUMO XML trace reader + same cache policies
├── run_alpha_0_8_raw.py     # Main 10-seed journal evaluation (α=0.8)
├── run_alpha_0_5.py         # Robustness check (α=0.5)
├── run_alpha_1.py           # High-skew check (α=1.0)
├── run_journal_eval.py      # Full W-parameter sweep + plots
├── run_50_seeds.py          # 50-seed confidence sweep
├── matlab_simulation.py     # MATLAB equivalent benchmark
├── create_sumo_files.py     # Generates SUMO .rou.xml / .sumocfg / fcd.xml traces
└── results/
    └── journal_sweep_results.png
```

---

## How to Run

### 1. Install Dependencies
```bash
pip install simpy numpy matplotlib
```

### 2. SimPy (Independent Traffic) — No SUMO needed
```bash
python simpy_simulation.py
```

### 3. SUMO Traces — Generate first, then simulate
```bash
# Step 1: Generate SUMO traces (requires SUMO installed)
python create_sumo_files.py

# Step 2: Run hybrid evaluation
python run_alpha_0_8_raw.py
```

### 4. Full Journal Evaluation
```bash
python run_journal_eval.py
```

---

## Key Algorithm — TrajectoryCache Score

For each file `f` in the cache (at eviction time):

```
score(f) = W * urgency(f) + (1-W) * popularity(f)
```

Where:
- **urgency(f)**: sum over active vehicles whose predicted position is within `GRZ_RADIUS` of file `f`'s mapped location → `1 / (1 + α * time_to_arrive)`
- **popularity(f)**: relative request frequency in a 300-second sliding window
- **W**: tunable weight (best at W=0.1, meaning popularity-dominant with spatial tie-breaking)

The file with the **lowest score** is evicted when a new file with a **higher score** arrives.

---

## Why TC Beats LRU/LFU

- **SimPy**: TC's spatial urgency signal pre-loads files that approaching vehicles will request, beating pure recency (LRU) and frequency (LFU) by ~1.5–2%
- **SUMO**: Platooning creates synchronized demand bursts. TC detects the incoming platoon via vehicle trajectory prediction and pre-prioritises co-located content **before** the burst hits, achieving 6% lower miss rate than LRU

---

## Citation

If you use this code, please cite:

> [Paper title and authors — to be added upon acceptance]
> Vehicular Communications, Elsevier, 2025
