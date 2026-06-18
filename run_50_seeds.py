import subprocess
import os
import random
import matplotlib.pyplot as plt
import numpy as np

import simpy_simulation
import sumo_cache_sim
import create_sumo_files

# 1. 50 random seeds
random.seed(42)
seeds_50 = [random.randint(1000, 99999) for _ in range(50)]

print(f"Generated 50 random seeds: {seeds_50[:5]}...")

w_values = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]

# --- SIMPY SWEEP ---
print("Running SimPy 50-seed sweep...")
simpy_results = {'LRU': [], 'LFU': []}
for w in w_values:
    simpy_results[w] = []

for idx, seed in enumerate(seeds_50):
    if (idx+1) % 10 == 0:
        print(f"SimPy progress: {idx+1}/50")
    
    # Run the exact same config as before: cache size 20, 100 files, 10km
    lru, lfu, tc_caches = simpy_simulation.run_simulation(seed, NUM_CARS=600, NUM_FILES=100, CACHE_CAPACITY=20, w_sweep=w_values)
    
    total = lru.hits + lru.misses
    simpy_results['LRU'].append(lru.misses / total * 100 if total > 0 else 0)
    simpy_results['LFU'].append(lfu.misses / total * 100 if total > 0 else 0)
    for w in w_values:
        tc = tc_caches[w]
        simpy_results[w].append(tc.misses / total * 100 if total > 0 else 0)

simpy_mean = {k: np.mean(v) for k, v in simpy_results.items()}

# --- SUMO SWEEP ---
print("Generating SUMO route files...")
create_sumo_files.create_network_files()
for seed in seeds_50:
    create_sumo_files.create_routes_file(seed)
    create_sumo_files.create_sumo_config(seed)

print("Running SUMO executable for all 50 seeds (this will take a few minutes)...")
# Run SUMO
sumo_exe = r"C:\Program Files (x86)\Eclipse\Sumo\bin\sumo.exe"
for idx, seed in enumerate(seeds_50):
    if (idx+1) % 10 == 0:
        print(f"SUMO exe progress: {idx+1}/50")
    cfg = f"highway_{seed}.sumocfg"
    result = subprocess.run([sumo_exe, "-c", cfg], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error running SUMO for seed {seed}: {result.stderr}")
        raise RuntimeError(f"SUMO failed for seed {seed}")

print("Parsing XML and running cache simulation for SUMO traces...")
sumo_results = {'LRU': [], 'LFU': []}
for w in w_values:
    sumo_results[w] = []

for idx, seed in enumerate(seeds_50):
    if (idx+1) % 10 == 0:
        print(f"SUMO parse progress: {idx+1}/50")
    
    lru, lfu, tc_caches = sumo_cache_sim.run_sumo_simulation(seed)
    total = lru.hits + lru.misses
    sumo_results['LRU'].append(lru.misses / total * 100 if total > 0 else 0)
    sumo_results['LFU'].append(lfu.misses / total * 100 if total > 0 else 0)
    for w in w_values:
        tc = tc_caches[w]
        sumo_results[w].append(tc.misses / total * 100 if total > 0 else 0)

sumo_mean = {k: np.mean(v) for k, v in sumo_results.items()}

print("\n--- FINAL SIMPY MEANS ---")
print(f"LRU: {simpy_mean['LRU']:.2f}% | LFU: {simpy_mean['LFU']:.2f}%")
for w in w_values:
    print(f"W={w}: {simpy_mean[w]:.2f}%")
    
print("\n--- FINAL SUMO MEANS ---")
print(f"LRU: {sumo_mean['LRU']:.2f}% | LFU: {sumo_mean['LFU']:.2f}%")
for w in w_values:
    print(f"W={w}: {sumo_mean[w]:.2f}%")

# --- PLOTTING ---
plt.figure(figsize=(12, 5))

# Plot 1: SimPy
plt.subplot(1, 2, 1)
w_list = [w for w in w_values]
simpy_tc_means = [simpy_mean[w] for w in w_values]

plt.plot(w_list, simpy_tc_means, 'o-', color='green', linewidth=2, label='TrajectoryCache')
plt.axhline(simpy_mean['LRU'], color='red', linestyle='--', label='LRU Baseline')
plt.axhline(simpy_mean['LFU'], color='blue', linestyle='--', label='LFU Baseline')

plt.title('SimPy (Independent Traffic)\n50-Seed Average Miss Rate vs. W')
plt.xlabel('W (Urgency Weight)')
plt.ylabel('Cache Miss Rate (%)')
plt.legend()
plt.grid(True)

# Plot 2: SUMO
plt.subplot(1, 2, 2)
sumo_tc_means = [sumo_mean[w] for w in w_values]

plt.plot(w_list, sumo_tc_means, 'o-', color='green', linewidth=2, label='TrajectoryCache')
plt.axhline(sumo_mean['LRU'], color='red', linestyle='--', label='LRU Baseline')
plt.axhline(sumo_mean['LFU'], color='blue', linestyle='--', label='LFU Baseline')

plt.title('SUMO Krauss Model (Platooning)\n50-Seed Average Miss Rate vs. W')
plt.xlabel('W (Urgency Weight)')
plt.ylabel('Cache Miss Rate (%)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("50_seed_sweep_results.png", dpi=300)
print("Graph saved as 50_seed_sweep_results.png")
