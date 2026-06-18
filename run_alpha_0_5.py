import random
import matplotlib.pyplot as plt
import numpy as np
import simpy_simulation
import sumo_cache_sim

NUM_SEEDS = 10
random.seed(42)
seeds = [random.randint(1000, 99999) for _ in range(NUM_SEEDS)]

w_values = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
baselines = ['LRU', 'LFU', 'FIFO', 'Random']
all_algos = baselines + w_values

# Dictionaries to store raw misses and totals
simpy_misses = {a: [] for a in all_algos}
simpy_totals = {a: [] for a in all_algos}
sumo_misses = {a: [] for a in all_algos}
sumo_totals = {a: [] for a in all_algos}

print(f"Starting Journal Evaluation Sweep across {NUM_SEEDS} seeds...")

for idx, seed in enumerate(seeds):
    if idx % 5 == 0:
        print(f"Processing seed {idx+1}/{NUM_SEEDS}...")
        
    # --- SIMPY ---
    lru, lfu, fifo, rand_cache, tc_caches = simpy_simulation.run_simulation(seed, NUM_CARS=600, NUM_FILES=200, CACHE_CAPACITY=20, w_sweep=w_values, zipf_alpha=0.5)
    
    simpy_misses['LRU'].append(lru.misses)
    simpy_totals['LRU'].append(lru.hits + lru.misses)
    simpy_misses['LFU'].append(lfu.misses)
    simpy_totals['LFU'].append(lfu.hits + lfu.misses)
    simpy_misses['FIFO'].append(fifo.misses)
    simpy_totals['FIFO'].append(fifo.hits + fifo.misses)
    simpy_misses['Random'].append(rand_cache.misses)
    simpy_totals['Random'].append(rand_cache.hits + rand_cache.misses)
    
    for w in w_values:
        tc = tc_caches[w]
        simpy_misses[w].append(tc.misses)
        simpy_totals[w].append(tc.hits + tc.misses)
        
    # --- SUMO ---
    s_lru, s_lfu, s_fifo, s_rand, s_tc_caches = sumo_cache_sim.run_sumo_simulation(seed, zipf_alpha=0.5)
    
    sumo_misses['LRU'].append(s_lru.misses)
    sumo_totals['LRU'].append(s_lru.hits + s_lru.misses)
    sumo_misses['LFU'].append(s_lfu.misses)
    sumo_totals['LFU'].append(s_lfu.hits + s_lfu.misses)
    sumo_misses['FIFO'].append(s_fifo.misses)
    sumo_totals['FIFO'].append(s_fifo.hits + s_fifo.misses)
    sumo_misses['Random'].append(s_rand.misses)
    sumo_totals['Random'].append(s_rand.hits + s_rand.misses)
    
    for w in w_values:
        s_tc = s_tc_caches[w]
        sumo_misses[w].append(s_tc.misses)
        sumo_totals[w].append(s_tc.hits + s_tc.misses)

# --- METRIC CALCULATION ---
def calc_metrics(misses_arr, totals_arr):
    miss_rate = []
    latency = []
    backhaul = []
    
    for i in range(len(misses_arr)):
        m = misses_arr[i]
        t = totals_arr[i]
        h = t - m
        
        if t > 0:
            mr = (m / t) * 100.0
            lat = (h * 5 + m * 50) / t
            bw = h * 1  # 1MB per file hit saved from backhaul
        else:
            mr = 0
            lat = 0
            bw = 0
            
        miss_rate.append(mr)
        latency.append(lat)
        backhaul.append(bw)
        
    return np.mean(miss_rate), np.mean(latency), np.mean(backhaul)

results_simpy = {}
results_sumo = {}

for a in all_algos:
    results_simpy[a] = calc_metrics(simpy_misses[a], simpy_totals[a])
    results_sumo[a] = calc_metrics(sumo_misses[a], sumo_totals[a])

# --- PLOTTING ---
fig, axs = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Journal Evaluation: TrajectoryCache vs Baselines (50 Seeds)', fontsize=16)

metrics = ['Cache Miss Rate (%)', 'Avg Download Latency (ms)', 'Backhaul Traffic Saved (MB)']

for row, (env_name, res_dict) in enumerate([('SimPy (Independent Traffic)', results_simpy), ('SUMO (Platooning Traffic)', results_sumo)]):
    print(f"\n--- {env_name.upper()} ---")
    for a in all_algos:
        mr, lat, bw = res_dict[a]
        print(f"{a:>10}: Miss Rate = {mr:>5.2f}%, Latency = {lat:>6.2f}ms, Backhaul Saved = {bw:>6.2f}MB")
        
    for col, metric_name in enumerate(metrics):
        ax = axs[row, col]
        
        tc_vals = [res_dict[w][col] for w in w_values]
        ax.plot(w_values, tc_vals, 'o-', color='green', linewidth=2, label='TrajectoryCache')
        
        ax.axhline(res_dict['LRU'][col], color='red', linestyle='--', label='LRU')
        ax.axhline(res_dict['LFU'][col], color='blue', linestyle='--', label='LFU')
        ax.axhline(res_dict['FIFO'][col], color='purple', linestyle=':', label='FIFO')
        ax.axhline(res_dict['Random'][col], color='orange', linestyle=':', label='Random')
        
        if row == 0:
            ax.set_title(metric_name)
        
        ax.set_xlabel('W (Urgency Weight)')
        
        if col == 0:
            ax.set_ylabel(env_name)
            
        ax.grid(True)
        if row == 0 and col == 0:
            ax.legend(fontsize=8)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("alpha_0_5_results.png", dpi=300)
print("\nEvaluation Complete! Graph saved as alpha_0_5_results.png")
