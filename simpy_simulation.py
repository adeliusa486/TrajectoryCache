"""
DEPRECATED LEGACY SCRIPT — DO NOT USE FOR NEW EXPERIMENTS
==========================================================
This file was the original prototype implementation of TrajectoryCache used
to generate early exploratory results. It contains known parameter mismatches
with the canonical implementation in src/trajectorycache/:

    GRZ_RADIUS = 150.0   →  should be r_rel = 800.0  (configs/simulation.yaml)
    T_PREDICT  = 3.0     →  should be t_pred = 30.0  (configs/simulation.yaml)
    ALPHA      = 0.5     →  should be alpha_d = 0.1  (configs/simulation.yaml)
    W          = 0.5     →  should be W = 0.2 (optimal from sweep)

These parameter differences cause the urgency signal to fire almost never
(r_rel=150 on a 10 km road), producing incorrect results.

For all reproducible experiments, use:
    python scripts/run_benchmark.py              (single seed)
    python scripts/run_multiseed.py              (10 seeds, alpha sweep)
    python scripts/run_density_sweep.py          (density analysis)
    python scripts/run_wsweep.py                 (W sensitivity)

This file is retained for historical reference only and will be removed
in a future cleanup commit.
"""
import warnings
warnings.warn(
    "simpy_simulation.py is a deprecated legacy prototype with wrong TC parameters. "
    "Use scripts/run_benchmark.py instead.",
    DeprecationWarning, stacklevel=2
)

import simpy
import random
import statistics

# --- CONFIGURATION ---
HIGHWAY_LENGTH = 10000       # meters
TOWER_POS = 2500            # middle of highway
TOWER_RADIUS = 300          # coverage radius
NUM_FILES = 200
SIMULATION_TIME = 3600      # 1 simulated hour
FILE_SIZE_MB = 1


class LRUCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.hits = 0
        self.misses = 0
        self.store = []

    def request(self, file_id, current_time, active_cars):
        if file_id in self.store:
            self.hits += 1
            self.store.remove(file_id)
            self.store.append(file_id)
        else:
            self.misses += 1
            if len(self.store) >= self.capacity:
                self.store.pop(0)
            self.store.append(file_id)

class LFUCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.hits = 0
        self.misses = 0
        self.store = []
        self.window = []

    def request(self, file_id, current_time, active_cars):
        self.window.append((current_time, file_id))
        while self.window and self.window[0][0] < current_time - 300:
            self.window.pop(0)

        if file_id in self.store:
            self.hits += 1
            self.store.remove(file_id)
            self.store.append(file_id)
        else:
            self.misses += 1
            if len(self.store) >= self.capacity:
                counts = {f: 0 for f in self.store}
                for w_time, w_file in self.window:
                    if w_file in counts:
                        counts[w_file] += 1
                
                min_count = min(counts.values())
                for f in self.store:
                    if counts[f] == min_count:
                        self.store.remove(f)
                        break
            self.store.append(file_id)

class FIFOCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.hits = 0
        self.misses = 0
        self.store = []

    def request(self, file_id, current_time, active_cars):
        if file_id in self.store:
            self.hits += 1
        else:
            self.misses += 1
            if len(self.store) >= self.capacity:
                self.store.pop(0)
            self.store.append(file_id)

class RandomCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.hits = 0
        self.misses = 0
        self.store = []

    def request(self, file_id, current_time, active_cars):
        if file_id in self.store:
            self.hits += 1
        else:
            self.misses += 1
            if len(self.store) >= self.capacity:
                idx = random.randint(0, len(self.store) - 1)
                self.store.pop(idx)
            self.store.append(file_id)

class TCCache:
    def __init__(self, capacity, file_locations):
        self.capacity = capacity
        self.hits = 0
        self.misses = 0
        self.store = []
        self.file_locations = file_locations
        self.GRZ_RADIUS = 150.0
        self.ALPHA = 0.5
        self.T_PREDICT = 3.0
        self.window = []
        self.W = 0.5

    def _calc_urgency_score(self, f_id, active_cars):
        file_location = self.file_locations[f_id]
        score = 0.0
        for car in active_cars:
            predicted_car_position = car['pos'] + car['speed'] * car['dir'] * self.T_PREDICT
            distance = abs(file_location - predicted_car_position)
            
            if distance <= self.GRZ_RADIUS:
                time_to_arrive = distance / car['speed']
                urgency = 1.0 / (1.0 + self.ALPHA * time_to_arrive)
                score += urgency
        return score

    def request(self, file_id, current_time, active_cars):
        self.window.append((current_time, file_id))
        while self.window and self.window[0][0] < current_time - 300:
            self.window.pop(0)

        if file_id in self.store:
            self.hits += 1
            self.store.remove(file_id)
            self.store.append(file_id)
        else:
            self.misses += 1
            if len(self.store) < self.capacity:
                self.store.append(file_id)
            else:
                candidates = self.store + [file_id]
                urgencies = {f: self._calc_urgency_score(f, active_cars) for f in candidates}
                
                counts = {f: 0 for f in candidates}
                for w_time, w_file in self.window:
                    if w_file in counts:
                        counts[w_file] += 1
                        
                max_count = max(counts.values()) if counts else 0
                popularities = {}
                for f in candidates:
                    popularities[f] = (counts[f] / max_count) if max_count > 0 else 0.0
                    
                scores = {}
                for f in candidates:
                    scores[f] = self.W * urgencies[f] + (1.0 - self.W) * popularities[f]
                
                cache_scores = {f: scores[f] for f in self.store}
                lowest_score_file = min(cache_scores, key=cache_scores.get)
                lowest_score = cache_scores[lowest_score_file]
                
                new_score = scores[file_id]
                
                if lowest_score < new_score:
                    self.store.remove(lowest_score_file)
                    self.store.append(file_id)

class Car:
    def __init__(self, id, speed, direction, start_pos, start_time):
        self.id = id
        self.speed = speed
        self.direction = direction
        self.start_pos = start_pos
        self.start_time = start_time
        self.requested_files = set()

def car_process(env, car, file_locations, pop_prob, caches, active_cars_dict):
    if car.start_time > 0:
        yield env.timeout(car.start_time)
        

    
    while env.now < SIMULATION_TIME:
        dt = env.now - car.start_time
        pos = car.start_pos + car.direction * car.speed * dt
        
        if pos < 0 or pos > HIGHWAY_LENGTH:
            if car.id in active_cars_dict:
                del active_cars_dict[car.id]
            break
            
        if abs(pos - TOWER_POS) <= TOWER_RADIUS:
            # Update state in dict
            active_cars_dict[car.id] = {'id': car.id, 'pos': pos, 'speed': car.speed, 'dir': car.direction}
            
            best_file = -1
            min_dist = float('inf')
            eligible_files = []
            
            for f_id in range(len(file_locations)):
                f_pos = file_locations[f_id]
                dist = abs(f_pos - pos)
                
                is_ahead = False
                if car.direction == 1 and f_pos > pos:
                    is_ahead = True
                elif car.direction == -1 and f_pos < pos:
                    is_ahead = True
                    
                if is_ahead and dist <= 300:
                    if f_id not in car.requested_files:
                        if random.random() < pop_prob[f_id]:
                            eligible_files.append((f_id, dist))
                            
            if eligible_files:
                eligible_files.sort(key=lambda x: x[1])
                best_file = eligible_files[0][0]
                car.requested_files.add(best_file)
                            
            if best_file != -1:
                active_cars = list(active_cars_dict.values())
                for cache in caches:
                    cache.request(best_file, env.now, active_cars)
                
                f_pos = file_locations[best_file]
                dist_to_file = abs(f_pos - pos)
                time_to_pass = dist_to_file / car.speed
                
                wait_time = max(min(time_to_pass + 0.001, 5.0), 0.01)
                yield env.timeout(wait_time)
            else:
                yield env.timeout(1.0)
        else:
            if car.id in active_cars_dict:
                del active_cars_dict[car.id]
                
            if car.direction == 1 and pos < TOWER_POS - TOWER_RADIUS:
                dist_to_entry = (TOWER_POS - TOWER_RADIUS) - pos
                yield env.timeout(max(dist_to_entry / car.speed, 0.01))
            elif car.direction == -1 and pos > TOWER_POS + TOWER_RADIUS:
                dist_to_entry = pos - (TOWER_POS + TOWER_RADIUS)
                yield env.timeout(max(dist_to_entry / car.speed, 0.01))
            else:
                break

def run_simulation(seed, NUM_CARS=300, CACHE_CAPACITY=20, NUM_FILES=100, w_sweep=None, zipf_alpha=0.8):
    random.seed(seed)
    import numpy as np
    np.random.seed(seed)
    
    cache_capacity = CACHE_CAPACITY
    
    file_locations = [random.uniform(0, HIGHWAY_LENGTH) for _ in range(NUM_FILES)]
    
    ranks = np.arange(1, NUM_FILES + 1)
    zipf_probs = (1.0 / ranks**zipf_alpha)
    zipf_probs = zipf_probs / np.sum(zipf_probs)
    max_prob = np.max(zipf_probs)
    pop_prob = (zipf_probs / max_prob) * 0.9
    
    cars = []
    for i in range(NUM_CARS):
        speed = random.uniform(15, 30)
        direction = random.choice([1, -1])
        start_pos = 0 if direction == 1 else HIGHWAY_LENGTH
        dist_to_coverage = 2200 if direction == 1 else 2200
        time_to_coverage = dist_to_coverage / speed
        max_start_time = SIMULATION_TIME - time_to_coverage - 10
        start_time = random.uniform(0, max(0, max_start_time))
        cars.append(Car(i, speed, direction, start_pos, start_time))
        
    env = simpy.Environment()
    active_cars_dict = {}
    
    lru = LRUCache(cache_capacity)
    lfu = LFUCache(cache_capacity)
    fifo = FIFOCache(cache_capacity)
    rand_cache = RandomCache(cache_capacity)
    
    if w_sweep is None:
        w_sweep = [0.5]
    
    tc_caches = {w: TCCache(cache_capacity, file_locations) for w in w_sweep}
    for w in w_sweep:
        tc_caches[w].W = w
    
    caches = [lru, lfu, fifo, rand_cache] + list(tc_caches.values())
    
    for car in cars:
        env.process(car_process(env, car, file_locations, pop_prob, caches, active_cars_dict))
        
    env.run(until=SIMULATION_TIME)
    
    return lru, lfu, fifo, rand_cache, tc_caches

def print_stats(name, cache):
    total = cache.hits + cache.misses
    miss_rate = (cache.misses / total * 100) if total > 0 else 0
    print(f"{name:<4} hits={cache.hits:<4} misses={cache.misses:<4} total={total:<4} miss_rate={miss_rate:.2f}%", flush=True)
    return miss_rate

if __name__ == "__main__":
    seeds = [42, 123, 456, 789, 1001, 2024, 555, 999, 7777, 3141]
    w_values = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]

    lru_miss_rates = []
    lfu_miss_rates = []
    tc_miss_rates = {w: [] for w in w_values}

    for seed in seeds:
        print(f"\nSeed {seed} - SimPy TrajectoryCache (NUM_CARS=600, NUM_FILES=200, CACHE=20, alpha=0.8)", flush=True)
        lru, lfu, fifo, rand_cache, tc_caches = run_simulation(
            seed, NUM_CARS=600, NUM_FILES=200, CACHE_CAPACITY=20,
            w_sweep=w_values, zipf_alpha=0.8
        )
        lru_miss = print_stats("LRU:", lru)
        lfu_miss = print_stats("LFU:", lfu)
        for w in w_values:
            tc_miss = print_stats(f"TC(W={w}):", tc_caches[w])
            tc_miss_rates[w].append(tc_miss)

        lru_miss_rates.append(lru_miss)
        lfu_miss_rates.append(lfu_miss)
        print()

    import statistics
    print("--- SimPy Summary across 10 seeds (config matches paper) ---")
    print(f"LRU: mean={statistics.mean(lru_miss_rates):.2f}% std={statistics.stdev(lru_miss_rates):.2f}%")
    print(f"LFU: mean={statistics.mean(lfu_miss_rates):.2f}% std={statistics.stdev(lfu_miss_rates):.2f}%")
    for w in w_values:
        print(f"TC(W={w}): mean={statistics.mean(tc_miss_rates[w]):.2f}% std={statistics.stdev(tc_miss_rates[w]):.2f}%")
