"""
DEPRECATED LEGACY SCRIPT — DO NOT USE FOR NEW EXPERIMENTS
==========================================================
This file parses SUMO FCD XML trace files (fcd_{seed}.xml) that are NOT
included in the repository. It also contains a TCCache implementation with
known parameter mismatches vs. the canonical src/trajectorycache/ package:

    GRZ_RADIUS = 150.0   →  should be r_rel = 800.0  (configs/simulation.yaml)
    T_PREDICT  = 3.0     →  should be t_pred = 30.0  (configs/simulation.yaml)
    ALPHA      = 0.5     →  should be alpha_d = 0.1  (configs/simulation.yaml)

Without the SUMO FCD trace files this script cannot run. The SUMO traces were
generated with SUMO 1.27 using the Krauss car-following model on a straight
10 km highway network. To regenerate them see README.md §Reproducing SUMO Results.

For all reproducible experiments without SUMO, use the canonical kinematic pipeline:
    python scripts/run_benchmark.py
    python scripts/run_multiseed.py

This file is retained for historical reference and will be removed in a future
cleanup commit once SUMO traces are archived in a Zenodo dataset.
"""
import xml.etree.ElementTree as ET
import random
import math
import numpy as np
HIGHWAY_LENGTH = 10000
TOWER_POS = 5000
TOWER_RADIUS = 500
NUM_FILES = 200
CACHE_CAPACITY = 20


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
                
                min_count = min(counts.values()) if counts else 0
                for f in self.store:
                    if f not in counts or counts[f] == min_count:
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
    def __init__(self, capacity, file_locations, w=0.5):
        self.capacity = capacity
        self.hits = 0
        self.misses = 0
        self.store = []
        self.file_locations = file_locations
        self.GRZ_RADIUS = 150.0
        self.ALPHA = 0.5
        self.T_PREDICT = 3.0
        self.window = []
        self.W = w

    def _calc_urgency_score(self, f_id, active_cars):
        file_location = self.file_locations[f_id]
        score = 0.0
        for car in active_cars:
            predicted_car_position = car['pos'] + car['speed'] * car['dir'] * self.T_PREDICT
            distance = abs(file_location - predicted_car_position)
            
            if distance <= self.GRZ_RADIUS:
                if car['speed'] > 0:
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


def run_sumo_simulation(seed, zipf_alpha=0.8):
    np.random.seed(seed)
    random.seed(seed)
    
    file_locations = [random.uniform(0, HIGHWAY_LENGTH) for _ in range(NUM_FILES)]
    
    ranks = np.arange(1, NUM_FILES + 1)
    zipf_probs = (1.0 / ranks**zipf_alpha)
    zipf_probs = zipf_probs / np.sum(zipf_probs)
    max_prob = np.max(zipf_probs)
    pop_prob = (zipf_probs / max_prob) * 0.9
    
    lru = LRUCache(CACHE_CAPACITY)
    lfu = LFUCache(CACHE_CAPACITY)
    fifo = FIFOCache(CACHE_CAPACITY)
    rand_cache = RandomCache(CACHE_CAPACITY)
    
    w_values = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
    tc_caches = {w: TCCache(CACHE_CAPACITY, file_locations, w=w) for w in w_values}
    
    requested_files = {}
    
    tree = ET.parse(f"fcd_{seed}.xml")
    root = tree.getroot()
    
    # Process timestep by timestep
    for timestep in root.findall('timestep'):
        t = float(timestep.get('time'))
        
        active_cars = []
        car_data_map = {}
        
        # Read all cars in this second
        for vehicle in timestep.findall('vehicle'):
            vid = vehicle.get('id')
            x = float(vehicle.get('x'))
            speed = float(vehicle.get('speed'))
            # Calculate direction based on angle. Eastbound angle=90 -> dir=1, Westbound angle=270 -> dir=-1
            angle = float(vehicle.get('angle'))
            direction = 1 if angle < 180 else -1
            
            # Distance from tower
            if abs(x - TOWER_POS) <= TOWER_RADIUS:
                car_info = {'id': vid, 'pos': x, 'speed': speed, 'dir': direction}
                active_cars.append(car_info)
                car_data_map[vid] = car_info
                
        # Now process requests
        for car in active_cars:
            vid = car['id']
            pos = car['pos']
            direction = car['dir']
            
            if vid not in requested_files:
                requested_files[vid] = set()
                
            best_file = -1
            eligible_files = []
                    
            for f_id in range(NUM_FILES):
                f_pos = file_locations[f_id]
                dist = abs(f_pos - pos)
                
                is_ahead = False
                if direction == 1 and f_pos > pos:
                    is_ahead = True
                elif direction == -1 and f_pos < pos:
                    is_ahead = True
                    
                if is_ahead and dist <= 300:
                    if f_id not in requested_files[vid]:
                        if random.random() < pop_prob[f_id]:
                            eligible_files.append((f_id, dist))
                            
            if eligible_files:
                eligible_files.sort(key=lambda x: x[1])
                best_file = eligible_files[0][0]
                requested_files[vid].add(best_file)
                            
            if best_file != -1:
                    
                    lru.request(best_file, t, active_cars)
                    lfu.request(best_file, t, active_cars)
                    fifo.request(best_file, t, active_cars)
                    rand_cache.request(best_file, t, active_cars)
                    for tc in tc_caches.values():
                        tc.request(best_file, t, active_cars)
                    
    return lru, lfu, fifo, rand_cache, tc_caches

def get_stats(cache):
    total = cache.hits + cache.misses
    miss_rate = (cache.misses / total * 100) if total > 0 else 0
    return miss_rate

if __name__ == "__main__":
    seeds = [42, 123, 456, 999, 1024, 7777, 1111, 2222, 3333, 4444]
    
    results = {}
    
    print("--- 10-Seed W-Parameter Sweep (SUMO Hybrid) ---")
    print(f"{'Seed':<6} | {'LRU':<6} | {'LFU':<6} | {'W=0.1':<6} | {'W=0.2':<6} | {'W=0.3':<6} | {'W=0.5':<6} | {'W=0.7':<6} | {'W=0.9':<6}")
    print("-" * 80)
    
    for seed in seeds:
        lru, lfu, tc_caches = run_sumo_simulation(seed)
        lru_miss = get_stats(lru)
        lfu_miss = get_stats(lfu)
        
        tc_res = {}
        for w, tc in tc_caches.items():
            tc_res[w] = get_stats(tc)
            
        results[seed] = {
            'LRU': lru_miss,
            'LFU': lfu_miss,
            **tc_res
        }
        
        print(f"{seed:<6} | {lru_miss:<6.2f} | {lfu_miss:<6.2f} | {tc_res[0.1]:<6.2f} | {tc_res[0.2]:<6.2f} | {tc_res[0.3]:<6.2f} | {tc_res[0.5]:<6.2f} | {tc_res[0.7]:<6.2f} | {tc_res[0.9]:<6.2f}")
        
    print("-" * 80)
    
    w_vals = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
    avg_lru = sum(res['LRU'] for res in results.values()) / len(seeds)
    avg_lfu = sum(res['LFU'] for res in results.values()) / len(seeds)
    
    print("MEAN RESULTS:")
    print(f"{'MEAN':<6} | {avg_lru:<6.2f} | {avg_lfu:<6.2f} | ", end="")
    for w in w_vals:
        avg_tc = sum(res[w] for res in results.values()) / len(seeds)
        print(f"{avg_tc:<6.2f} | ", end="")
    print()
