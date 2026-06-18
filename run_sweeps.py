import simulation

def run_sweeps():
    # Hour 1 - Density sweep
    print("HOUR 1: Density Sweep")
    densities = [100, 200, 300, 400, 500]
    for d in densities:
        simulation.NUM_CARS = d
        print(f"\n--- Density: {d} cars ---")
        lru, lfu, tc = simulation.run_simulation(42)
        simulation.print_stats("LRU:", lru)
        simulation.print_stats("LFU:", lfu)
        simulation.print_stats("TC:", tc)

    # Hour 2 - Cache size sweep
    print("\n====================================")
    print("HOUR 2: Cache Size Sweep")
    simulation.NUM_CARS = 300 # Reset
    sizes = [5, 10, 20, 40, 80]
    for s in sizes:
        simulation.CACHE_CAPACITY_MB = s
        simulation.CACHE_CAPACITY = s // simulation.FILE_SIZE_MB
        print(f"\n--- Cache Size: {s} MB ---")
        lru, lfu, tc = simulation.run_simulation(42)
        simulation.print_stats("LRU:", lru)
        simulation.print_stats("LFU:", lfu)
        simulation.print_stats("TC:", tc)

    # Hour 3 - Ablation Study
    print("\n====================================")
    print("HOUR 3: Ablation Study")
    simulation.CACHE_CAPACITY_MB = 20
    simulation.CACHE_CAPACITY = 20 // simulation.FILE_SIZE_MB # Reset

    # (a) TC-Full as-is
    print("\n--- TC-Full ---")
    lru, lfu, tc_full = simulation.run_simulation(42)
    simulation.print_stats("LRU:", lru)
    simulation.print_stats("LFU:", lfu)
    simulation.print_stats("TC:", tc_full)

    # We will use monkey-patching to alter the TC class for the ablations
    original_init = simulation.TCCache.__init__

    # (b) TC-NoPred
    print("\n--- TC-NoPred (T_PREDICT = 0.0) ---")
    def no_pred_init(self, capacity, file_locations):
        original_init(self, capacity, file_locations)
        self.T_PREDICT = 0.0
    simulation.TCCache.__init__ = no_pred_init
    _, _, tc_nopred = simulation.run_simulation(42)
    simulation.print_stats("TC:", tc_nopred)

    # (c) TC-NoPop
    print("\n--- TC-NoPopularity (W = 1.0) ---")
    def no_pop_init(self, capacity, file_locations):
        original_init(self, capacity, file_locations)
        self.W = 1.0
    simulation.TCCache.__init__ = no_pop_init
    _, _, tc_nopop = simulation.run_simulation(42)
    simulation.print_stats("TC:", tc_nopop)

    # (d) TC-EqualWeight
    print("\n--- TC-EqualWeight ---")
    simulation.TCCache.__init__ = original_init # Reset init
    original_calc_urgency = simulation.TCCache._calc_urgency_score
    def equal_weight_calc(self, f_id, active_cars):
        file_location = self.file_locations[f_id]
        score = 0.0
        for car in active_cars:
            predicted_car_position = car['pos'] + car['speed'] * car['dir'] * self.T_PREDICT
            distance = abs(file_location - predicted_car_position)
            if distance <= self.GRZ_RADIUS:
                score += 1.0
        return score
    simulation.TCCache._calc_urgency_score = equal_weight_calc
    _, _, tc_eqweight = simulation.run_simulation(42)
    simulation.print_stats("TC:", tc_eqweight)
    
    # Restore original method to be safe
    simulation.TCCache._calc_urgency_score = original_calc_urgency

if __name__ == "__main__":
    run_sweeps()
