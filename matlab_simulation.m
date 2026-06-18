% TrajectoryCache Simulation in MATLAB
% This script mirrors the discrete-time Python sandbox.
% It tests LRU, LFU, and TC on the exact same logic.
clear; clc;

% Configuration
HIGHWAY_LENGTH = 10000;
TOWER_POS = 5000;
TOWER_RADIUS = 500;
NUM_FILES = 200;
SIMULATION_TIME = 3600;
NUM_CARS = 600;
CACHE_CAPACITY = 20;

seeds = [999, 1024, 7777];
fprintf('Starting MATLAB Simulation across %d seeds...\n', length(seeds));
lru_total_misses = 0;
lfu_total_misses = 0;
tc_total_misses = 0;
lru_total_reqs = 0;
lfu_total_reqs = 0;
tc_total_reqs = 0;

for s = 1:length(seeds)
    rng(seeds(s)); % Set MATLAB's random seed
    
    file_locations = rand(1, NUM_FILES) * HIGHWAY_LENGTH;
    
    % Car properties
    car_speed = 15 + rand(1, NUM_CARS) * 15;
    car_dir = randi([0, 1], 1, NUM_CARS) * 2 - 1; % 1 or -1
    car_start_pos = zeros(1, NUM_CARS);
    car_start_pos(car_dir == -1) = HIGHWAY_LENGTH;
    
    car_start_time = zeros(1, NUM_CARS);
    for i = 1:NUM_CARS
        time_to_cov = 2200 / car_speed(i);
        max_start = SIMULATION_TIME - time_to_cov - 10;
        if max_start > 0
            car_start_time(i) = rand() * max_start;
        else
            car_start_time(i) = 0;
        end
    end
    
    % Caches
    lru_store = []; lru_hits = 0; lru_misses = 0;
    lfu_store = []; lfu_hits = 0; lfu_misses = 0; lfu_window = [];
    tc_store = []; tc_hits = 0; tc_misses = 0; tc_window = [];
    
    last_req_file = -1 * ones(1, NUM_CARS);
    last_req_time = -100 * ones(1, NUM_CARS);
    
    % Main simulation loop
    for t = 1:SIMULATION_TIME
        active_cars = [];
        % Find active cars in coverage
        for c = 1:NUM_CARS
            dt = t - car_start_time(c);
            if dt >= 0
                pos = car_start_pos(c) + car_dir(c) * car_speed(c) * dt;
                if pos >= 0 && pos <= HIGHWAY_LENGTH && abs(pos - TOWER_POS) <= TOWER_RADIUS
                    active_cars = [active_cars; c, pos, car_speed(c), car_dir(c)];
                end
            end
        end
        
        num_active = size(active_cars, 1);
        
        for idx = 1:num_active
            c = active_cars(idx, 1);
            pos = active_cars(idx, 2);
            speed = active_cars(idx, 3);
            dir = active_cars(idx, 4);
            
            best_file = -1;
            min_dist = inf;
            moved_past_last_file = false;
            
            if last_req_file(c) ~= -1
                last_f_pos = file_locations(last_req_file(c));
                if (dir == 1 && pos > last_f_pos) || (dir == -1 && pos < last_f_pos)
                    moved_past_last_file = true;
                end
            end
            
            for f_id = 1:NUM_FILES
                f_pos = file_locations(f_id);
                dist = abs(f_pos - pos);
                is_ahead = (dir == 1 && f_pos > pos) || (dir == -1 && f_pos < pos);
                
                if is_ahead && dist <= 300
                    if f_id ~= last_req_file(c) && dist < min_dist
                        min_dist = dist;
                        best_file = f_id;
                    end
                end
            end
            
            if best_file ~= -1
                time_passed = (t - last_req_time(c) >= 5);
                if last_req_file(c) == -1 || moved_past_last_file || time_passed
                    % REQUEST FILE
                    last_req_file(c) = best_file;
                    last_req_time(c) = t;
                    
                    % LRU
                    if ismember(best_file, lru_store)
                        lru_hits = lru_hits + 1;
                        lru_store(lru_store == best_file) = [];
                        lru_store = [lru_store, best_file];
                    else
                        lru_misses = lru_misses + 1;
                        if length(lru_store) >= CACHE_CAPACITY
                            lru_store(1) = [];
                        end
                        lru_store = [lru_store, best_file];
                    end
                    
                    % LFU
                    lfu_window = [lfu_window; t, best_file];
                    lfu_window(lfu_window(:,1) < t - 300, :) = [];
                    if ismember(best_file, lfu_store)
                        lfu_hits = lfu_hits + 1;
                        lfu_store(lfu_store == best_file) = [];
                        lfu_store = [lfu_store, best_file];
                    else
                        lfu_misses = lfu_misses + 1;
                        if length(lfu_store) >= CACHE_CAPACITY
                            counts = zeros(1, length(lfu_store));
                            for k = 1:length(lfu_store)
                                counts(k) = sum(lfu_window(:,2) == lfu_store(k));
                            end
                            [~, min_idx] = min(counts);
                            lfu_store(min_idx) = [];
                        end
                        lfu_store = [lfu_store, best_file];
                    end
                    
                    % TC
                    tc_window = [tc_window; t, best_file];
                    tc_window(tc_window(:,1) < t - 300, :) = [];
                    if ismember(best_file, tc_store)
                        tc_hits = tc_hits + 1;
                        tc_store(tc_store == best_file) = [];
                        tc_store = [tc_store, best_file];
                    else
                        tc_misses = tc_misses + 1;
                        if length(tc_store) < CACHE_CAPACITY
                            tc_store = [tc_store, best_file];
                        else
                            candidates = [tc_store, best_file];
                            scores = zeros(1, length(candidates));
                            for k = 1:length(candidates)
                                f = candidates(k);
                                % Urgency
                                u_score = 0;
                                for ac_idx = 1:num_active
                                    ac_pos = active_cars(ac_idx, 2);
                                    ac_speed = active_cars(ac_idx, 3);
                                    ac_dir = active_cars(ac_idx, 4);
                                    
                                    p_pos = ac_pos + ac_dir * ac_speed * 3.0;
                                    d = abs(file_locations(f) - p_pos);
                                    if d <= 150.0
                                        tta = d / ac_speed;
                                        u_score = u_score + (1.0 / (1.0 + 0.5 * tta));
                                    end
                                end
                                
                                % Popularity
                                counts = zeros(1, length(candidates));
                                for cnd = 1:length(candidates)
                                    counts(cnd) = sum(tc_window(:,2) == candidates(cnd));
                                end
                                max_count = max(counts);
                                if max_count > 0
                                    p_score = sum(tc_window(:,2) == f) / max_count;
                                else
                                    p_score = 0;
                                end
                                
                                scores(k) = 0.5 * u_score + 0.5 * p_score;
                            end
                            
                            cache_scores = scores(1:end-1);
                            [~, min_idx] = min(cache_scores);
                            
                            if cache_scores(min_idx) < scores(end)
                                tc_store(min_idx) = [];
                                tc_store = [tc_store, best_file];
                            end
                        end
                    end
                end
            end
        end
    end
    
    t_lru = lru_hits + lru_misses;
    t_lfu = lfu_hits + lfu_misses;
    t_tc = tc_hits + tc_misses;
    
    fprintf('Seed %d -> LRU Miss: %.2f%% | LFU Miss: %.2f%% | TC Miss: %.2f%%\n', ...
        seeds(s), (lru_misses/t_lru)*100, (lfu_misses/t_lfu)*100, (tc_misses/t_tc)*100);
        
    lru_total_misses = lru_total_misses + lru_misses;
    lfu_total_misses = lfu_total_misses + lfu_misses;
    tc_total_misses = tc_total_misses + tc_misses;
    lru_total_reqs = lru_total_reqs + t_lru;
    lfu_total_reqs = lfu_total_reqs + t_lfu;
    tc_total_reqs = tc_total_reqs + t_tc;
end

fprintf('\n--- MATLAB Summary ---\n');
fprintf('LRU Mean Miss Rate: %.2f%%\n', (lru_total_misses/lru_total_reqs)*100);
fprintf('LFU Mean Miss Rate: %.2f%%\n', (lfu_total_misses/lfu_total_reqs)*100);
fprintf('TC Mean Miss Rate:  %.2f%%\n', (tc_total_misses/tc_total_reqs)*100);
