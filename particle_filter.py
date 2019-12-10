import csv
import numpy as np
import scipy as sp
from IPython import embed
import scipy.stats as stats
from tqdm import tqdm

def read_data(filename = "simulated_dataset.csv"):
    header_flag = True
    terrain_map = []
    measurements = []
    true_aircraft_pos = []
    with open(filename, newline = '') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in spamreader:
            if header_flag:
                header_flag = False
            else:
                terrain_map.append(float(row[1]))
                if (row[2] != 'NA'): measurements.append(float(row[2]))
                if (row[3] != 'NA'): true_aircraft_pos.append(float(row[3]))
    return terrain_map, measurements, true_aircraft_pos

def read_config(filename = "config.csv"):
    with open(filename, newline = '') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        rows = []
        for row in spamreader:
            rows.append(row)
    return [int(i) for i in rows[1][1:]]
        

terrain_map, measurements, true_aircraft_pos = read_data()
terrain_map = np.array(terrain_map)
measurements = np.array(measurements)
true_aircraft_pos = np.array(true_aircraft_pos)

min_v, max_v, aircraft_height, noise_mean, noise_sd = read_config()

class node:
    def __init__(self, name_, parent_ = None):
        self.val = name_
        self.parent = parent_
        self.prob = 0
        if parent_:
            self.likelihood = parent_.likelihood
        else:
            self.likelihood = 0

def calc_likelihood(measured_dist, ind):
    assert ind < len(terrain_map)
    height_terrain_below = aircraft_height - measured_dist
    return stats.norm.pdf(height_terrain_below, loc = terrain_map[ind] + noise_mean, scale = noise_sd)

def get_path(my_node):
    if my_node.val == -1:
        return []
    assert my_node.parent != None
    my_path = get_path(my_node.parent)
    my_path = my_path + [my_node.val]
    return my_path

def write_csv(vec, filename = "particle_filter.csv"):
    with open(filename, 'w') as f:
        f.write('"","x"\n')
        for i in range(len(vec)):
            f.write('"{0}",{1}\n'.format(i + 1, vec[i]))

def main(resampling_size = 300):
    root_ = node(-1221)
    expansion_frontier = []
    # Initialization
    for i in range(-1, len(terrain_map)):
        current_node = node(i, root_)
        expansion_frontier.append(current_node)
    # expansion
    for measurement_cnt in tqdm(range(len(measurements))):
        possible_vel = np.arange(min_v, max_v + 1)
        next_candidates = []
        my_measurement = measurements[measurement_cnt]
        new_frontier = []
        for cur_node in expansion_frontier:
            temp = cur_node.val + possible_vel
            temp = temp[temp < len(terrain_map)]
            if not len(temp): continue
            likelihood_vec = np.array([calc_likelihood(my_measurement, i) for i in temp])
            next_ind = np.argmax(likelihood_vec)
            next_pos = temp[next_ind]
            next_likelihood = likelihood_vec[next_ind]
            new_node = node(next_pos, cur_node)
            new_node.likelihood += np.log(next_likelihood)
            new_node.prob = next_likelihood
            new_frontier.append(new_node)
        # Resampling
        if len(new_frontier) > resampling_size:
            prob_vec = np.array([i.prob for i in new_frontier])
            prob_vec = prob_vec / np.sum(prob_vec)
            expansion_frontier = np.random.choice(new_frontier,
                size = resampling_size,
                replace = False,
                p = prob_vec)
        else:
            expansion_frontier = new_frontier
    # End of expansion
    a_node = max(expansion_frontier, key = lambda x : x.likelihood)
    a_path = get_path(a_node)
    a_path = np.array(a_path) + 1 # to fix the different array index in R and in Python
    print("Head")
    print(a_path[:30])
    print("Tail")
    print(a_path[-30:])
    write_csv(a_path)

if __name__ == "__main__":
    main()