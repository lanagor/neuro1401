import math
import numpy as np
import matplotlib.pyplot as plt
import os 
from scipy import stats
import seaborn as sns

params = {"M": 100, "grid_size": 50, "T": 100, "sample_size": 100, "num_samples": 100, "alpha": None,
			"ro": 0.5, "locations": 8, "num_trials": 1000, "array_sizes": [1, 2, 4, 8], "gamma": None}
sample_points = np.linspace(-math.pi, math.pi, num=100, endpoint=False)

# randomly generates a sampling of preferred orientations for i neurons 
# representing j locations
# evenly chooses 100 positions in range [-pi, pi)
def gen_phi(params):
	phi = np.empty((params["M"], params["locations"]))
	possible_orientations = np.linspace(-math.pi, math.pi, params["M"], endpoint=False)
	for j in range(params["locations"]):
		np.random.shuffle(possible_orientations)
		for i in range(params["M"]):
			phi[i][j] = possible_orientations[i]
	return phi

# generates alpha grid of weights of stimuli: 1 if stimulus, 0 if not
def gen_alpha(params, num_cued):
	return np.concatenate((np.ones(num_cued), np.zeros(params["locations"] - num_cued)))

# creates array of actual orientation values
def gen_theta(size):
	return np.random.uniform(-math.pi, math.pi, size=size)	

# generates logarithmically spaced grid of ro and eta values, as in experiment 1
def gen_ro_eta_grid(params): 
	out = np.empty((params["grid_size"], params["grid_size"], 2))
	ro_array = np.logspace(-4, 2, params["grid_size"], base=2.0)
	eta_array = np.logspace(0, 6, params["grid_size"], base=2.0)
	for i in range(params["grid_size"]):
		ro = ro_array[i]
		for j in range(params["grid_size"]):
			out[i][j][0] = ro
			out[i][j][1] = eta_array[j]
	return out

# mean activaton (eq. 3.5)
def mean_activation(ro): 
	f_ro =  np.i0(1. / ro).item() * np.exp(1. / ro).item()
	return f_ro

# driving input to ith neuron econding stimulus at jth location
def feed_forward(theta, ro, phi, params, j):
	driving_input = np.empty(params["M"])
	for i in range(params["M"]):
		# Commented out line below because it was computing wrong expression
		driving_input[i] = (math.cos(phi[i][j] - theta[j]) - 1) / ro 
		# driving_input[i] = (math.cos(phi[i][j] - theta[j])) / ro - 1
	return np.exp(driving_input) 

# firing rate of a neuron as a function of jth theta value
def firing_rate(theta, ro, phi, j, params): 
	firing_rate = np.empty((params["M"], params["locations"]))
	f = feed_forward(theta, ro, phi, params, j)

	M = params["M"]
	alpha = params["alpha"]
	gamma = params["gamma"]

	f_ro = mean_activation(ro)
	sum_alpha = np.sum(alpha)
	for i in range(params["M"]):
		firing_rate[i][j] = gamma * (1. / params["alpha"]) * (f[i] / (M * f_ro))
	return firing_rate

# generates spikes by sampling from equation 5
# returns phi_i, where i is neuron with max spikes
def gen_epsilon(phi, theta, j, params):
	spike_max = -float('inf')
	argmax = None
	ro = params["ro"]
	r = firing_rate(theta, ro, phi, j, params)
	for i in range(params["M"]):
		spike = np.random.poisson(r[i][j] * params["T"])
		if spike > spike_max: 
			spike_max = spike
			argmax = i
	phi_i = phi[argmax][j]
	return circle_subtract(phi_i, theta[j])

def circle_subtract(theta1, theta2):
	difference = theta1 - theta2
	return difference - (difference / (2 * math.pi))

# finds delta theta/ recall error
def calc_error(theta_j, phi, j, params, sample_points):
	eta = params["eta"]
	m = np.random.poisson(eta)
	max_sum = -float('inf')
	argmax = None

	for theta in range(len(sample_points)):
		summation = 0
		for i in range(m):
			epsilon = gen_epsilon(phi, theta_j, j, params)
			summation += math.cos(sample_points[theta] - epsilon)
		if summation > max_sum:
			max_sum = summation
			argmax = theta
	return argmax 

# runs experiment 1 as described in simulation and model fitting
def run_experiment_1(params, phi):
	# matrix for storing number of errors of each size: initialized to 0
	output = np.zeros((len(params["array_sizes"]), params["sample_size"]))
	# params["gamma"] = np.random.randint(20, 640)
	params['gamma'] = 20
	# print("Gamma: {}".format(params['gamma']))

	# generates grid with all possible combos of ro/eta values ('simulation and model building', pg 3643)
	ro_eta_grid = gen_ro_eta_grid(params)

	r, e = (np.random.randint(0, params["grid_size"]) - 1, np.random.randint(0, params["grid_size"]) - 1)
	params["ro"], params["eta"] = ro_eta_grid[r][e][0],  ro_eta_grid[r][e][1]
	print("Gamma: {}, Rho: {}".format(params['gamma'], params['ro']))

	# discrete, uniformly distributed set of points in range [-pi, pi)
	sample_points = np.linspace(-math.pi, math.pi, params["sample_size"], endpoint=False)

	# experiment for 1, 2, 4, and 8 possible stimuli
	for pos in range(len(params["array_sizes"])):
		num_pos = params["array_sizes"][pos]

		# orientation of stimuli
		theta = gen_theta(num_pos)

		# salience of position: to start, 1 if stimulus and 0 if not
		params["alpha"] = num_pos
		# print(params["alpha"])
		# runs 100 trials 
		for i in range(params["num_trials"]):

			# choose a location to test on
			j = max(np.random.randint(0, params["array_sizes"][pos] + 1) - 1, 0)

			# sample from error distribution
			delta_theta = calc_error(theta, phi, j, params, sample_points)

			output[pos][delta_theta] += 1

			if i % 10 == 0:

				print("Trial: {}, Number of positions: {}".format(i, num_pos))
		# print(params)
		print("Output for {} positions: {}".format(num_pos, output[pos, :]))
	return output

if __name__ ==  "__main__":
	phi = gen_phi(params)
	results = run_experiment_1(params, phi)	

	# Since experiment takes a while to run, saving results locally (with option to load for plotting)
	np.save(os.getcwd() + "/results.npy", results)
	imported_results = np.load(os.getcwd() + "/results.npy")
	sample_points = np.linspace(-math.pi, math.pi, params["sample_size"], endpoint=False)
	variance, kurtosis = np.empty((len(params["array_sizes"]),)), np.empty((len(params["array_sizes"]),))
	for position in range(len(params['array_sizes'])):
		data = imported_results[position]
		kurtosis[position] = stats.kurtosis(data)
		variance[position] = np.var(data)

		print(data)
		plt.plot(sample_points, data)
		# sns.kdeplot(imported_results[position])
		plt.title("Error Distribution for {} Stimuli".format(params['array_sizes'][position]))
		plt.show()
		# FIX HISTOGRAM (CURRENTLY ONLY PLOTTING FOR POSITIVE VALUES ON X-AXIS)
		plt.hist(imported_results[position], bins=50, range=(-np.pi, np.pi))
		plt.title("Histogram of Errors for {} Stimuli".format(params['array_sizes'][position]))
		plt.show()
	# plt.scatter(params["array_sizes"], variance)
	# plt.xlabel("Items")
	# plt.ylabel("variance")
	# plt.show()
	# plt.scatter(params["array_sizes"], kurtosis)
	# plt.xlabel("Items")
	# plt.ylabel("kurtosis")
	# plt.show()
