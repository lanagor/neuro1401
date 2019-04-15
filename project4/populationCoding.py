import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import stats
import seaborn as sns

params = {
    "M": 100,
    "grid_size": 50,
    "T": 100,
    "num_samples": 100,
    "alpha": None,
    "rho": 0.5,
    "locations": 8,
    "num_trials": 1000,
    "array_sizes": [1, 2, 4, 8],
    "gamma": None
}


# firing rate of a neuron as a function of jth theta value
def firing_rate(theta, phi, j, params):
    irho = 1. / params["rho"]
    f = np.exp((np.cos(phi[:, j] - theta[j]) - 1) * irho)
    # alpha = params["alpha"]
    # sum_alpha = np.sum(alpha)
    f_rho = np.i0(irho) * np.exp(irho)
    denom = f_rho * params["M"] * params["alpha"]
    return params["gamma"] * f / denom


def calc_error(theta_j, phi, j, params, sample_points):
    m = np.random.poisson(params["eta"])
    rt = firing_rate(theta_j, phi, j, params) * params["T"]

    # generates spikes by sampling from equation 5
    rand = np.random.poisson(rt, size=(m, params["num_samples"], len(rt))).argmax(axis=2).T
    diff = phi[rand, j] - theta_j[j]
    epsilon = diff - (diff / (2 * np.pi))
    return np.sum(np.cos(sample_points[:, None] - epsilon), axis=1)


# runs experiment 1 as described in simulation and model fitting
def run_experiment_1(params, phi):
    # matrix for storing number of errors of each size: initialized to 0
    output = np.zeros((len(params["array_sizes"]), params["num_samples"]))
    # params["gamma"] = np.random.randint(20, 640)
    params['gamma'] = 20
    # print("Gamma: {}".format(params['gamma']))

    # generates grid with all possible combos of ro/eta values ('simulation and model building', pg 3643)
    rho = np.random.choice(np.logspace(-4, 2, params["grid_size"], base=2.0))
    eta = np.random.choice(np.logspace(0, 6, params["grid_size"], base=2.0))
    params["rho"], params["eta"] = rho, eta
    params["rho"] = 0.52

    print("Gamma: {}, Rho: {}".format(params['gamma'], params['rho']))

    # discrete, uniformly distributed set of points in range [-pi, pi)
    points = np.linspace(-np.pi, np.pi, params["num_samples"], endpoint=False)

    # experiment for 1, 2, 4, and 8 possible stimuli
    for pos in range(len(params["array_sizes"])):
        # salience of position: to start, 1 if stimulus and 0 if not
        num_pos = params["array_sizes"][pos]
        params["alpha"] = num_pos

        # orientation of stimuli
        theta = np.random.uniform(-np.pi, np.pi, size=params["alpha"])

        # runs 100 trials
        for i in range(params["num_trials"]):

            # choose a location to test on
            j = np.random.randint(params["array_sizes"][pos])

            # sample from error distribution
            delta_theta = calc_error(theta, phi, j, params, points)

            output[pos, delta_theta.argmax()] += 1

            if i % 100 == 0:
                print("Trial: {}, Number of positions: {}".format(i, num_pos))
        # print(params)
        print("Output for {} positions: {}".format(num_pos, output[pos, :]))
    return output


if __name__ == "__main__":
    # generates Mxlocations preferred orientations from [-pi, pi)
    np.random.seed(5)
    orientations = np.linspace(-np.pi, np.pi, params["M"], endpoint=False)
    perms = [np.random.permutation(orientations) for j in range(params['locations'])]
    phi = np.stack(perms).T

    results = run_experiment_1(params, phi)

    # Since experiment takes a while to run, saving results locally (with option to load for plotting)
    np.save(os.getcwd() + "/results.npy", results)
    results = np.load(os.getcwd() + "/results.npy")
    sample_points = np.linspace(-np.pi, np.pi, params["num_samples"], endpoint=False)

    kurtosis = stats.kurtosis(results, axis=1)
    variance = np.var(results, axis=1)

    maxval = np.max(results)

    print(results)
    fig, axs = plt.subplots(1, len(params['array_sizes']), sharey=True)
    for i, (size, data) in enumerate(zip(params['array_sizes'], results)):
        ax = axs[i]
        ax.plot(sample_points, data)
        ax.set_title("Error Distribution for {} Stimuli".format(size))
    plt.show()

    fig, axs = plt.subplots(1, len(params['array_sizes']), sharey=True)
    for i, (size, data) in enumerate(zip(params['array_sizes'], results)):
        ax = axs[i]
        # sns.kdeplot(imported_results[position])
        # FIX HISTOGRAM (CURRENTLY ONLY PLOTTING FOR POSITIVE VALUES ON X-AXIS)
        ax.hist(data, bins=50, range=(-np.pi, np.pi))
        ax.set_title("Histogram of Errors for {} Stimuli".format(size))
    plt.show()
