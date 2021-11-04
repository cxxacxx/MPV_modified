import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
from Utils import Algo, Plot
from matplotlib import cm
matplotlib.use('Qt5Agg')

np.random.seed(2)


def generate_linear_states(F, R):
    state = np.zeros((100, 2))
    measurements = np.zeros((100, 2))
    state[0, 1] = 100
    measurements[0] = state[0] + np.random.multivariate_normal(mean=(0, 0), cov=R)
    for i in range(1, state.shape[0]):
        state[i] = np.matmul(F, state[i - 1])
        measurements[i] = state[i] + np.random.multivariate_normal(mean=(0, 0), cov=R)
    return state, measurements


def generate_phases_and_observations(F, R, Q, dt):
    state = np.zeros((100, 2))
    phase_cycle_vec = np.zeros(state.shape)
    measurements = np.zeros((state.shape[0], 1))
    state[0, 1] = 1
    for i in range(1, state.shape[0]):
        # state[i] = F @ state[i - 1] + np.random.multivariate_normal(mean=(0, 0), cov=0)
        state[i] = F @ state[i - 1]
    for i in range(state.shape[0]):
        phase_cycle_vec[i] = calc_q_q_int(state[i])
        if i > 0:
            phase_cycle_vec[i] += np.random.multivariate_normal(mean=(0, 0), cov=R)
        measurements[i, 0] = (1/(2 * np.pi)) * np.arctan2(phase_cycle_vec[i, 1], phase_cycle_vec[i, 0])
    measurements[:, 0] = utils.correct_measurement(measurements[:, 0], period=1)
    return state, phase_cycle_vec, measurements




def calc_q_q_int(x):
    p = x[0]
    return np.array([np.cos(2 * np.pi * p), np.sin(2 * np.pi * p)])


def calc_y_nominal(x):
    p = x[0]
    v = x[1]
    return np.array([np.tan(p), v / (np.cos(p) ** 2)])


def calc_H(x):
    p = x[0]
    v = x[1]
    H = np.array([[1 / (np.cos(p) ** 2 + 1e-4), 0],
                  [2 * np.tan(p) * v / (np.cos(p) ** 2 + 1e-4), 1 / (np.cos(p) ** 2 + 1e-4)]])
    return H


def plot_phase_cycle(phase_cycle_vec):
    plt.figure(figsize=(8, 4))
    plt.plot(phase_cycle_vec[:, 0], phase_cycle_vec[:, 1])
    plt.xlabel(r'Thigh angle (rad)', fontsize=14)
    plt.ylabel(r'Thigh angle integration $rad \cdot s$', fontsize=14)
    plt.axis('equal')


def plot_result(measurements, state, filter_result):
    plt.figure(figsize=(8, 4))
    plt.plot(range(0, len(measurements)), measurements[:, 0], label='Measurements')
    plt.plot(range(0, len(state)), state[:, 0], label='State')
    plt.plot(range(0, len(filter_result)), filter_result[:, 0], label='Kalman Filter')
    plt.legend()
    plt.xlabel('Time', fontsize=14)
    plt.ylabel('velocity [m]', fontsize=14)


def plot_error(measurements, state, filter_result):
    plt.figure(figsize=(8, 4))
    plt.plot(range(0, len(measurements)), measurements[:, 0] - state[:, 0], label='Measurements error')
    plt.plot(range(0, len(filter_result)), filter_result[:, 0] - state[:, 0], label='Kalman filter error')
    plt.legend()
    plt.xlabel('Time', fontsize=14)
    plt.ylabel('velocity [m]', fontsize=14)



if __name__ == '__main__':
    Plot.plot_kalman_filter()
