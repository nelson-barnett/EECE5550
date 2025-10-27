import numpy as np
from matplotlib import pyplot as plt
from numpy.linalg import norm
from numpy.random import multivariate_normal


def update_state(mu_t, u_t, dt, sigma, R):
    mu_t1 = mu_t + dt * u_t
    return mu_t1 + multivariate_normal([0, 0], R), mu_t1, sigma + R


def take_measurement(p, L, Q=None):
    delta = 0 if Q is None else multivariate_normal([0, 0], Q)
    return np.array([norm(p - L[0]), norm(p - L[1])]) + delta


def measurement_jacobian(p, L):
    return np.array(
        [
            [(p[0] - L[0][0]) / norm(p - L[0]), (p[1] - L[0][1]) / norm(p - L[0])],
            [(p[0] - L[1][0]) / norm(p - L[1]), (p[1] - L[1][1]) / norm(p - L[1])],
        ]
    )


def kalman_gain(sigma, H, Q):
    return sigma @ H.T @ np.linalg.inv(H @ sigma @ H.T + Q)


if __name__ == "__main__":
    # Setup/given values
    L = np.array([[5, 5], [-5, 5]])
    R = 0.1 * np.eye(2)
    Q = 0.5 * np.eye(2)
    dt = 0.5
    t_min = 0
    t_max = 40
    time = np.arange(t_min, t_max + dt, dt)
    vels = np.array([[1, 0], [0, -1], [-1, 0], [0, 1]])

    # Pre-allocate
    true_locs = np.zeros((len(time), 2))
    loc_means = np.zeros((len(time), 2))
    loc_stds = np.zeros((len(time), 2))

    # Initialize
    mu_t = np.zeros(2)
    sigma_t = np.eye(2)

    for i, t in enumerate(time):
        # Set vel
        if 0 <= t <= 10:
            v_t = vels[0]
        elif 10 < t <= 20:
            v_t = vels[1]
        elif 20 < t <= 30:
            v_t = vels[2]
        else:
            v_t = vels[3]

        # Prediction
        x_t1, mu_t1, sigma_t1 = update_state(mu_t, v_t, dt, sigma_t, R)

        # Measurement
        z_t1 = take_measurement(x_t1, L, Q)

        # Update
        H = measurement_jacobian(mu_t1, L)
        K = kalman_gain(sigma_t1, H, Q)

        mu_t = mu_t1 + K @ (z_t1 - take_measurement(mu_t1, L))
        sigma_t = (np.eye(len(mu_t1)) - K @ H) @ sigma_t1

        # Store
        true_locs[i] = x_t1
        loc_means[i] = mu_t
        loc_stds[i] = np.sqrt(np.diag(sigma_t))

    # Plot
    fig = plt.figure(figsize=(13, 13))
    ax = fig.add_subplot()
    ax.errorbar(
        loc_means[:, 0],
        loc_means[:, 1],
        xerr=3 * loc_stds[:, 0],
        yerr=3 * loc_stds[:, 1],
        label="3*sigma estimate",
        alpha=0.6,
        markersize=2,
        capsize=3,
        elinewidth=1,
    )
    ax.plot(true_locs[:, 0], true_locs[:, 1], "-o", label="True Traj", alpha=0.6)
    ax.plot(loc_means[:, 0], loc_means[:, 1], "-o", label="Kalman Traj", alpha=0.6)
    ax.scatter(L[:, 0], L[:, 1], s=20, label="Landmarks", c="red")
    ax.legend()
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.3)
    plt.show()
