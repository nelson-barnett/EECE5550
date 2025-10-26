import numpy as np
from numpy.linalg import norm
from numpy.random import multivariate_normal


def state_transition_model(p, v, dt, R):
    return p + dt * v + multivariate_normal([0, 0], R)


def measurement_model(p, L, Q):
    # L is known points
    return np.array([norm(p - L[0]), norm(p - L[1])]) + multivariate_normal([0, 0], Q)


def state_prop(p, v, dt, Sigma, R):
    mu_hat = state_transition_model(p, v, dt, R)
    Sigma_hat = Sigma + R
    return multivariate_normal(mu_hat, Sigma_hat), mu_hat, Sigma_hat


def measurement_update(p, v, dt, L, Sigma, R, Q):
    _, mu_hat, Sigma_hat = state_prop(p, v, dt, Sigma, R)

    H = np.array(
        [
            [(p[0] - L[0][0]) / norm(p - L[0]), (p[1] - L[0][1]) / norm(p - L[0])],
            [(p[0] - L[1][0]) / norm(p - L[1]), (p[1] - L[1][1]) / norm(p - L[1])],
        ]
    )

    K = Sigma_hat @ H.T @ np.linalg.inv(H @ Sigma_hat @ H.T + Q)

    h = np.array([norm(mu_hat - L[0]), norm(mu_hat - L[1])])
    mu = mu_hat + K * (measurement_model(p, L, Q) - h)
    Sigma = (np.eye(2) - K @ H) @ Sigma_hat

    return mu, Sigma


if __name__ == "__main__":
    # Setup
    L = np.array([[5, 5], [-5, 5]])

    R = 0.1 * np.eye(2)
    Q = 0.5 * np.eye(2)

    p0 = multivariate_normal(np.zeros(2), np.eye(2))
    dt = 0.5

    v = np.array([1, 0])

    pnext = state_transition_model(p0, v, dt, R)

    # Run
