import numpy as np
from scipy.linalg import expm
from matplotlib import pyplot as plt


def particle_filter_prop(t1, Xt1, phidot_l, phidot_r, t2, r, w, sigma_l, sigma_r):
    rng = np.random.default_rng()

    X_tnext = []
    for xi in Xt1:
        noise_l = rng.normal(0, sigma_l**2)
        noise_r = rng.normal(0, sigma_r**2)

        omega_dot = np.array(
            [
                [
                    0,
                    -(r / w) * (phidot_r + noise_r - (phidot_l + noise_l)),
                    (r / 2) * (phidot_r + noise_r + phidot_l + noise_l),
                ],
                [(r / w) * (phidot_r + noise_r - (phidot_l + noise_l)), 0, 0],
                [0, 0, 0],
            ]
        )

        X_tnext.append(xi @ expm((t2 - t1) * omega_dot))

    return X_tnext


def particle_filter_update(Xt, zt, sigma_p):
    X_bar_t = []
    for i, xt in enumerate(Xt):
        res = zt - xt[:-1, -1]
        w = (1 / (2 * np.pi * (sigma_p**2))) * np.exp(
            (-(sigma_p**2) / 2) * (res.T.dot(res))
        )

        # This is probably wrong:
        X_bar_t.append(w * xt)

    return X_bar_t


if __name__ == "__main__":
    phidot_l = 1.5
    phidot_r = 2.0
    r = 0.25
    w = 0.5
    sigma_l = 0.05
    sigma_r = 0.05
    sigma_p = 0.1

    # (e)
    N = 1000
    x0 = np.eye(3)
    X0 = [x0 for _ in range(N)]
    X10 = particle_filter_prop(0, X0, phidot_l, phidot_r, 10, r, w, sigma_l, sigma_r)

    X10_position = np.asarray([x[:-1, -1] for x in X10])
    position_mean = X10_position.mean(axis=0)
    position_cov = np.cov(X10_position.T)

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.scatter(X10_position[:, 0], X10_position[:, 1], s=1, c="blue")
    ax.scatter(position_mean[0], position_mean[1], s=5, c="red")
    ax.set_title(
        f"Scatterplot at t = 10.\nMean = {position_mean}, cov = {position_cov}"
    )
    plt.show(block=False)

    # (f) 1 (this may not be what is meant)
    times = [5, 10, 15, 20]
    colors = ["green", "blue", "purple", "cyan"]

    fig = plt.figure()
    ax = fig.add_subplot()
    means = np.zeros((len(times), 2))
    covs = []
    for i, t in enumerate(times):
        Xt = particle_filter_prop(0, X0, phidot_l, phidot_r, t, r, w, sigma_l, sigma_r)
        Xt_position = np.asarray([x[:-1, -1] for x in Xt])
        position_mean = Xt_position.mean(axis=0)
        position_cov = np.cov(Xt_position.T)

        means[i] = position_mean
        covs.append(position_cov)

        ax.scatter(
            Xt_position[:, 0], Xt_position[:, 1], s=1, c=colors[i], label=f"t = {t}"
        )
        ax.scatter(position_mean[0], position_mean[1], s=5, c="red")

    ax.legend(loc="upper left", bbox_to_anchor=(1, 1), markerscale=3)
    plt.show()

    # (f) 2 (this is probably what is meant)
    times = [0, 5, 10, 15, 20]
    colors = ["green", "blue", "purple", "cyan"]

    fig = plt.figure()
    ax = fig.add_subplot()
    means = np.zeros((len(times), 2))
    covs = []
    Xt = X0
    for i, t in enumerate(times[:-1]):
        Xt = particle_filter_prop(
            t, Xt, phidot_l, phidot_r, times[i + 1], r, w, sigma_l, sigma_r
        )
        Xt_position = np.asarray([x[:-1, -1] for x in Xt])
        position_mean = Xt_position.mean(axis=0)
        position_cov = np.cov(Xt_position.T)

        means[i] = position_mean
        covs.append(position_cov)

        ax.scatter(
            Xt_position[:, 0], Xt_position[:, 1], s=1, c=colors[i], label=f"t = {t}"
        )
        ax.scatter(position_mean[0], position_mean[1], s=5, c="red")

    ax.legend(loc="upper left", bbox_to_anchor=(1, 1), markerscale=3)
    plt.show()

    # (g)
    measurements = np.array(
        [[1.6561, 1.2847], [1.0505, 3.1059], [-0.9875, 3.2118], [-1.645, 1.1978]]
    )
    fig = plt.figure()
    ax = fig.add_subplot()
    means = np.zeros((len(times), 2))
    covs = []
    Xt = X0
    for i, t in enumerate(times[:-1]):
        Xt = particle_filter_prop(
            t, Xt, phidot_l, phidot_r, times[i + 1], r, w, sigma_l, sigma_r
        )
        
        Xt = particle_filter_update(Xt, measurements[i], sigma_p)
        
        Xt_position = np.asarray([x[:-1, -1] for x in Xt])
        position_mean = Xt_position.mean(axis=0)
        position_cov = np.cov(Xt_position.T)

        means[i] = position_mean
        covs.append(position_cov)

        ax.scatter(
            Xt_position[:, 0], Xt_position[:, 1], s=1, c=colors[i], label=f"t = {t}"
        )
        ax.scatter(position_mean[0], position_mean[1], s=5, c="red")

    ax.legend(loc="upper left", bbox_to_anchor=(1, 1), markerscale=3)
    plt.show()
