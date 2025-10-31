import argparse
import numpy as np
from scipy.linalg import expm
from matplotlib import pyplot as plt


def particle_filter_prop(t1, Xt1, phidot_l, phidot_r, t2, r, w, sigma_l, sigma_r):
    # Initialize
    rng = np.random.default_rng()
    X_tnext = []
    # For each particle in Xt1 (list)
    for xi in Xt1:
        # Compute new noise for each velocity
        noise_l = rng.normal(0, sigma_l)
        noise_r = rng.normal(0, sigma_r)

        # Propagate model
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

        # TS -> G
        X_tnext.append(xi @ expm((t2 - t1) * omega_dot))

    return X_tnext


def particle_filter_update(Xt, zt, sigma_p):
    # Initialize
    rng = np.random.default_rng()
    W = np.zeros(len(Xt))
    # For each particle in Xt
    for i, xt in enumerate(Xt):
        # Get weight of measurement zt given particle
        res = zt - xt[:-1, -1]
        W[i] = (1 / (2 * np.pi * (sigma_p**2))) * np.exp(
            (-1 / (2 * sigma_p**2)) * (res.T.dot(res))
        )

    # Sample particles with replacement from Xt given weights
    return rng.choice(Xt, len(Xt), p=W / sum(W))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", action="store_true")
    parser.add_argument("-f", action="store_true")
    parser.add_argument("-g", action="store_true")
    args = parser.parse_args()

    # Settings
    phidot_l = 1.5
    phidot_r = 2.0
    r = 0.25
    w = 0.5
    sigma_l = 0.05
    sigma_r = 0.05
    sigma_p = 0.1

    # This is same for (e), (f), (g)
    N = 1000
    x0 = np.eye(3)
    X0 = [x0 for _ in range(N)]

    # Same for (f) and (g)
    times = [0, 5, 10, 15, 20]
    colors = ["green", "blue", "purple", "cyan"]

    ####### (e) #######
    if args.e:
        # Run
        X10 = particle_filter_prop(
            0, X0, phidot_l, phidot_r, 10, r, w, sigma_l, sigma_r
        )
        X10_position = np.asarray([x[:-1, -1] for x in X10])
        position_mean = X10_position.mean(axis=0)
        position_cov = np.cov(X10_position.T)

        print("--e--")
        print("position mean: ", position_mean)
        print("position cov: ", position_cov)

        # Plot
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.scatter(X10_position[:, 0], X10_position[:, 1], s=1, c="blue")
        ax.scatter(position_mean[0], position_mean[1], s=5, c="red", label="mean")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Part (e): Scatterplot at t = 10")
        ax.legend(markerscale=2)

        block = False if args.f or args.g else True
        plt.show(block=block)

    ####### (f) #######
    if args.f:
        # Initialize
        fig = plt.figure()
        ax = fig.add_subplot()
        means = np.zeros((len(times) - 1, 2))
        covs = []

        # Run
        Xt = X0
        for i, t in enumerate(times[:-1]):
            # Get next pose
            Xt = particle_filter_prop(
                t, Xt, phidot_l, phidot_r, times[i + 1], r, w, sigma_l, sigma_r
            )
            # Extract position and stats
            Xt_position = np.asarray([x[:-1, -1] for x in Xt])
            position_mean = Xt_position.mean(axis=0)
            position_cov = np.cov(Xt_position.T)

            # Store
            means[i] = position_mean
            covs.append(position_cov)

            # Add to plot
            ax.scatter(
                Xt_position[:, 0],
                Xt_position[:, 1],
                s=1,
                c=colors[i],
                label=f"t = {times[i + 1]}",
            )
            ax.scatter(position_mean[0], position_mean[1], s=5, c="red")

        print("--(f)--")
        print("means:")
        [print(f"t = {times[i + 1]}: {m}") for i, m in enumerate(means)]
        print("\ncovs:")
        [print(f"t = {times[i + 1]}: {c}") for i, c in enumerate(covs)]

        # Display
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Part (f): Dead reckoning")
        ax.legend(markerscale=5)
        block = False if args.g else True
        plt.show(block=block)

    ####### (g) #######
    if args.g:
        # Setup
        measurements = np.array(
            [[1.6561, 1.2847], [1.0505, 3.1059], [-0.9875, 3.2118], [-1.645, 1.1978]]
        )

        # Initialize
        fig = plt.figure()
        ax = fig.add_subplot()
        means = np.zeros((len(times) - 1, 2))
        covs = []

        # Run
        Xt = X0
        for i, t in enumerate(times[:-1]):
            # Propagate
            Xt = particle_filter_prop(
                t, Xt, phidot_l, phidot_r, times[i + 1], r, w, sigma_l, sigma_r
            )

            # Update
            Xt = particle_filter_update(Xt, measurements[i], sigma_p)

            # Get position and stats
            Xt_position = np.asarray([x[:-1, -1] for x in Xt])
            position_mean = Xt_position.mean(axis=0)
            position_cov = np.cov(Xt_position.T)

            # Store
            means[i] = position_mean
            covs.append(position_cov)

            # Add to plot
            ax.scatter(
                Xt_position[:, 0],
                Xt_position[:, 1],
                s=1,
                c=colors[i],
                label=f"t = {times[i + 1]}",
            )
            ax.scatter(position_mean[0], position_mean[1], s=5, c="red")

        print("--(g)--")
        print("means:")
        [print(f"t = {times[i + 1]}: {m}") for i, m in enumerate(means)]
        print("\ncovs:")
        [print(f"t = {times[i + 1]}: {c}") for i, c in enumerate(covs)]

        # Display
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Part (g): Using filtered estimates from measurements")
        ax.legend(markerscale=5)
        plt.show()
