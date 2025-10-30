import numpy as np
from numpy.linalg import norm
import plotly.graph_objects as go


def EstimateCorrespondences(X, Y, t, R, d_max):
    C = np.empty((0, 2), dtype=int)  # Unknown size
    X_trans = (R @ X.T).T + t
    for i, x_trans in enumerate(X_trans):
        # Find ind of closest value in y to current transformed pose
        j = np.argmin(norm(Y - x_trans, axis=1) ** 2)

        # Check min dist is met and add to output if it is
        if norm(Y[j] - x_trans) < d_max:
            C = np.vstack((C, [i, j]))

    return C


def ComputeOptimalRigidRegistration(X, Y, C):
    # Get useful values from C
    K = C.shape[0]
    i_list = C[:, 0]
    j_list = C[:, 1]

    # Get centroid
    x_mean = np.sum(X[i_list, :], axis=0) / K
    y_mean = np.sum(Y[j_list, :], axis=0) / K

    # Mean-zero
    Xp = X[i_list, :] - x_mean
    Yp = Y[j_list, :] - y_mean

    # Covariance
    W = (Xp.T @ Yp) / K

    # SVD
    U, Sigma, Vt = np.linalg.svd(W)

    # Build diagonal matrix
    tmp = np.eye(len(Sigma))
    tmp[-1, -1] = np.linalg.det(U @ Vt.T)

    # Find rotation
    R = U @ tmp @ Vt

    return y_mean - R @ x_mean, R


def ICP(X, Y, t0, R0, d_max, num_ICP_iters):
    t_hat = t0
    R_hat = R0

    for i in range(num_ICP_iters):
        C = EstimateCorrespondences(X, Y, t_hat, R_hat, d_max)
        t_hat, R_hat = ComputeOptimalRigidRegistration(X, Y, C)

    return t_hat, R_hat, C


if __name__ == "__main__":
    # Path to point clouds
    pclX_path = "pclX.txt"
    pclY_path = "pclY.txt"

    # Load
    X = np.loadtxt(pclX_path, delimiter=" ")
    Y = np.loadtxt(pclY_path, delimiter=" ")

    # Initialize
    t0 = np.zeros(3)
    R0 = np.eye(3)
    d_max = 0.25
    num_ICP_iters = 30

    # Run ICP
    t, R, C = ICP(X, Y, t0, R0, d_max, num_ICP_iters)

    # Extract values from C
    K = C.shape[0]
    i_list = C[:, 0]
    j_list = C[:, 1]

    # Calculate error
    RMSE = np.sqrt(
        (1 / K) * np.sum(norm(Y[j_list, :] - ((R @ X[i_list, :].T).T + t), axis=1) ** 2)
    )

    # Get transformed X points
    X_trans = (R @ X.T).T + t

    # Build figure using plotly because it makes a nice 3D plot
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=Y[:, 0],
                y=Y[:, 1],
                z=Y[:, 2],
                mode="markers",
                marker_size=2,
                marker_color="blue",
                name="Y points",
            ),
            go.Scatter3d(
                x=X_trans[:, 0],
                y=X_trans[:, 1],
                z=X_trans[:, 2],
                mode="markers",
                marker_size=2,
                marker_color="green",
                name="X points transformed",
            ),
            go.Scatter3d(
                x=X[:, 0],
                y=X[:, 1],
                z=X[:, 2],
                mode="markers",
                marker_size=2,
                marker_color="red",
                name="X points",
            ),
        ]
    )

    fig.update_layout(legend_itemwidth=200, legend_itemsizing="constant")

    fig.show()

    print(R)
    print(t)
    print(RMSE)
