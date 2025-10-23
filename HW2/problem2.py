import numpy as np
from numpy.linalg import norm
from matplotlib import pyplot as plt


def EstimateCorrespondences(X, Y, t, R, d_max):
    C = []  # Unknown size
    # C = np.empty((0,2))  # Unknown size
    # Assuming rows are the points
    for i, x in enumerate(X):
        # Calculate size of diff separately
        res = norm(Y - (R.dot(x) + t), axis=1)
        
        # Get index
        j = np.argmin(res**2)

        # Add index
        if res[j] < d_max:
            # Drop Y[i,:] from the array??
            # int(j) ?
            # C = np.vstack((C, [i,j]))
            
            # If C is list, just append a tuple
            C.append((i, j))

    return C


def ComputeOptimalRigidRegistration(X, Y, C):
    # If C is matrix
    # K = C.shape[0]
    # i_list = C[:,0]
    # j_list = C[:,1]
    
    # If C is list of tuples
    K = len(C)
    i_list = [i for i, _ in C]
    j_list = [j for _, j in C]

    x_mean = np.sum(X[i_list, :], axis=0) / K
    y_mean = np.sum(Y[j_list, :], axis=0) / K

    Xp = X[i_list, :] - x_mean
    Yp = Y[j_list, :] - y_mean

    W = (Xp.T @ Yp) / K

    U, Sigma, Vt = np.linalg.svd(W)

    tmp = np.eye(len(Sigma))
    tmp[-1, -1] = np.linalg.det(U @ Vt.T)

    R = U @ tmp @ Vt

    return (y_mean - R @ x_mean, R)


def ICP(X, Y, t0, R0, d_max, num_ICP_iters):
    t_hat = t0
    R_hat = R0

    for i in range(num_ICP_iters):
        C = EstimateCorrespondences(X, Y, t_hat, R_hat, d_max)
        t_hat, R_hat = ComputeOptimalRigidRegistration(X, Y, C)

    return (t_hat, R_hat, C)


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

    t, R, C = ICP(X, Y, t0, R0, d_max, num_ICP_iters)

    # If C is array
    # K = C.shape[0]
    # i_list = C[:,0]
    # j_list = C[:,1]
    
    # If C is list of tuples
    K = len(C)
    i_list = [i for i, _ in C]
    j_list = [j for _, j in C]

    RMSE = np.sqrt((1 / K) * (norm(Y[j_list, :] - ((R @ X[i_list, :].T).T + t)) ** 2))

    # Plotting
    X_trans = (R @ X.T).T + t

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.scatter(Y[:, 0], Y[:, 1], Y[:, 2], c="red", alpha=0.7)
    ax.scatter(X_trans[:, 0], X_trans[:, 1], X_trans[:, 2], c="green", alpha=0.7)
