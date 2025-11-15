from pathlib import Path
import gtsam
from matplotlib import pyplot as plt
import gtsam.symbol_shorthand as sym
import dt_apriltags as apriltags
import cv2
import numpy as np
from argparse import ArgumentParser


# Problem 1:
def get_camera_calibration(
    data_dir="calibration_images",
    n_cols=6,
    n_rows=8,
    side_length=0.01,
    print_result=False,
):
    # Set top left corner = (0,0,0)
    # +x axis points to the right
    # +y axis points down
    # +z axis points out of page

    # Define 3D points using above coordinate system
    x_pos = np.arange(side_length, side_length * (n_rows + 1), side_length)
    y_pos = np.arange(side_length, side_length * (n_cols + 1), side_length)
    xx, yy, zz = np.meshgrid(x_pos, y_pos, 0)

    # Generic points for all images
    p_base = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]

    # Containers
    U = []
    P = []
    for file in Path(data_dir).iterdir():
        # Read and convert to grayscale
        img_gray = cv2.imread(file, cv2.IMREAD_GRAYSCALE)

        # Get corners
        ret, corners = cv2.findChessboardCorners(img_gray, (n_rows, n_cols), None)

        if ret:
            # Add corners if they're detected
            U.append(corners)
            P.append(p_base)

    # Get calibration matrix
    _, K, _, _, _ = cv2.calibrateCamera(
        np.asarray(P, dtype=np.float32), U, img_gray.shape[::-1], None, None
    )

    if print_result:
        print("Camera calibration matrix:")
        print(K)

    return K


# Problem 2
def estimate_pose(K, image_path="vslam/frame_0.jpg", tag_type="tag36h11", side_length=0.01):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    detector = apriltags.Detector(tag_type)
    detections = detector.detect(img)
    tag0_points = detections[0].corners

    edge_to_cent = side_length / 2

    K_gtsam = gtsam.Cal3_S2(fx=K[0, 0], fy=K[1, 1], s=K[0, 1], u0=K[0, 2], v0=K[1, 2])

    lb3 = gtsam.Point3(-edge_to_cent, edge_to_cent, 0)
    rb3 = gtsam.Point3(edge_to_cent, edge_to_cent, 0)
    rt3 = gtsam.Point3(edge_to_cent, -edge_to_cent, 0)
    lt3 = gtsam.Point3(-edge_to_cent, -edge_to_cent, 0)

    points_3d = [lb3, rb3, rt3, lt3]
    points_2d = [gtsam.Point2(x[0], x[1]) for x in tag0_points]

    # Generate noise models
    constrained_noise = gtsam.noiseModel.Constrained.All(3)
    measurement_noise = gtsam.noiseModel.Isotropic.Sigma(2, 1)
    prior_noise = gtsam.noiseModel.Isotropic.Sigma(6, 1)

    # Initialize nonlinear factor graph
    graph = gtsam.NonlinearFactorGraph()

    # Initialize initial values of the graph
    init_values = gtsam.Values()

    # Add prior for first pose (X(0)) -- believe both frames are aligned at t = 0
    graph.add(
        gtsam.PriorFactorPose3(
            sym.X(0), gtsam.Pose3(gtsam.Rot3(), np.array([0, 0, 0.25])), prior_noise
        )
    )

    # Make 4 measurements at t = 1, (all therefore correspond to the same pose -- X(1))
    # For each measurement:
    # 1. Add a PriorFactor of the 3D point w.r.t AprilTag coordinate frame and known side length
    # 2. Add a factor for the reprojection error given K.
    # ---- This is error in measurement (cam_pt) given the corresponding point (in PriorFactor at P(i))
    # 3. Add the 3D point to the initial values
    for i, (p_pt, cam_pt) in enumerate(zip(points_3d, points_2d), 1):
        graph.add(gtsam.PriorFactorPoint3(sym.P(i), p_pt, constrained_noise))
        graph.add(
            gtsam.GenericProjectionFactorCal3_S2(
                cam_pt, measurement_noise, sym.X(1), sym.P(i), K_gtsam
            )
        )
        init_values.insert(sym.P(i), p_pt)

    # Add initial estimates for the two poses (X(0) with no measurements and X(1) with measurements)
    init_values.insert(sym.X(0), gtsam.Pose3())
    init_values.insert(sym.X(1), gtsam.Pose3(gtsam.Rot3(), np.array([0, 0, -0.05])))

    # Generate optimizer and print value at time point with measurements
    res = gtsam.LevenbergMarquardtOptimizer(graph, init_values).optimize()
    print("Pose:")
    print(res.atPose3(sym.X(1)))


# Problem 3
def run_vslam(data_dir, K, tag_size=0.01, tag_type="tag36h11"):
    detector = apriltags.Detector(tag_type)

    graph = gtsam.NonlinearFactorGraph()
    initial_values = gtsam.Values()
    constrained_noise = gtsam.noiseModel.Constrained.All(6)
    isotropic_noise = gtsam.noiseModel.Isotropic.Sigma(6, 1)

    seen_tags = dict()

    # Set first pose
    graph.add(gtsam.PriorFactorPose3(sym.L(0), gtsam.Pose3(), constrained_noise))

    for i, file in enumerate(Path(data_dir).iterdir()):
        # At each image (pose), find all the tags and relative poses
        img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        detections = detector.detect(
            img,
            estimate_tag_pose=True,
            camera_params=(K[0, 0], K[1, 1], K[0, 2], K[1, 2]),
            tag_size=tag_size,
        )

        initial_values.insert(
            sym.X(i), gtsam.Pose3(gtsam.Rot3(), np.array([0, 0, -0.05]))
        )
        for tag in detections:
            this_id = tag.tag_id
            this_pose = gtsam.Pose3(gtsam.Rot3(tag.pose_R), tag.pose_t)
            graph.add(
                gtsam.BetweenFactorPose3(
                    sym.X(i), sym.L(this_id), this_pose, isotropic_noise
                )
            )

            # Check if the tag has been seen before and make it an initial value if it's not already
            if this_id in seen_tags and not initial_values.exists(sym.L(this_id)):
                initial_values.insert(sym.L(this_id), this_pose)
            else:
                seen_tags[this_id] = this_pose

    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_values)
    res = optimizer.optimize()

    #### Plotting ####
    # Collect values for faster plotting
    tag_positions = np.zeros((len(seen_tags), 3))
    for tag_num in seen_tags:
        tag_pos = res.atPose3(sym.L(tag_num))
        tag_positions[tag_num] = tag_pos.translation()

    cam_positions = np.zeros((i, 3))
    for frame in range(i):
        cam_pos = res.atPose3(sym.X(frame))
        cam_positions[frame] = cam_pos.translation()

    # generate fig
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.scatter(
        tag_positions[:, 0],
        tag_positions[:, 1],
        tag_positions[:, 2],
        c="red",
        marker="s",
        depthshade=False,
    )
    ax.scatter(
        cam_positions[:, 0],
        cam_positions[:, 1],
        cam_positions[:, 2],
        c="blue",
        depthshade=False,
    )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    plt.show()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-p1", "--problem1", action="store_true")
    parser.add_argument("-p2", "--problem2", action="store_true")
    parser.add_argument("-p3", "--problem3", action="store_true")
    args = parser.parse_args()
    
    if not args.problem1 and not args.problem2 and not args.problem3:
        print("No arguments given.")
    else:
        K = get_camera_calibration()
    
    if args.problem1:
        print("Camera calibration matrix:")
        print(K)
        
    if args.problem2:
        estimate_pose(K)
    
    if args.problem3:
        run_vslam("vslam", K)
