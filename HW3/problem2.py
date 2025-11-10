import cv2
import gtsam
import gtsam.symbol_shorthand as ss
from apriltag import apriltag
import numpy as np
from pathlib import Path


def get_camera_calibration():
    # Setup
    data_dir = "calibration_images"
    n_cols = 6
    n_rows = 8
    side_length = 0.01  # meters

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

    return K


def estimate_pose():
    image_path = "vslam/frame_0.jpg"

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    detector = apriltag("tag36h11")

    detections = detector.detect(img)
    tag0_points = detections[0]["lb-rb-rt-lt"]

    side_length = 0.01
    edge_to_cent = side_length / 2

    K = get_camera_calibration()
    K = gtsam.Cal3_S2(fx=K[0, 0], fy=K[1, 1], s=K[0, 1], u0=K[0, 2], v0=K[1, 2])

    lb3 = gtsam.Point3(-edge_to_cent, edge_to_cent, 0)
    rb3 = gtsam.Point3(edge_to_cent, edge_to_cent, 0)
    rt3 = gtsam.Point3(edge_to_cent, -edge_to_cent, 0)
    lt3 = gtsam.Point3(-edge_to_cent, -edge_to_cent, 0)

    points_3d = [lb3, rb3, rt3, lt3]
    # points_2d = [gtsam.Point2(x[0],x[1]) for x in tag0_points]

    graph = gtsam.NonlinearFactorGraph()
    init_values = gtsam.Values()
    constrained_noise = gtsam.noiseModel.Constrained.All(3)
    measurement_noise = gtsam.noiseModel.Isotropic.Sigma(2, 1)
    for i, (p_pt, cam_pt) in enumerate(zip(points_3d, tag0_points)):
        graph.add(gtsam.PriorFactorPoint3(ss.P(i), p_pt, constrained_noise))
        graph.add(
            gtsam.GenericProjectionFactorCal3_S2(
                cam_pt, measurement_noise, ss.X(0), ss.P(i), K
            )
        )
        init_values.insert(ss.P(i), p_pt)

    init_values.insert(ss.X(0), gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(0, 0, -0.05)))

    res = gtsam.LevenbergMarquardtOptimizer(graph, init_values).optimize()
    return res.atPose3(ss.X(0))
