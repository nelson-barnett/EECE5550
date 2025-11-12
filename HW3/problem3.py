from pathlib import Path
import gtsam
import gtsam.symbol_shorthand as sym
import dt_apriltags as apriltags
import cv2
import numpy as np


def get_camera_calibration(
    data_dir="calibration_images", n_cols=6, n_rows=8, side_length=0.01
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

    return K


def problem3(data_dir, K, tag_size=0.01):
    # Initialize detector and graph
    detector = apriltags.Detector("tag36h11")
    graph = gtsam.NonlinearFactorGraph()
    initial_values = gtsam.Values()
    constrained_noise = gtsam.noiseModel.Constrained.All(3)
    isotropic_noise = gtsam.noiseModel.Isotropic.Sigma(2, 1)

    seen_tags = dict()
    seen_tag_poses = dict()

    gtsam_K = gtsam.Cal3_S2(fx=K[0, 0], fy=K[1, 1], s=K[0, 1], u0=K[0, 2], v0=K[1, 2])

    side_length = 0.01
    edge_to_cent = side_length / 2
    lb3 = gtsam.Point3(-edge_to_cent, edge_to_cent, 0)
    rb3 = gtsam.Point3(edge_to_cent, edge_to_cent, 0)
    rt3 = gtsam.Point3(edge_to_cent, -edge_to_cent, 0)
    lt3 = gtsam.Point3(-edge_to_cent, -edge_to_cent, 0)

    points_3d = [lb3, rb3, rt3, lt3]

    # Set priors
    graph.add(gtsam.PriorFactorPoint3(sym.L(0), gtsam.Pose3(), constrained_noise))

    initial_values.insert(sym.X(0), gtsam.Pose3())

    for i, file in enumerate(Path(data_dir).iterdir()):
        # At each image (pose), find all the tags and relative poses
        img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        detections = detector.detect(
            img,
            estimate_tag_pose=True,
            camera_params=(K[0, 0], K[1, 1], K[0, 2], K[1, 2]),
            tag_size=tag_size,
        )

        for tag in detections:
            # At each tag (use tag_id)
            tag_id = tag.tag_id

            # Get current pose
            tag_pose = gtsam.Pose3(gtsam.Rot3(tag.pose_R), tag.pose_t)

            # Add the pose for this tag to container
            if tag_id not in seen_tag_poses:
                seen_tag_poses[tag_id] = {}
            seen_tag_poses[tag_id][i] = tag_pose

            # Check if this tag has been seen before,
            if tag_id in seen_tags:
                # If it has, add a BetweenFactor
                for pose_ind in seen_tags[tag_id]:
                    prev_pose = seen_tag_poses[tag_id][pose_ind]
                    # Inverse here is fine because the data type of Pose3 is known so inverse is fast (no transpose operator, also)
                    pose_transform = prev_pose.compose(tag_pose.inverse())
                    graph.add(
                        gtsam.BetweenFactorPose3(
                            sym.X(pose_ind), sym.X(i), pose_transform, isotropic_noise
                        )
                    )
                seen_tags[tag_id].append(i)
            else:
                # If it hasn't, associate it with this pose
                seen_tags[tag_id] = [i]

            # Either way, add it as an observation at this pose
            for j, corner in enumerate(tag.corners):
                graph.add(
                    gtsam.GenericProjectionFactorCal3_S2(
                        corner, isotropic_noise, sym.X(i), sym.L(tag_id + j), gtsam_K
                    )
                )


if __name__ == "__main__":
    K = get_camera_calibration()
    problem3("data_dir", K)
