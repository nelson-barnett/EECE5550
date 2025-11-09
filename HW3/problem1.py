import cv2
import numpy as np
from pathlib import Path

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
x_pos = np.arange(side_length, side_length * (n_cols + 1), side_length)
y_pos = np.arange(side_length, side_length * (n_rows + 1), side_length)
xx, yy, zz = np.meshgrid(x_pos, y_pos, 0)

# Generic points for all images
p_base = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]

# Containers
U = []
P = []
for file in Path(data_dir).iterdir():
    # Read and convert to grayscale
    img = cv2.imread(file)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Get corners
    ret, corners = cv2.findChessboardCorners(img_gray, (n_rows, n_cols), None)

    if ret:
        # Add corners if they're detected
        U.append(corners)
        P.append(p_base)

# Get calibration matrix
_, mtx, _, _, _ = cv2.calibrateCamera(
    np.asarray(P, dtype=np.float32), U, img_gray.shape[::-1], None, None
)
