import cv2
import numpy as np
import glob

def get_homography(world_points, image_points):
    M = []
    for i in range(len(world_points)):
        X, Y, _ = world_points[i] # Ignoring Z axis
        u, v = image_points[i]
        M += [[-X, -Y, -1, 0, 0, 0, u * X, u * Y, u]]  # axi
        M += [[0, 0, 0, -X, -Y, -1, v * X, v * Y, v]]  # ayi

    _, _, Vt = np.linalg.svd(M)

    H = np.resize(Vt[-1, :], (3, 3))
    H /= H[2, 2]
    return H

def get_vij(H, i, j):
    return [H[0, i] * H[0, j],
            H[0, i] * H[1, j] + H[1, i] * H[0, j],
            H[1, i] * H[1, j],
            H[2, i] * H[0, j] + H[0, i] * H[2, j],
            H[2, i] * H[1, j] + H[1, i] * H[2, j],
            H[2, i] * H[2, j]]


chessboard_size = (9, 6)
world_points = []
for j in range(chessboard_size[1]):
    for i in range(chessboard_size[0]):
        world_points += [[i, j, 0]]

V = []
images = glob.glob("images/*.jpg")
for image_path in images:
    image = cv2.imread(image_path)
    found, corners = cv2.findChessboardCorners(image, chessboard_size, None)
    if not found:
        print("Can't find the chessboard in the image " + str(image_path))
        continue

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

    H = get_homography(world_points, np.resize(corners2, [len(corners2), 2]).tolist())

    v12 = get_vij(H, 0, 1)
    v11 = get_vij(H, 0, 0)
    v22 = get_vij(H, 1, 1)

    V += [v12, np.subtract(v11, v22).tolist()]

U, S, Vt = np.linalg.svd(V) # Vb = 0

b = Vt[-1, :]
b11, b12, b22, b13, b23, b33 = b
B = [[b11, b12, b13],
     [b12, b22, b23],
     [b13, b23, b33]]

try:
    A = np.linalg.cholesky(B) # A * A^T = B
except np.linalg.LinAlgError: # Cholesky failed, likely because B is not positive-definite
    B = -np.array(B)
    A = np.linalg.cholesky(B)

K = np.linalg.inv(A.T) # A = K^-T
K /= K[2, 2]
print(K)
