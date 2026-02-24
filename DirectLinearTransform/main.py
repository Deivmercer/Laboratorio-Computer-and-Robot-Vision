import numpy as np

# 20251015_142006
image_points = [(67, 1146),
                (82, 1216),
                (739, 807),
                (797, 593),
                (1116, 633),
                (1355, 177),
                (1368, 1000),
                (1431, 1373),
                (1842, 790)]
world_points = [
    (0, 0, 0),
    (0, 0, -2.7),
    (16.4, 15.2, 6.3),
    (11.5, 44, 8.4),
    (34.2, 29.7, 7.5),
    (43.5, 104, 30.5),
    (41.1, 4.2, 0),
    (43.5, 2.2, -10),
    (47, 24.2, 4.6)
]

M = []
for i in range(len(world_points)):
    X, Y, Z = world_points[i]
    u, v = image_points[i]
    M += [(-X, -Y, -Z, -1, 0, 0, 0, 0, u * X, u * Y, u * Z, u)] # axi
    M += [(0, 0, 0, 0, -X, -Y, -Z, -1, v * X, v * Y, v * Z, v)] # ayi

U, S, Vt = np.linalg.svd(M)

P = np.resize(Vt[-1, :], (3, 4))

H_inv = np.linalg.inv(P[:, 0:3]) # P = KR * [I_3 | -X_0 ] = [H | h]

R_inv, K_inv = np.linalg.qr(H_inv)  # H^-1 = R^-1 * K^-1
K = np.linalg.inv(K_inv)
K /= K[2, 2]  # Normalizing K
R = np.linalg.matrix_transpose(R_inv)
print(K)
