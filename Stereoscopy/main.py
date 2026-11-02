import cv2
import matplotlib.pyplot as plt
import numpy as np

left_image = cv2.imread("images/view0.png")
right_image = cv2.imread("images/view1.png")
h, w, _ = left_image.shape

plt.figure()
plt.axis("off")
plt.subplot(121); plt.imshow(left_image[:, :, ::-1]); plt.title("Left image")
plt.subplot(122); plt.imshow(right_image[:, :, ::-1]); plt.title("Right image")
plt.show()

# 1. Rilevare feature nelle immagini
sift = cv2.SIFT.create()
left_keypoints = sift.detect(left_image)
right_keypoints = sift.detect(right_image)

# 2. Calcolare i descrittori delle feature
left_keypoints, left_descriptors = sift.compute(left_image, left_keypoints)
right_keypoints, right_descriptors = sift.compute(right_image, right_keypoints)

# 3. Trovare le associazioni tra feature delle due immagini
matcher = cv2.BFMatcher.create()
matches = matcher.knnMatch(left_descriptors, right_descriptors, k=2)
good_left_points = []
good_right_points = []
good_matches = []
for match in matches: # Lowe's ratio test
    m = match[0]
    n = match[1]
    if m.distance < 0.8 * n.distance:
        good_matches += [match]
        good_left_points += [left_keypoints[m.queryIdx].pt]
        good_right_points += [right_keypoints[m.trainIdx].pt]

# 4. (Opzionale) Mostrare le associazioni
drawn_matches = cv2.drawMatchesKnn(left_image, left_keypoints, right_image, right_keypoints, good_matches, None)
plt.figure(); plt.axis("off"); plt.imshow(drawn_matches[:, :, ::-1]); plt.title("Matches"); plt.show()

# 5. Stimare la matrice fondamentale
good_left_points = np.float32(good_left_points)
good_right_points = np.float32(good_right_points)
F, mask = cv2.findFundamentalMat(good_left_points, good_right_points, cv2.FM_LMEDS)

# 6. (Opzionale) La stima della matrice fondamentale riconoscerà automaticamente le associazioni non valide. Mostrare le
# associazioni valide
valid_left_points = []
valid_right_points = []
valid_matches = []
for i, m in enumerate(mask.ravel()):
    if i == 0:
        continue
    valid_match = good_matches[i]
    valid_matches += [valid_match]
    valid_left_points += [left_keypoints[valid_match[0].queryIdx].pt]
    valid_right_points += [right_keypoints[valid_match[0].trainIdx].pt]
drawn_valid_matches = cv2.drawMatchesKnn(left_image, left_keypoints, right_image, right_keypoints, valid_matches, None)
plt.figure(); plt.axis("off"); plt.imshow(drawn_valid_matches[:, :, ::-1]); plt.title("Valid matches"); plt.show()

# 7. Stimare le trasformazioni per rettificare le immagini
valid_left_points = np.float32(valid_left_points)
valid_right_points = np.float32(valid_right_points)
_, left_image_homography, right_image_homography = cv2.stereoRectifyUncalibrated(valid_left_points, valid_right_points, F, (w, h))

# 8. Rettificare le immagini
left_rectified = cv2.warpPerspective(left_image, left_image_homography, (w, h))
right_rectified = cv2.warpPerspective(right_image, right_image_homography, (w, h))
plt.figure()
plt.axis("off")
plt.subplot(121); plt.imshow(left_rectified[:, :, ::-1]); plt.title("Left rectified")
plt.subplot(122); plt.imshow(right_rectified[:, :, ::-1]); plt.title("Right rectified")
plt.show()

left_gray = cv2.cvtColor(left_rectified, cv2.COLOR_BGR2GRAY)
right_gray = cv2.cvtColor(right_rectified, cv2.COLOR_BGR2GRAY)
anaglyph = cv2.merge([right_gray, right_gray, left_gray])
plt.figure(); plt.axis("off"); plt.imshow(cv2.cvtColor(anaglyph, cv2.COLOR_BGR2RGB)); plt.title("Stereo Anaglyph"); plt.show()

# 9. Calcolare la disparity
stereo = cv2.StereoSGBM.create()
disparity = stereo.compute(left_rectified, right_rectified)
plt.figure(); plt.axis("off"); plt.imshow(disparity, cmap='gray'); plt.colorbar(label="Disparity (pixels)"); plt.title("Disparity"); plt.show()
