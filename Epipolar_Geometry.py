UBIT = '<asrathor>'
import numpy as np
np.random.seed(sum([ord(c) for c in UBIT]))
import cv2

# The two images are read.
img1 = cv2.imread('tsucuba_left.png')
img2 = cv2.imread('tsucuba_right.png')

# Both the read images are converted to grayscale.
img1_gray = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
img2_gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)


def epipolar_geometry():

    # Task 2.1: The SIFT features are extracted using openCV.
    sift = cv2.xfeatures2d.SIFT_create()
    img1_keypoints, des1 = sift.detectAndCompute(img1_gray,None)
    img2_keypoints, des2 = sift.detectAndCompute(img2_gray,None)

    # After the features are extracted, the keypoints are drawn on the grayscale images.
    new_img1 = cv2.drawKeypoints(img1_gray,img1_keypoints,img1,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    new_img2 = cv2.drawKeypoints(img2_gray,img2_keypoints,img2,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Both the new images with the keypoints are saved.
    cv2.imwrite('task2_sift1.jpg', new_img1)

    cv2.imwrite('task2_sift2.jpg', new_img2)

    # The keypoints are matched using Brute-Force Matcher with k-nn (k=2)
    bf = cv2.BFMatcher()
    all_matches = bf.knnMatch(des1,des2,k=2)

    # A threshold is applied to filter the good matches.
    good_matches = []
    for m, n in all_matches:
        if m.distance < 0.75*n.distance:
            good_matches.append(m)

    # The draw matches is used to create a new image showing matching between the keypoints on both the images.
    img3 = cv2.drawMatches(img1_gray, img1_keypoints, img2_gray, img2_keypoints, good_matches, None, flags=2)

    # The new image with all the matches (both outliers and inliers) is saved.
    cv2.imwrite('task2_matches_knn.jpg', img3)

    # ----------------------------------------------------------------------------------------

    # Task 2.2: In order to compute the fundamental matrix F, the coordinates of points in both the images
    # is computed and later fed into the findFundamentalMat function.
    img1_pts = np.float32([img1_keypoints[m.queryIdx].pt for m in good_matches])
    img2_pts = np.float32([img2_keypoints[m.trainIdx].pt for m in good_matches])

    # The fundamental matrix function from openCV
    # (https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html)
    # is used to get the matrix F and mask comprising positions of inliers and outliers.
    F, mask = cv2.findFundamentalMat(img1_pts,img2_pts,cv2.RANSAC)
    print(F)
    overall_points = mask.ravel().tolist()

    # ----------------------------------------------------------------------------------------

    # Task 2.3: Similar to Task 1, To find 10 random matches using inliers, we first need to
    # extract only the inliers points.
    # It is worth noting that the overall_points comprises of 0,1 values where 0
    # denotes the outlier and 1 denote the inlier position in good_matches array.
    inliers = np.zeros(len(overall_points))

    # For random extraction of 10 inliers, one can start from a random start
    # point in the overall_points array and check if value in overall_points match 1, indicating inlier.
    inliers_n = []
    while len(inliers_n) != 10:
        start_point = np.random.randint(1, len(overall_points))
        if overall_points[start_point] == 1:
            inliers_n.append(start_point)
            inliers[start_point] = 1

    # In order to get only the inliers points, the randomized positions in inliners array where value is 1
    # is extracted from overall image points.
    img1_pts = img1_pts[inliers == 1]
    img2_pts = img2_pts[inliers == 1]
    print(img1_pts.shape)

    # To compute the epilines in an image for points in another image, computeCorrespondEpilines function
    # was used. (https://docs.opencv.org/2.4/modules/features2d/doc/drawing_function_of_keypoints_and_matches.html)
    lines1 = cv2.computeCorrespondEpilines(img1_pts,1,F)
    lines1 = lines1.reshape(-1,3)

    height, width = img2_gray.shape

    # The image is converted back to BGR since the line will show only white otherwise.
    # The original image global variable cannot be used since it is now comprised of all keypoints which we
    # don't want to show.
    img1_new = cv2.cvtColor(img2_gray,cv2.COLOR_GRAY2BGR)
    color =[]
    for i in range(10):
        color.append(tuple(np.random.randint(0,255,3).tolist()))

    # To draw the lines and corresponding points, referred openCV doc:https://docs.opencv.org/3.2.0/da/de9/tutorial_py_epipolar_geometry.html
    for height,i,j in zip(lines1,img2_pts,range(0,9)):
            x0, y0 = [0, int(-height[2] / height[1])]
            x1, y1 = [width, int(-(height[2] + height[0] * width) / height[1])]
            img1_new = cv2.line(img1_new, (x0, y0), (x1, y1), color[j], 1)
            img1_new = cv2.circle(img1_new, tuple(i), 4, color[j])
    cv2.imwrite('task2_epi_right.jpg',img1_new)

    lines2 = cv2.computeCorrespondEpilines(img2_pts, 2, F)
    lines2 = lines2.reshape(-1, 3)

    height, width = img1_gray.shape
    img2_new = cv2.cvtColor(img1_gray,cv2.COLOR_GRAY2BGR)
    for height,i,j in zip(lines2,img1_pts,range(0,9)):
            x0, y0 = [0, int(-height[2] / height[1])]
            x1, y1 = [width, int(-(height[2] + height[0] * width) / height[1])]
            img2_new = cv2.line(img2_new, (x0, y0), (x1, y1), color[j], 1)
            img2_new = cv2.circle(img2_new, tuple(i), 4, color[j])
    cv2.imwrite('task2_epi_left.jpg',img2_new)


def disparity():

    # The parameters were defined based on many tries.
    stereo = cv2.StereoSGBM_create()
    stereo.setMinDisparity(16)
    stereo.setNumDisparities(112-16-16)
    stereo.setBlockSize(15)
    stereo.setSpeckleRange(32)
    stereo.setSpeckleWindowSize(100)
    stereo.setUniquenessRatio(10)
    stereo.setDisp12MaxDiff(1)
    disparity = stereo.compute(img1_gray, img2_gray).astype(np.float32)/16.0
    cv2.imwrite('task2_disparity.jpg',disparity)

# For running task 2.1,2.2,2.3
epipolar_geometry()
# For running task 2.4
disparity()