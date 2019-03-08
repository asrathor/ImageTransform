UBIT = '<asrathor>'
import numpy as np
np.random.seed(sum([ord(c) for c in UBIT]))
import cv2

# The two images are read.
img1 = cv2.imread('mountain1.jpg')
img2 = cv2.imread('mountain2.jpg')

# Both the read images are converted to grayscale.
img1_gray = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
img2_gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)


def main_function():

    # Task 1.1: The SIFT features are extracted using openCV.
    sift = cv2.xfeatures2d.SIFT_create()
    img1_keypoints, des1 = sift.detectAndCompute(img1_gray,None)
    img2_keypoints, des2 = sift.detectAndCompute(img2_gray,None)

    # After the features are extracted, the keypoints are drawn on the grayscale images.
    new_img1 = cv2.drawKeypoints(img1_gray,img1_keypoints,img1,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    new_img2 = cv2.drawKeypoints(img2_gray,img2_keypoints,img2,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Both the new images with the keypoints are saved.
    cv2.imwrite('task1_sift1.jpg',new_img1)

    cv2.imwrite('task1_sift2.jpg',new_img2)

    # ----------------------------------------------------------------------------------------

    # Task 1.2: The keypoints are matched using Brute-Force Matcher with k-nn (k=2)
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
    cv2.imwrite('task1_matches_knn.jpg',img3)

    # ----------------------------------------------------------------------------------------

    # Task 1.3: In order to compute the homography matrix H, the coordinates of points in both the images
    # is computed and later fed into the homography function.
    # Referred openCV doc: https://docs.opencv.org/3.4/dc/dc3/tutorial_py_matcher.html
    # It is necessary to use float32 otherwise an error, related to format, results in the findHomography function.
    img1_pts = np.float32([img1_keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    img2_pts = np.float32([img2_keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # The homography function from openCV
    # (https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html)
    # is used to get the matrix H and mask comprising positions of inliers and outliers.
    H, mask = cv2.findHomography(img1_pts,img2_pts,cv2.RANSAC)
    print(H)
    overall_points = mask.ravel().tolist()

    # ----------------------------------------------------------------------------------------
    # Task 1.4: To find 10 random matches using inliers, we first need to extract only the inliers points.
    # It is worth noting that the overall_points comprises of 0,1 values where 0
    # denotes the outlier and 1 denote the inlier position in good_matches array.
    inliers = np.zeros(len(overall_points))

    # For random extraction of 10 inliers, one can start from a random start
    # point in the overall_points array and get 10 successive positions of 1s which
    # indicate inliers.
    inliers_n = []
    while len(inliers_n) != 10:
        start_point = np.random.randint(1, len(overall_points))
        if overall_points[start_point] == 1:
            inliers_n.append(start_point)
            inliers[start_point] = 1

    # Tried to use random colors, however, sometimes the colors are barely visible.
    # color = tuple(np.random.randint(0, 255, 3).tolist())

    # print(len(overall_points))

    # The draw params argument can specify the color, mask determining which matches
    # are drawn (https://docs.opencv.org/2.4/modules/features2d/doc/drawing_function_of_keypoints_and_matches.html).
    draw_params = dict(matchColor=(199,21,133), singlePointColor=None, matchesMask=inliers, flags=2)
    img4 = cv2.drawMatches(img1_gray, img1_keypoints, img2_gray, img2_keypoints, good_matches, None, **draw_params)

    # The resulting image is saved.
    cv2.imwrite('task1_matches.jpg',img4)

    # ----------------------------------------------------------------------------------------

    # Task 1.5: We first create an array with extreme values depicting size of the current images.
    # It is necessary to use float32 otherwise an error, related to format, results in the
    # perspectiveTransform function.
    ext_arr = np.float32([[0, 0], [0, img2_gray.shape[0]], [img2_gray.shape[1], img2_gray.shape[0]], [img2_gray.shape[1], 0]]).reshape(-1, 1, 2)
    print(ext_arr)
    print('----------')

    # Then we compute the perspective transformation with previously computed homography matrix H.
    transformed_coor = cv2.perspectiveTransform(ext_arr, H)
    print('-----------')
    print(transformed_coor)

    # In order to get the size of new warped image, we need to first compute the max and min of current image size
    # transformed matrix. The first column of the matrix represents the width of the image while second column
    # represents the height of the image (this was actually perceived from seeing the printed values of matrix
    # and coming up with this assumption).

    min_width = -(np.int16(transformed_coor.min(axis=0)[0,0])-1)
    min_height = -(np.int16(transformed_coor.min(axis=0)[0,1])-1)
    max_width = np.int16(ext_arr.max(axis=0)[0, 0])
    max_height = np.int16(ext_arr.max(axis=0)[0, 1])
    print(min_width)
    print(min_height)
    print(max_width)
    print(max_height)

    # The homography matrix needs to be recomputed for the transformated coordinates.
    temp_arr = np.zeros((3,3))
    temp_arr[0][0] = 1
    temp_arr[1][1] = 1
    temp_arr[2][2] = 1
    temp_arr[0][2] = min_width
    temp_arr[1][2] = min_height
    new_H = temp_arr.dot(H)

    # The new height and width of desired wrapped image is calculated.
    wrap_width = max_width + min_width
    wrap_height = max_height + min_height

    # Finally, the first image is wrapped in the overall desired image using new homography matrix.
    wrap_img = cv2.warpPerspective(img1_gray, new_H, (wrap_width, wrap_height))

    # The second image is added to the overall desired image at the defined location.
    wrap_img[min_height: img1_gray.shape[0] + min_height, min_width: img1_gray.shape[1] + min_width] = img2_gray

    cv2.imwrite('task1_pano.jpg',wrap_img)

# For running whole task1
main_function()
