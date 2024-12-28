import cv2
import numpy as np
import random

def calculate_ransac_iterations(inlier_ratio, success_probability, min_samples=4):
    """Calculate the number of RANSAC iterations needed to ensure success_probability."""
    return int(np.log(1 - success_probability) / np.log(1 - inlier_ratio**min_samples))

def sift_descriptor(image):
    """Extract SIFT features and descriptors."""
    sift = cv2.xfeatures2d.SIFT_create()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    return keypoints, descriptors

def pixel_descriptor(image, patch_size=8):
    """Extract descriptors by concatenating pixel values around keypoints."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    keypoints = cv2.goodFeaturesToTrack(gray, maxCorners=500, qualityLevel=0.01, minDistance=10)
    keypoints = [cv2.KeyPoint(x=p[0][0], y=p[0][1], _size=patch_size) for p in keypoints]

    descriptors = []
    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        x1, y1 = max(0, x - patch_size // 2), max(0, y - patch_size // 2)
        x2, y2 = min(w, x + patch_size // 2), min(h, y + patch_size // 2)
        patch = gray[y1:y2, x1:x2].flatten()
        if len(patch) == patch_size * patch_size:
            descriptors.append(patch)
    descriptors = np.array(descriptors, dtype=np.float32)
    return keypoints, descriptors

def match_descriptors(desc1, desc2, ratio=0.7):
    """Match descriptors using FLANN-based matcher."""
    index_params = dict(algorithm=1, trees=5)  # FLANN_INDEX_KDTREE = 1
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(desc1, desc2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < ratio * n.distance:
            good_matches.append(m)

    return good_matches

def estimate_homography(kp1, kp2, matches):
    """Estimate the homography matrix using RANSAC."""
    if len(matches) < 4:
        return None, 0

    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 4.0)
    inliers = mask.sum() if mask is not None else 0

    return H, inliers

def compare_descriptors(image1, image2, descriptor_func1, descriptor_func2, ransac_tests=100):
    """Compare two descriptor methods using RANSAC iterations and inlier ratio."""
    kp1_1, desc1_1 = descriptor_func1(image1)
    kp1_2, desc1_2 = descriptor_func2(image1)

    kp2_1, desc2_1 = descriptor_func1(image2)
    kp2_2, desc2_2 = descriptor_func2(image2)

    # Match descriptors
    matches1 = match_descriptors(desc1_1, desc2_1)
    matches2 = match_descriptors(desc1_2, desc2_2)

    print("Matches 1: {}".format(len(matches1)))
    print("Matches 2: {}".format(len(matches2)))

    # Visualize the matches
    img_matches_1 = cv2.drawMatches(image1, kp1_1, image2, kp2_1, matches1, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    img_matches_2 = cv2.drawMatches(image1, kp1_2, image2, kp2_2, matches2, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Display the matches
    cv2.imshow("Matches 1 (SIFT)", img_matches_1)
    cv2.imshow("Matches 2 (Pixel Value)", img_matches_2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Evaluate with RANSAC
    inlier_ratios_1 = []
    inlier_ratios_2 = []

    for _ in range(ransac_tests):
        H1, inliers1 = estimate_homography(kp1_1, kp2_1, random.sample(matches1, min(len(matches1), 10)))
        H2, inliers2 = estimate_homography(kp1_2, kp2_2, random.sample(matches2, min(len(matches2), 10)))

        inlier_ratios_1.append(float(inliers1) / len(matches1) if matches1 else 0)
        inlier_ratios_2.append(float(inliers2) / len(matches2) if matches2 else 0)

    avg_inlier_ratio_1 = np.mean(inlier_ratios_1)
    avg_inlier_ratio_2 = np.mean(inlier_ratios_2)

    if avg_inlier_ratio_1 <= 0:
        print("Error: avg_inlier_ratio_1 is too low. Check the matching results.")
        return None
    if avg_inlier_ratio_2 <= 0:
        print("Error: avg_inlier_ratio_2 is too low. Check the matching results.")
        return None

    iterations_1 = calculate_ransac_iterations(avg_inlier_ratio_1, 0.5)
    iterations_2 = calculate_ransac_iterations(avg_inlier_ratio_2, 0.5)

    return {
        "descriptor1": {
            "average_inlier_ratio": avg_inlier_ratio_1,
            "ransac_iterations": iterations_1
        },
        "descriptor2": {
            "average_inlier_ratio": avg_inlier_ratio_2,
            "ransac_iterations": iterations_2
        }
    }

if __name__ == "__main__":
    image1 = cv2.imread("./data1/112_1298.JPG")
    image2 = cv2.imread("./data1/112_1299.JPG")

    # image1 = cv2.resize(image1, (480, 320))
    # image2 = cv2.resize(image2, (480, 320))

    results = compare_descriptors(image1, image2, sift_descriptor, pixel_descriptor)

    if results is not None:
        print("Comparison Results:")
        print("SIFT Descriptor:")
        print("  Average Inlier Ratio: {:.2f}".format(results["descriptor1"]["average_inlier_ratio"]))
        print("  RANSAC Iterations: {}".format(results["descriptor1"]["ransac_iterations"]))

        print("Pixel Value Descriptor:")
        print("  Average Inlier Ratio: {:.2f}".format(results["descriptor2"]["average_inlier_ratio"]))
        print("  RANSAC Iterations: {}".format(results["descriptor2"]["ransac_iterations"]))
    else:
        print("RANSAC comparison failed due to insufficient inliers.")
