import cv2
import numpy as np
import os
import sys

class Stitcher:
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.images = self._load_images()
        self.matcher = Matchers()

    def _load_images(self):
        """Load all images from the folder and resize them for processing."""
        image_files = [f for f in os.listdir(self.folder_path) if f.endswith(('jpg', 'png', 'jpeg', '.JPG', '.PNG', '.JPEG'))]
        if not image_files:
            raise ValueError("No image files found in the folder.")
        images = [cv2.resize(cv2.imread(os.path.join(self.folder_path, img)), (480, 320)) for img in sorted(image_files)]
        print("Loaded {} images from {}".format(len(images), self.folder_path))
        return images

    def stitch(self):
        """Perform stitching on the loaded images."""
        if len(self.images) < 2:
            raise ValueError("Need at least two images to perform stitching.")

        self.left_image = self.images[0]
        for i in range(1, len(self.images)):
            H = self.matcher.match(self.left_image, self.images[i])
            if H is None:
                print("Skipping image {} due to matching failure.".format(i))
                continue

            dsize, offset = self._calculate_warp_size(H, self.left_image, self.images[i])
            tmp = cv2.warpPerspective(self.images[i], np.dot(offset, H), dsize)
            self.left_image = self._blend_images(self.left_image, tmp)

        return self.left_image

    def _calculate_warp_size(self, H, base_image, next_image):
        """Calculate the size of the warped image and its offset."""
        h, w = base_image.shape[:2]
        corners = np.array([[0, 0, 1], [w, 0, 1], [0, h, 1], [w, h, 1]])
        warped_corners = np.dot(H, corners.T).T
        warped_corners /= warped_corners[:, -1][:, np.newaxis]
        min_x, min_y = np.min(warped_corners, axis=0)[:2]
        max_x, max_y = np.max(warped_corners, axis=0)[:2]

        dsize = (int(max(max_x, w) - min(min_x, 0)), int(max(max_y, h) - min(min_y, 0)))
        offset = np.array([[1, 0, -min(min_x, 0)], [0, 1, -min(min_y, 0)], [0, 0, 1]])
        return dsize, offset

    def _blend_images(self, base_image, warped_image):
        """Blend two images together by averaging overlapping areas."""
        h1, w1 = base_image.shape[:2]
        h2, w2 = warped_image.shape[:2]
        h, w = max(h1, h2), max(w1, w2)

        blended_image = np.zeros((h, w, 3), dtype=np.uint8)
        blended_image[:h1, :w1] = base_image

        for y in range(h2):
            for x in range(w2):
                if np.any(warped_image[y, x]):
                    if np.all(blended_image[y, x] == 0):
                        blended_image[y, x] = warped_image[y, x]
                    else:
                        blended_image[y, x] = (
                            blended_image[y, x].astype(np.float32) * 0.5 +
                            warped_image[y, x].astype(np.float32) * 0.5
                        ).astype(np.uint8)

        return blended_image

class Matchers:
    def __init__(self):
        self.surf = cv2.xfeatures2d.SURF_create()
        index_params = dict(algorithm=0, trees=5)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)

    def match(self, image1, image2):
        """Find homography matrix between two images."""
        features1 = self._get_features(image1)
        features2 = self._get_features(image2)

        matches = self.flann.knnMatch(features2['des'], features1['des'], k=2)
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append((m.trainIdx, m.queryIdx))

        if len(good_matches) > 4:
            points1 = np.float32([features1['kp'][i].pt for i, _ in good_matches])
            points2 = np.float32([features2['kp'][i].pt for _, i in good_matches])
            H, _ = cv2.findHomography(points2, points1, cv2.RANSAC, 4)
            return H
        return None

    def _get_features(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.surf.detectAndCompute(gray, None)
        return {'kp': keypoints, 'des': descriptors}

if __name__ == '__main__':
    folders = ['data1', 'data2', 'data3', 'data4']

    for folder in folders:
        print("Processing folder: {}".format(folder))
        stitcher = Stitcher(folder)
        result = stitcher.stitch()

        output_file = folder + '_stitched.jpg'
        cv2.imwrite(output_file, result)
        print("Stitched image saved to {}".format(output_file))
        cv2.imshow("Stitched Image - {}".format(folder), result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
