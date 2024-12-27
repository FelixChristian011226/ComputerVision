import cv2
import numpy as np
import os
import sys

class Stitcher:
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.images = self._load_images()
        self.matcher = Matchers()
        self.left_list = []
        self.right_list = []
        self.center_image = None
        self._prepare_image_lists()

    def _load_images(self):
        """Load all images from the folder and resize them for processing."""
        image_files = [f for f in os.listdir(self.folder_path) if f.endswith(('jpg', 'png', 'jpeg', '.JPG', '.PNG', '.JPEG'))]
        if not image_files:
            raise ValueError("No image files found in the folder.")
        images = [cv2.resize(cv2.imread(os.path.join(self.folder_path, img)), (480, 320)) for img in sorted(image_files)]
        print("Loaded {} images from {}".format(len(images), self.folder_path))
        return images

    def stitch(self):
        self._shift_left()
        self._shift_right()
        return self.left_image

    def _prepare_image_lists(self):
        self.center_idx = len(self.images) // 2
        self.center_image = self.images[self.center_idx]
        self.left_list = self.images[:self.center_idx + 1]
        self.right_list = self.images[self.center_idx + 1:]

    def _shift_left(self):
        a = self.left_list[0]
        for b in self.left_list[1:]:
            H = self.matcher.match(a, b)
            if H is None:
                print("No homography could be computed.")
                continue

            xh = np.linalg.inv(H)
            ds = np.dot(xh, np.array([a.shape[1], a.shape[0], 1]))
            ds = ds / ds[-1]
            f1 = np.dot(xh, np.array([0, 0, 1]))
            f1 = f1 / f1[-1]
            xh[0, -1] += abs(f1[0])
            xh[1, -1] += abs(f1[1])
            ds = np.dot(xh, np.array([a.shape[1], a.shape[0], 1]))
            offsety = abs(int(f1[1]))
            offsetx = abs(int(f1[0]))
            dsize = (max(int(ds[0]) + offsetx, a.shape[1] + b.shape[1]),
                    max(int(ds[1]) + offsety, a.shape[0] + b.shape[0]))

            tmp = cv2.warpPerspective(a, xh, dsize)

            if tmp.shape[0] < offsety + b.shape[0] or tmp.shape[1] < offsetx + b.shape[1]:
                new_dsize = (max(tmp.shape[1], offsetx + b.shape[1]),
                            max(tmp.shape[0], offsety + b.shape[0]))
                expanded_tmp = np.zeros((new_dsize[1], new_dsize[0], 3), dtype=tmp.dtype)
                expanded_tmp[:tmp.shape[0], :tmp.shape[1]] = tmp
                tmp = expanded_tmp

            tmp[offsety:offsety + b.shape[0], offsetx:offsetx + b.shape[1]] = b
            a = tmp

        self.left_image = tmp

    def _shift_right(self):
        for next_image in self.right_list:
            H = self.matcher.match(self.left_image, next_image)
            if H is not None:
                dsize = self._calculate_warp_size_right(self.left_image, next_image, H)
                tmp = cv2.warpPerspective(next_image, H, dsize)
                self.left_image = self._blend_images(self.left_image, tmp)

    def _calculate_warp_size(self, image, H):
        ds = np.dot(H, np.array([image.shape[1], image.shape[0], 1]))
        ds /= ds[-1]
        offsetx = abs(int(np.dot(H, np.array([0, 0, 1]))[0]))
        offsety = abs(int(np.dot(H, np.array([0, 0, 1]))[1]))
        dsize = (int(ds[0]) + offsetx, int(ds[1]) + offsety)
        return dsize, offsetx, offsety

    def _calculate_warp_size_right(self, base_image, next_image, H):
        txyz = np.dot(H, np.array([next_image.shape[1], next_image.shape[0], 1]))
        txyz /= txyz[-1]
        return (int(txyz[0]) + base_image.shape[1], int(txyz[1]) + base_image.shape[0])

    def _blend_images(self, base_image, warped_image):
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
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
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
            points1 = np.float32([features1['kp'][i].pt for (i, _) in good_matches])
            points2 = np.float32([features2['kp'][i].pt for (_, i) in good_matches])
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
