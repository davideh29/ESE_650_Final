import numpy as np
import cv2
from os import listdir, rename
from os.path import isfile, join


def convert_binary(color_path, img_path):
    # Get file names
    img_files = [f for f in listdir(color_path) if isfile(join(color_path, f))]
    # Loop through images
    for i in range(0, len(img_files)):
        # Load segmentation image
        img = cv2.imread(color_path + img_files[i], 0)
        img[img != 204] = 255
        img[img == 204] = 0
        cv2.imwrite(img_path + img_files[i], img_path + img)
    return


def convert_numerical(img_path):
    img_files = [f for f in listdir(img_path) if isfile(join(img_path, f))]
    for i in range(0, len(img_files)):
        b = img_files[i].replace(".", "_")
        a = [s for s in b.split("_") if s.isdigit()]
        rename(img_path + img_files[i], a[0])
    return


def load_testing_data(normalized_frames_path, binary_masks_path):
    # Get file names
    # video_files = [int(f) for f in listdir(normalized_frames_path) if isfile(join(normalized_frames_path, f))]
    # bin_mask_files = [int(f) for f in listdir(binary_masks_path) if isfile(join(binary_masks_path, f))]
    video_files = [int(f.split(".")[0]) for f in listdir(normalized_frames_path) if
                   isfile(join(normalized_frames_path, f))]
    bin_mask_files = [int(f.split(".")[0]) for f in listdir(binary_masks_path) if isfile(join(binary_masks_path, f))]
    # Sort to make sure color images and masks line up
    video_files = np.sort(video_files)
    bin_mask_files = np.sort(bin_mask_files)
    # List of image red pixel values
    video_images = []
    bin_mask = []
    # Loop through images
    for i in range(0, len(video_files)):
        # Load color image
        video_images.append(cv2.imread(normalized_frames_path + format(video_files[i]) + '.tif', 1))
        # Load segmentation image
        bin_mask.append(cv2.imread(binary_masks_path + format(bin_mask_files[i]) + '.png', 0))
    # Convert to ndarray and return
    video_images = np.array(video_images)
    bin_mask = np.array(bin_mask)
    return video_images, bin_mask


def gen_features(video_img, bin_mask, min_area):
    # Output vector of features
    feat_vect = []
    contour_vect = []
    centroid_vect = []
    all_feats = []
    for i in range(0, len(video_img)):
        # Segmentation image
        bw_img = bin_mask[i]
        # Find contours
        ret, thresh = cv2.threshold(bw_img, 127, 255, 0)
        _, contours, _ = cv2.findContours(thresh, 1, 2)
        contours_final = []
        # Feature vector for current image
        img_feats = []
        centroid_contours = []
        for c in range(0, len(contours)):
            cnt = np.squeeze(contours[c])
            if cv2.contourArea(cnt) > min_area:
                img_feat, centroid = contour_feats(cnt)
                img_feats.append(img_feat)
                centroid_contours.append(centroid)
                contours_final.append(cnt)
        contour_vect.append(contours_final)
        feat_vect.append(img_feats)
        centroid_vect.append(np.vstack(centroid_contours))
        if i == 0:
            all_feats = img_feats
        else:
            all_feats = np.concatenate([all_feats, img_feats])
    # Normalize features
    for i in range(0, len(feat_vect)):
        # FEATURE SCALING to [0, 1]
        feat_vect_i = np.array(feat_vect[i])
        min_feats = np.tile(np.min(all_feats, axis=0), (feat_vect_i.shape[0], 1))
        max_feats = np.tile(np.max(all_feats, axis=0), (feat_vect_i.shape[0], 1))
        feat_vect[i] = np.divide(np.subtract(feat_vect_i, min_feats), np.subtract(max_feats, min_feats))
    mean_area = np.mean(np.vstack(feat_vect)[:, 2])
    return feat_vect, contour_vect, centroid_vect, mean_area


def contour_feats(contour):
    img_feat = np.empty(16)
    M = cv2.moments(contour)
    # Centroid
    centroid = np.array([M['m10'] / M['m00'], M['m01'] / M['m00']])
    img_feat[0] = centroid[0]
    img_feat[1] = centroid[1]
    # Area
    img_feat[2] = cv2.contourArea(contour)
    # Perimeter
    img_feat[3] = cv2.arcLength(contour, True)
    # Calculate distances from centroid and circularity measures
    dist = np.sum((contour - centroid) ** 2, axis=1)
    v11 = np.sum(np.prod(contour - centroid, axis=1))
    v02 = np.sum(np.square(contour - centroid)[:, 1])
    v20 = np.sum(np.square(contour - centroid)[:, 0])
    # Circularity
    m = 0.5 * (v02 + v20)
    n = 0.5 * np.sqrt(4 * v11 ** 2 + (v20 - v02) ** 2)
    img_feat[4] = (m - n) / (m + n)
    # Min/max distance
    img_feat[5] = dist.min()
    img_feat[6] = dist.max()
    # Mean distance
    img_feat[7] = dist.mean()
    img_feat[8] = dist.std()
    img_feat[9:16] = cv2.HuMoments(M).flatten()
    return img_feat, centroid
