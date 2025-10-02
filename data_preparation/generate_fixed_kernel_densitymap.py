import numpy as np
from scipy.ndimage import gaussian_filter
from sklearn.neighbors import NearestNeighbors
import os
import cv2
import json
from tqdm import tqdm
from joblib import Parallel, delayed

# ---------------------------
# Default hyperparameters
# ---------------------------
K_NEIGHBORS = 4   # was 3
BETA = 0.2        # was 0.3
N_JOBS = -1        # parallel workers
# ---------------------------

def generate_adaptive_kernel_densitymap(image, points, k=K_NEIGHBORS, beta=BETA):
    h, w = image.shape[:2]
    densitymap = np.zeros((h, w), dtype=np.float32)

    if len(points) == 0:
        return densitymap

    pts = np.array(points)

    # Adjust kernel parameters based on point count
    point_count = len(pts)

    # Lower k and beta for sparse scenes
    if point_count <= 30:
        k = 2
        beta = 0.15
        min_sigma = 0.3
    elif point_count <= 100:
        k = 3
        beta = 0.2
        min_sigma = 0.5
    else:
        k = 4
        beta = 0.3
        min_sigma = 1.0

    if point_count == 1:
        # Single point — fixed small Gaussian
        x, y = int(pts[0][0]), int(pts[0][1])
        if 0 <= x < w and 0 <= y < h:
            temp = np.zeros((h, w), dtype=np.float32)
            temp[y, x] = 1.0
            densitymap = gaussian_filter(temp, sigma=1.2, mode='constant')
        return densitymap

    tree = NearestNeighbors(
        n_neighbors=min(k+1, len(pts)), algorithm='kd_tree'
    ).fit(pts)
    distances, _ = tree.kneighbors(pts)

    # mean distance to k nearest neighbors
    if len(pts) > k:
        avg_dists = np.mean(distances[:, 1:k+1], axis=1)
    else:
        avg_dists = np.mean(distances[:, 1:], axis=1)

    # compute sigma per point
    sigmas = beta * avg_dists
    sigmas[sigmas < min_sigma] = min_sigma  # adaptive min

    for i, (x, y) in enumerate(pts):
        if x < 0 or x >= w or y < 0 or y >= h:
            continue
        temp = np.zeros((h, w), dtype=np.float32)
        temp[int(y), int(x)] = 1.0
        densitymap += gaussian_filter(temp, sigmas[i], mode='constant')

    # Normalize to match original count
    if densitymap.sum() > 0:
        densitymap = densitymap / densitymap.sum() * len(points)

    return densitymap


def load_via_points(json_path, image_filename):
    with open(json_path, 'r') as f:
        data = json.load(f)
    for key, value in data.items():
        if value['filename'] == image_filename:
            points = []
            regions = value['regions']
            if isinstance(regions, dict):
                region_iter = regions.values()
            elif isinstance(regions, list):
                region_iter = regions
            else:
                region_iter = []
            for region in region_iter:
                shape = region.get('shape_attributes', {})
                if 'cx' in shape and 'cy' in shape:
                    points.append([shape['cx'], shape['cy']])
            return points
    return []


def process_image(image_file, images_dir, densitymaps_dir, json_path):
    image_path = os.path.join(images_dir, image_file)
    image = cv2.imread(image_path)
    if image is None:
        print(f"⚠️ Skipping unreadable file: {image_file}")
        return
    points = load_via_points(json_path, image_file)
    densitymap = generate_adaptive_kernel_densitymap(image, points)
    save_name = os.path.splitext(image_file)[0] + ".npy"
    np.save(os.path.join(densitymaps_dir, save_name), densitymap)


if __name__ == '__main__':
    for phase in ['train', 'test']:
        images_dir = f'../data/{phase}_data/images/'
        densitymaps_dir = f'../data/{phase}_data/densitymaps/'
        json_path = '../data/via_export_json.json'

        if not os.path.exists(densitymaps_dir):
            os.makedirs(densitymaps_dir)

        if not os.path.exists(images_dir):
            print(f'⚠️ Warning: {images_dir} does not exist. Skipping.')
            continue

        image_file_list = os.listdir(images_dir)

        # run in parallel
        Parallel(n_jobs=N_JOBS)(
            delayed(process_image)(img, images_dir, densitymaps_dir, json_path)
            for img in tqdm(image_file_list, desc=f"Processing {phase}")
        )

        print(f"✅ Adaptive density maps generated for {phase}_data.")