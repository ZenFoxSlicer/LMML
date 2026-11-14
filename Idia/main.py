"""Find copies of the same photograph; write image_a,image_b to output.csv.

Requires NumPy and OpenCV (opencv-python or opencv-python-headless).
Accepts an optional ZIP/directory argument, or discovers input in the working
folder. No network, model downloads, training labels, or fixed image names.
"""

import os

# Avoid excessive nested BLAS/OpenCV parallelism on CPU-only evaluators.
for _variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")

import csv
import hashlib
import sys
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
WORKERS = max(1, min(4, os.cpu_count() or 1))


def discover_input():
    if len(sys.argv) > 1:
        return Path(sys.argv[1])
    for name in ("test-input.zip", "input.zip", "test.zip",
                 "test-input", "input", "test"):
        path = Path(name)
        if path.exists():
            return path
    archives = sorted(Path('.').glob('*.zip'))
    if len(archives) == 1:
        return archives[0]
    if len(archives) > 1:
        raise ValueError("Specify the input ZIP as the first argument.")
    for name in ("train-input", "train", "images"):
        if Path(name).is_dir():
            return Path(name)
    return Path('.')


def normalized(array):
    result = array.astype(np.float32).ravel()
    result -= result.mean()
    result /= max(float(np.linalg.norm(result)), 1e-8)
    return result


def image_features(record):
    name, content = record
    image = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Cannot decode image: " + name)
    height, width = image.shape
    if max(height, width) > 480:
        factor = 480 / max(height, width)
        image = cv2.resize(image, (max(1, round(width * factor)),
                                  max(1, round(height * factor))),
                           interpolation=cv2.INTER_AREA)
    keypoints, descriptors = cv2.SIFT_create(
        nfeatures=450, contrastThreshold=0.025
    ).detectAndCompute(image, None)
    points = np.array([point.pt for point in keypoints], np.float32).reshape(-1, 2)
    if descriptors is None:
        descriptors = np.empty((0, 128), np.uint8)
    else:
        descriptors = descriptors.astype(np.uint8)
    fine = normalized(cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA))
    coarse = normalized(cv2.resize(image, (16, 16), interpolation=cv2.INTER_AREA))
    # Store compressed thumbnails, not thousands of full-sized pixel arrays.
    packed = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes()
    digest = hashlib.sha256(content).digest()
    return name, points, descriptors, fine, coarse, packed, digest


def read_features(path):
    archive = zipfile.ZipFile(path) if path.is_file() else None
    try:
        if archive:
            files = sorted(name for name in archive.namelist()
                           if Path(name).suffix.lower() in EXTENSIONS
                           and not any(part.startswith('.') or part == '__MACOSX'
                                       for part in Path(name).parts))
        else:
            files = sorted(p for p in path.rglob('*') if p.is_file()
                           and p.suffix.lower() in EXTENSIONS
                           and not any(part.startswith('.') or part == '__MACOSX'
                                       for part in p.relative_to(path).parts))
        counts = Counter(Path(name).name for name in files)

        def load(name):
            label = Path(name).name
            if counts[label] > 1:
                label = str(name) if archive else name.relative_to(path).as_posix()
            content = archive.read(name) if archive else name.read_bytes()
            return image_features((label, content))

        # Batches bound the number of queued results held by Executor.map.
        features = []
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            for start in range(0, len(files), 128):
                features.extend(pool.map(load, files[start:start + 128]))
        return features
    finally:
        if archive:
            archive.close()


def candidate_pairs(features):
    count = len(features)
    pairs = set()
    # Two resolutions balance detailed structure and robustness to small crops
    # and rotations. Selection is independent of image names and input order.
    for field, neighbors in ((3, 24), (4, 32)):
        matrix = np.stack([item[field] for item in features])
        top_k = min(neighbors, count - 1)
        for start in range(0, count, 128):
            scores = matrix[start:start + 128] @ matrix.T
            for local, row in enumerate(scores):
                index = start + local
                row[index] = -2
                selected = np.argpartition(row, -top_k)[-top_k:]
                for other in selected:
                    if row[other] > 0.15:
                        pairs.add((min(index, int(other)), max(index, int(other))))
    return pairs


def find_duplicates(features):
    if len(features) < 2:
        return set()

    @lru_cache(maxsize=64)
    def pixels(index):
        return cv2.imdecode(np.frombuffer(features[index][5], np.uint8),
                            cv2.IMREAD_GRAYSCALE)

    def verify(pair):
        first, second = pair
        one, two = features[first], features[second]
        if one[6] == two[6]:
            return pair
        points1, desc1 = one[1], one[2]
        points2, desc2 = two[1], two[2]
        raw = float(one[3] @ two[3])
        # Re-encoding alone changes pixels very little. This also covers photos
        # with too few distinctive points for geometric registration.
        if raw >= 0.9995:
            return pair
        if min(len(desc1), len(desc2)) < 12:
            return None
        desc1 = desc1.astype(np.float32)
        desc2 = desc2.astype(np.float32)
        matcher = cv2.BFMatcher(cv2.NORM_L2)
        # Quickly reject unrelated candidates before matching every descriptor.
        # Spread the sample over the image's feature list, not one local patch.
        if min(len(desc1), len(desc2)) > 160:
            sample = np.linspace(0, len(desc1) - 1, 96, dtype=np.int32)
            preliminary = matcher.knnMatch(desc1[sample], desc2, k=2)
            if sum(m.distance < 0.75 * n.distance for m, n in preliminary) < 2:
                return None
        matches = matcher.knnMatch(desc1, desc2, k=2)
        good = [m for m, n in matches if m.distance < 0.75 * n.distance]
        if len(good) < 12:
            return None
        source = np.float32([points1[m.queryIdx] for m in good])
        target = np.float32([points2[m.trainIdx] for m in good])
        transform, mask = cv2.estimateAffinePartial2D(
            source, target, method=cv2.RANSAC, ransacReprojThreshold=2.0,
            maxIters=1500, confidence=0.995
        )
        if transform is None or mask is None or not np.isfinite(transform).all():
            return None
        inliers = mask.ravel().astype(bool)
        support = int(inliers.sum())
        if (support < 12 or support / len(good) < 0.65
                or support / min(len(desc1), len(desc2)) < 0.10):
            return None
        errors = np.linalg.norm(source @ transform[:, :2].T
                                + transform[:, 2] - target, axis=1)
        if np.median(errors[inliers]) > 0.5:
            return None
        area_scale = abs(float(np.linalg.det(transform[:, :2])))
        if area_scale < 1e-6:
            return None
        image1, image2 = pixels(first), pixels(second)
        size = (image2.shape[1], image2.shape[0])
        valid = cv2.warpAffine(np.ones(image1.shape, np.uint8), transform, size)
        valid = cv2.erode(valid, np.ones((5, 5), np.uint8)).astype(bool)
        area = int(valid.sum())
        # Either image may be a crop of the other; test both directions.
        coverage = max(area / image2.size, area / (image1.size * area_scale))
        if coverage < 0.95 or area < 100:
            return None
        warped = cv2.warpAffine(image1, transform, size)
        a = normalized(cv2.GaussianBlur(warped, (5, 5), 0)[valid])
        b = normalized(cv2.GaussianBlur(image2, (5, 5), 0)[valid])
        if float(a @ b) < 0.65:
            return None
        return pair

    candidates = candidate_pairs(features)
    # Exact copies must not be lost to a nearest-neighbor limit in large groups.
    by_digest = defaultdict(list)
    for index, item in enumerate(features):
        by_digest[item[6]].append(index)
    for group in by_digest.values():
        candidates.update(combinations(group, 2))
    accepted = set()
    ordered = sorted(candidates)
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for start in range(0, len(ordered), 512):
            for pair in pool.map(verify, ordered[start:start + 512]):
                if pair is not None:
                    accepted.add(pair)
    pixels.cache_clear()
    return accepted


def main():
    cv2.setNumThreads(1)
    cv2.setRNGSeed(0)
    features = read_features(discover_input())
    pairs = find_duplicates(features)
    rows = sorted({tuple(sorted((features[a][0], features[b][0]))) for a, b in pairs})
    with open('output.csv', 'w', encoding='utf-8', newline='') as target:
        writer = csv.writer(target, lineterminator='\n')
        writer.writerow(('image_a', 'image_b'))
        writer.writerows(rows)


if __name__ == '__main__':
    main()
