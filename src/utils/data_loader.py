from typing import Literal
import os, cv2, msgpack, gc, time, random
import numpy as np

from .feature_extraction import extract_features
from .image_processing import process_images


def load_data(images_dir: str, depth_dir: str | None = None, subset: int = 0) -> tuple[np.ndarray, np.ndarray, list[str]]:
    classes = os.listdir(images_dir)
    f_names = []

    weights = np.zeros((len(classes),))
    for i, c in enumerate(classes):
        f_names.append(os.listdir(os.path.join(images_dir, c)))
        weights[i] = len(f_names[-1])

    weights /= np.sum(weights)

    labels, images, depths = [], [], []
    for i, (class_name, image_names) in enumerate(zip(classes, f_names)):
        class_image_dir = os.path.join(images_dir, class_name)
        class_depth_dir = os.path.join(depth_dir, class_name)
        if not os.path.isdir(class_image_dir):
            continue

        if subset > 0:
            image_names = random.sample(image_names, int(weights[i] * subset))

        for image_name in image_names:
            img = cv2.imread(os.path.join(class_image_dir, image_name))

            if os.path.isdir(class_depth_dir):
                # Convert to double to avoid overflows when calculating features
                depth = cv2.imread(os.path.join(class_depth_dir, image_name), cv2.IMREAD_UNCHANGED).astype(np.double)
                depths.append(depth)

            images.append(img)
            labels.append(class_name)

    depths = np.nan_to_num(depths)
    return np.array(images), depths if depths.size > 0 else None, labels


def save_features(images_dir: str, depth_dir: str, features_dir: str, subset: int = 0):
    get_time = lambda t: round(time.perf_counter() - t, 2)

    if not os.path.exists(features_dir) or not os.path.isdir(features_dir):
        os.mkdir(features_dir)

    print("Loading data...")

    t_start = time.perf_counter()
    images, depth, labels = load_data(images_dir, depth_dir, subset)

    with open(f"{features_dir}/labels.msgpack", 'wb') as f:
        f.write(msgpack.packb(labels))

    del labels
    gc.collect()

    print(f"Loaded data and saved labels: {get_time(t_start)}s")

    t_start = time.perf_counter()
    gray_images, hsv_images = process_images(images)

    del images
    gc.collect()

    print(f"Processed images: {get_time(t_start)}s")

    t_start = time.perf_counter()
    features_2d, features_3d = extract_features(gray_images, hsv_images, depth)

    del depth, gray_images, hsv_images
    gc.collect()

    print(f"Extracted features: {get_time(t_start)}s")

    features = np.concatenate(features_3d, axis=1)
    with open(f"{features_dir}/features_3D.msgpack", 'wb') as f:
        f.write(msgpack.packb(features.tolist()))

    print(f"Features 3D Shape: {features.shape}")

    del features
    gc.collect()

    print("Saved 3D features")

    features = np.concatenate(features_2d + features_3d, axis=1)
    with open(f"{features_dir}/features_25D.msgpack", 'wb') as f:
        f.write(msgpack.packb(features.tolist()))

    print(f"Features 2.5D Shape: {features.shape}")

    del features, features_3d
    gc.collect()

    print("Saved normalized 2.5D features")

    features = np.concatenate(features_2d, axis=1)
    with open(f"{features_dir}/features_2D.msgpack", 'wb') as f:
        f.write(msgpack.packb(features.tolist()))

    print(f"Features 2D Shape: {features.shape}")

    del features, features_2d
    gc.collect()

    print("Saved normalized 2D features")


def load_features(features_dir: str, dimension: Literal['2', '25', '3'] = '2') -> tuple[np.ndarray, list[str]]:
    if dimension == '2':
        print("Loading 2D Features")
        with open(f"{features_dir}/features_2D.msgpack", 'rb') as f:
            features = np.array(msgpack.unpackb(f.read()))
    elif dimension == '3':
        print("Loading 3D Features")
        with open(f"{features_dir}/features_3D.msgpack", 'rb') as f:
            features = np.array(msgpack.unpackb(f.read()))
    else:
        print("Loading 2.5D Features")
        with open(f"{features_dir}/features_25D.msgpack", 'rb') as f:
            features = np.array(msgpack.unpackb(f.read()))

    with open(f"{features_dir}/labels.msgpack", 'rb') as f:
        labels = msgpack.unpackb(f.read())

    return features, labels