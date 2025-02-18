from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

import numpy as np
from typing import Literal
import datetime, cv2, os, threading, gc, random

from src.utils.data_loader import load_features, save_features, extract_features
from src.utils.image_processing import process_images, superpixel_segmentation, apply_mask
from src.classification.SVM import SVMModel


ADJUST_COOR = lambda c, r, rnge: (0, 2*r) if c - r < 0 else (rnge[1] - 1 - 2*r, rnge[1] - 1) if c + r >= rnge[1] else (c - r, c + r)


SOURCE_DIR = '/home/lucky/Development/CIRS/UXOBaselineModel/'

# Constants for directory paths
IMAGES_DIR = f'{SOURCE_DIR}/data/images/'
IMG_DIR = f'{SOURCE_DIR}/data/datasets/miami/dataset_2d/'
DEPTH_DIR = f'{SOURCE_DIR}/data/datasets/miami/dataset_3d/'
RESULTS_DIR = f'{SOURCE_DIR}/data/results/'
MODELS_DIR = f'{SOURCE_DIR}/data/models/'
FEATURES_DIR = f'{SOURCE_DIR}/data/features/'

DATASET_PATH = "/home/lucky/Development/CIRS/BaselineModel/data/datasets/miami/"

ORTHO1_PATH = "/home/lucky/Development/CIRS/BaselineModel/data/images/ortho_maps/plot1_18_240424_t2_ortho.tif"
DEM1_PATH = "/home/lucky/Development/CIRS/BaselineModel/data/images/ortho_maps/plot1_18_240424_t2_dem_rescaled.tif"
MASKS1_PATH = "/home/lucky/Development/CIRS/BaselineModel/data/images/ortho_maps/plot1_masks/"
TILES1_PATH = "/home/lucky/Development/CIRS/BaselineModel/data/images/ortho_maps/plot1_tiles/"
DEPTH1_PATH = "/home/lucky/Development/CIRS/BaselineModel/data/images/ortho_maps/plot1_depth/"

ORTHO3_PATH = "/home/lucky/Development/CIRS/BaselineModel/data/images/ortho_maps/plot3_18_240424_t2_ortho.tif"
DEM3_PATH = "/home/lucky/Development/CIRS/BaselineModel/data/images/ortho_maps/plot3_18_240424_t2_dem_rescaled.tif"
MASKS3_PATH = "/home/lucky/Development/CIRS/BaselineModel/data/images/ortho_maps/plot3_masks/"
TILES3_PATH = "/home/lucky/Development/CIRS/BaselineModel/data/images/ortho_maps/plot3_tiles/"
DEPTH3_PATH = "/home/lucky/Development/CIRS/BaselineModel/data/images/ortho_maps/plot3_depth/"

THREAD_COUNT = 64
WINDOW_SIZE = 400
ANGLES = (0, 90, 180, 270)
PATCH_SIZE = 128
UXO_THRESHHOLD = 0.4
INVALID_THRESHHOLD = 0.01
BG_PER_IMG = 2_000
UXO_SAMPLE_RATE = 0.01  # Percent of UXO pixels that will be used as centers for the patches.


def reconstruct_mosaic(tiles_path):
    get_row_column = lambda s: np.array(s.replace('.png', '').replace('r', '').replace('c', '').split('_')[-2:], int).tolist()

    row = []
    mosaic = []

    r_prev = 0
    for t_name in sorted(os.listdir(tiles_path), key=get_row_column):
        r, _ = get_row_column(t_name)
        if r != r_prev:
            mosaic.append(np.concatenate(row, axis=1))
            r_prev = r

            row.clear()

        t = cv2.imread(f"{tiles_path}/{t_name}", cv2.IMREAD_UNCHANGED)
        row.append(t)

    return np.concatenate(mosaic)


def process_tile(image, depth, mask, dataset_2d_dir, dataset_3d_dir, prefix):
    print(f"Started processing image")

    mask[mask < 3] = 0  # Set Non-UXO pixels to 0
    mask[mask == 99] = -1  # Set invalid areas to -1
    mask[np.average(image, axis=2) == 0] = -1  # Set true black pixels to -1
    mask[mask >= 3] = 1  # Set UXO pixels to 1

    if np.all(np.unique(mask) == -1):
        del image, mask, depth
        gc.collect()
        print("Finished processing image")
        return

    w, h = mask.shape
    uxo_indices = np.where(mask == 1)
    uxo_indices = list(zip(uxo_indices[0], uxo_indices[1]))
    uxo_indices = random.sample(uxo_indices, int(len(uxo_indices) * UXO_SAMPLE_RATE))

    # Oversample so that there are enough valid patches
    bg_indices = np.where(mask == 0)
    bg_indices = list(zip(bg_indices[0], bg_indices[1]))
    bg_indices = random.sample(bg_indices, 2 * BG_PER_IMG) if len(bg_indices) > 2 * BG_PER_IMG else bg_indices

    indices = uxo_indices + bg_indices

    del uxo_indices, bg_indices
    gc.collect()

    bg_count = 0
    t, m, d = None, None, None
    for i, (c_y, c_x) in enumerate(indices):
        del t, m, d
        gc.collect()

        radius = WINDOW_SIZE // 2

        x_s, x_e = ADJUST_COOR(c_x, radius, (0, w - 1))
        y_s, y_e = ADJUST_COOR(c_y, radius, (0, h - 1))

        t = image[y_s:y_e, x_s:x_e, :]
        m = mask[y_s:y_e, x_s:x_e]
        d = depth[y_s:y_e, x_s:x_e]

        # If the patch has any invalid area, skip
        if np.sum(m == -1)/m.size > INVALID_THRESHHOLD:
            continue

        t = cv2.resize(t, (PATCH_SIZE, PATCH_SIZE), interpolation=cv2.INTER_AREA)
        d = cv2.resize(d, (PATCH_SIZE, PATCH_SIZE), interpolation=cv2.INTER_AREA)

        d = np.astype(d, np.double)
        d -= np.min(d)
        d /= np.max(d)
        d = np.astype(255 * d, np.uint8)

        if np.sum(m == 1)/m.size >= UXO_THRESHHOLD:
            h_img, w_img = d.shape
            for angle in ANGLES:
                M = cv2.getRotationMatrix2D((w_img//2, h_img//2), angle, 1)  # Center, rotation angle, scale
                t_rot = cv2.warpAffine(t, M, (w_img, h_img))
                d_rot = cv2.warpAffine(d, M, (w_img, h_img))

                cv2.imwrite(f"{dataset_2d_dir}/uxo/{prefix}-{i}_{angle}.png", t_rot)
                cv2.imwrite(f"{dataset_3d_dir}/uxo/{prefix}-{i}_{angle}.png", d_rot)

            del t_rot, d_rot
            gc.collect()
        elif np.all(m == 0) and bg_count < BG_PER_IMG:
            cv2.imwrite(f"{dataset_2d_dir}/background/{prefix}-{i}.png", t)
            cv2.imwrite(f"{dataset_3d_dir}/background/{prefix}-{i}.png", d)
            bg_count += 1

    print("Finished processing image")

    del image, mask, depth, indices
    gc.collect()


def train_model(n_components: int = 100, dimension: Literal['2', '25', '3'] = '25', use_saved_features=True, subset_size=0):
    if not use_saved_features:
        save_features(IMG_DIR, DEPTH_DIR, FEATURES_DIR, subset=subset_size)

    # Load features and encode labels
    X_data, labels = load_features(FEATURES_DIR, dimension=dimension)
    le = LabelEncoder()
    y_data = le.fit_transform(labels)

    print(f"Training start time: {datetime.datetime.now().isoformat()}")

    # Transform and split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, stratify=y_data, test_size=0.1)

    # Train model on full dataset and save it
    model = SVMModel(model_dir=MODELS_DIR, n_components=n_components)
    model.train(X_train, y_train)
    model.save_model()

    # Evaluate on the test set
    y_pred = model.evaluate(X_test)

    # Save the classification report to a file
    if not os.path.exists(RESULTS_DIR) or not os.path.isdir(RESULTS_DIR):
        os.mkdir(RESULTS_DIR)

    with open(f"{RESULTS_DIR}/{model.model_name}_{dimension}D.txt", 'w') as f:
        y_test_original = le.inverse_transform(y_test)
        y_pred_original = le.inverse_transform(y_pred)
        print(classification_report(y_test_original, y_pred_original, zero_division=0))
        print(classification_report(y_test_original, y_pred_original, zero_division=0), file=f)


def run_inference(image_path, depth_path, region_size=400, window_size=400, subdivisions=9, threshold=3, dimension='25'):
    print(f"Running inference ({dimension}D) on:\n{image_path}\n")

    img = cv2.imread(image_path)
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

    print("Applying Superpixel Segmentation")
    labels, centroids = superpixel_segmentation(img, ruler=1, region_size=region_size)

    # Loop over each centroid
    imgs = []
    depths = []
    patches = []
    for c_y, c_x in centroids:
        window_radius = window_size // 2

        # Calculate patch boundaries
        x_start, x_end = ADJUST_COOR(c_x, window_radius, (0, img.shape[1]))
        y_start, y_end = ADJUST_COOR(c_y, window_radius, (0, img.shape[0]))

        # Determine number of subdivisions
        num_subdivisions = int(np.sqrt(subdivisions)) + 1

        # Create coordinate arrays for subdivisions
        x_coords = np.linspace(x_start, x_end, num_subdivisions, endpoint=True, dtype=int)
        y_coords = np.linspace(y_start, y_end, num_subdivisions, endpoint=True, dtype=int)
        
        # Calculate steps between subdivision boundaries
        x_steps = (x_coords[1:] - x_coords[:-1]) // 2
        y_steps = (y_coords[1:] - y_coords[:-1]) // 2
        for x_step in x_steps:
            for y_step in y_steps:
                x_sub_start, x_sub_end = ADJUST_COOR(x_step, window_radius, (0, img.shape[1]))
                y_sub_start, y_sub_end = ADJUST_COOR(y_step, window_radius, (0, img.shape[0]))

                # Extract and resize image patch
                img_patch = cv2.resize(img[x_sub_start:x_sub_end, y_sub_start:y_sub_end, :], (128, 128), interpolation=cv2.INTER_AREA)
                imgs.append(img_patch)

                # Extract, resize, and normalize depth patch
                depth_patch = cv2.resize(depth[x_sub_start:x_sub_end, y_sub_start:y_sub_end], (128, 128), interpolation=cv2.INTER_AREA)
                depth_patch = depth_patch.astype(np.double)
                depth_patch -= np.min(depth_patch)
                depth_patch /= np.max(depth_patch)
                depth_patch = np.nan_to_num(255 * depth_patch).astype(np.uint8).astype(np.double)
                depths.append(depth_patch)

                # Store corresponding label
                patches.append(labels[c_x, c_y])

    print("Processing patches")

    patches = np.array(patches)
    gray_images, hsv_images = process_images(imgs)
    features_2d, features_3d = extract_features(gray_images, hsv_images, None if dimension == '2' else depths)

    if dimension == '3':
        features = features_3d
    elif dimension == '2':
        features = features_2d
    else:
        features = np.concatenate([features_2d, features_3d], axis=1)

    print("Loading model and running inference")

    # SVM Model
    model = SVMModel(model_dir=MODELS_DIR)
    model.load_model(f"SVM_{dimension}D.pkl")

    y_pred = model.evaluate(features)

    uxos = patches[np.where(y_pred == 1)]
    for uxo in np.unique(uxos):
        if uxos[uxos == uxo].size < threshold:
            uxos[uxos == uxo] = -1

    uxo_mask = np.isin(labels, uxos).astype(int)
    uxo_mask[uxo_mask == 0] = -1

    return apply_mask(img, uxo_mask, mode='highlight')


if __name__ == "__main__":
    image_paths = (
        f"{SOURCE_DIR}/data/images/ortho_maps/plot1_tiles/plot1_18_240424_t2_ortho_r01_c03.png",
        f"{SOURCE_DIR}/data/images/ortho_maps/plot1_tiles/plot1_18_240424_t2_ortho_r01_c04.png",
        f"{SOURCE_DIR}/data/images/ortho_maps/plot1_tiles/plot1_18_240424_t2_ortho_r02_c03.png",
        f"{SOURCE_DIR}/data/images/ortho_maps/plot1_tiles/plot1_18_240424_t2_ortho_r03_c03.png"
    )

    depth_paths = (
        f"{SOURCE_DIR}/data/images/ortho_maps/plot1_depth/plot1_18_240424_t2_ortho_r01_c03.png",
        f"{SOURCE_DIR}/data/images/ortho_maps/plot1_depth/plot1_18_240424_t2_ortho_r01_c04.png",
        f"{SOURCE_DIR}/data/images/ortho_maps/plot1_depth/plot1_18_240424_t2_ortho_r02_c03.png",
        f"{SOURCE_DIR}/data/images/ortho_maps/plot1_depth/plot1_18_240424_t2_ortho_r03_c03.png"
    )

    ts: list[threading.Thread] = []
    for i, (image_path, depth_path) in enumerate(zip(image_paths, depth_paths)):
        func = lambda: cv2.imwrite(f"./inf_{i}.png", run_inference(image_path, depth_path, region_size=400, dimension='25'))
        t = threading.Thread(target=func)
        ts.append(t)
        ts[-1].start()

    for t in ts:
        t.join()
    exit()
