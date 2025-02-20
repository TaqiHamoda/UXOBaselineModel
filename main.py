from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

import numpy as np
from typing import Literal
import datetime, cv2, os, threading, gc, random, yaml

from src.utils.data_loader import load_features, save_features, extract_features
from src.utils.image_processing import process_images, superpixel_segmentation, apply_mask
from src.classification.SVM import SVMModel


ADJUST_COOR = lambda c, r, rnge: (0, 2*r) if c - r < 0 else (rnge[1] - 1 - 2*r, rnge[1] - 1) if c + r >= rnge[1] else (c - r, c + r)


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


def process_data(image, depth, mask, indices, bg_max, dataset_dir, prefix, uxo_threshold, invalid_threshold, window_size, patch_size, angles):
    print("Started thread")

    bg_count = 0
    w, h = mask.shape
    t, m, d = None, None, None
    for i, (c_y, c_x) in enumerate(indices):
        del t, m, d
        gc.collect()

        radius = window_size // 2

        x_s, x_e = ADJUST_COOR(c_x, radius, (0, w - 1))
        y_s, y_e = ADJUST_COOR(c_y, radius, (0, h - 1))

        t = image[y_s:y_e, x_s:x_e, :]
        m = mask[y_s:y_e, x_s:x_e]
        d = depth[y_s:y_e, x_s:x_e]

        # If the patch has any invalid area, skip
        if np.sum(m == -1)/m.size > invalid_threshold:
            continue

        t = cv2.resize(t, (patch_size, patch_size), interpolation=cv2.INTER_AREA)
        d = cv2.resize(d, (patch_size, patch_size), interpolation=cv2.INTER_AREA)

        d = np.astype(d, np.double)
        d -= np.min(d)
        d /= np.max(d)
        d = np.astype(255 * d, np.uint8)

        if np.sum(m == 1)/m.size >= uxo_threshold:
            h_img, w_img = d.shape
            for angle in angles:
                M = cv2.getRotationMatrix2D((w_img//2, h_img//2), angle, 1)  # Center, rotation angle, scale
                t_rot = cv2.warpAffine(t, M, (w_img, h_img))
                d_rot = cv2.warpAffine(d, M, (w_img, h_img))

                cv2.imwrite(f"{dataset_dir}/2D/uxo/{prefix}-{i}_{angle}.png", t_rot)
                cv2.imwrite(f"{dataset_dir}/3D/uxo/{prefix}-{i}_{angle}.png", d_rot)

            del t_rot, d_rot
            gc.collect()
        elif np.all(m == 0) and bg_count < bg_max:
            cv2.imwrite(f"{dataset_dir}/2D/background/{prefix}-{i}.png", t)
            cv2.imwrite(f"{dataset_dir}/3D/background/{prefix}-{i}.png", d)
            bg_count += 1


def create_dataset(image, depth, mask, dataset_dir, prefix='', bg_per_img=20_000, thread_count=64, uxo_sample_rate=0.01, uxo_threshold=0.4, invalid_threshold=0.01, window_size=400, patch_size=128, angles=(0, 90, 180, 270)):
    print(f"Started processing image {prefix}")

    mask[mask < 3] = 0  # Set Non-UXO pixels to 0
    mask[mask == 99] = -1  # Set invalid areas to -1
    mask[np.average(image, axis=2) == 0] = -1  # Set true black pixels to -1
    mask[mask >= 3] = 1  # Set UXO pixels to 1

    if np.all(np.unique(mask) == -1):
        del image, mask, depth
        gc.collect()
        print(f"Finished processing image {prefix}")
        return

    uxo_indices = np.where(mask == 1)
    uxo_indices = list(zip(uxo_indices[0], uxo_indices[1]))
    uxo_indices = random.sample(uxo_indices, int(len(uxo_indices) * uxo_sample_rate))

    # Oversample so that there are enough valid patches
    bg_indices = np.where(mask == 0)
    bg_indices = list(zip(bg_indices[0], bg_indices[1]))
    bg_indices = random.sample(bg_indices, 2 * bg_per_img) if len(bg_indices) > 2 * bg_per_img else bg_indices

    indices = uxo_indices + bg_indices

    del uxo_indices, bg_indices
    gc.collect()

    ts = []
    bg_count = bg_per_img // thread_count
    indices_count = len(indices) // thread_count
    for i in range(thread_count):
        ts.append(threading.Thread(
            target=process_data,
            args=(image, depth, mask, indices[i*indices_count:(i + 1)*indices_count], bg_count, dataset_dir, prefix, uxo_threshold, invalid_threshold, window_size, patch_size, angles)
        ))
        ts[-1].start()

    for t in ts:
        t.join()

    print(f"Finished processing image {prefix}")

    del image, mask, depth, indices
    gc.collect()


def train_model(dataset_dir, features_dir, models_dir, results_dir, test_size=0.1, n_components: int = 100, dimension: Literal['2', '25', '3'] = '25', use_saved_features=True, subset_size=0):
    if not use_saved_features:
        save_features(f"{dataset_dir}/2D/", f"{dataset_dir}/3D/", features_dir, subset=subset_size)

    # Load features and encode labels
    X_data, labels = load_features(features_dir, dimension=dimension)
    le = LabelEncoder()
    y_data = le.fit_transform(labels)

    print(f"Training start time: {datetime.datetime.now().isoformat()}")

    # Transform and split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, stratify=y_data, test_size=test_size)

    # Train model on full dataset and save it
    model = SVMModel(model_dir=models_dir, n_components=n_components)
    model.train(X_train, y_train)
    model.save_model()

    # Evaluate on the test set
    y_pred = model.evaluate(X_test)

    # Save the classification report to a file
    if not os.path.exists(results_dir) or not os.path.isdir(results_dir):
        os.mkdir(results_dir)

    with open(f"{results_dir}/{model.model_name}_{dimension}D.txt", 'w') as f:
        y_test_original = le.inverse_transform(y_test)
        y_pred_original = le.inverse_transform(y_pred)
        print(classification_report(y_test_original, y_pred_original, zero_division=0))
        print(classification_report(y_test_original, y_pred_original, zero_division=0), file=f)


def run_inference(image_path, depth_path, models_dir, model_name, region_size=400, window_size=400, patch_size=128, subdivide_axis=3, threshold=3, dimension: Literal['2', '25', '3']='25'):
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

        # Create coordinate arrays for subdivisions
        x_coords = np.linspace(x_start, x_end, subdivide_axis + 1, endpoint=True, dtype=int)
        y_coords = np.linspace(y_start, y_end, subdivide_axis + 1, endpoint=True, dtype=int)
        
        # Calculate steps between subdivision boundaries
        x_steps = (x_coords[1:] - x_coords[:-1]) // 2
        y_steps = (y_coords[1:] - y_coords[:-1]) // 2
        for x_step in x_steps:
            for y_step in y_steps:
                x_sub_start, x_sub_end = ADJUST_COOR(x_step, window_radius, (0, img.shape[1]))
                y_sub_start, y_sub_end = ADJUST_COOR(y_step, window_radius, (0, img.shape[0]))

                # Extract and resize image patch
                img_patch = cv2.resize(img[x_sub_start:x_sub_end, y_sub_start:y_sub_end, :], (patch_size, patch_size), interpolation=cv2.INTER_AREA)
                imgs.append(img_patch)

                # Extract, resize, and normalize depth patch
                depth_patch = cv2.resize(depth[x_sub_start:x_sub_end, y_sub_start:y_sub_end], (patch_size, patch_size), interpolation=cv2.INTER_AREA)
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
    model = SVMModel(model_dir=models_dir)
    model.load_model(model_name)

    y_pred = model.evaluate(features)

    uxos = patches[np.where(y_pred == 1)]
    for uxo in np.unique(uxos):
        if uxos[uxos == uxo].size < threshold:
            uxos[uxos == uxo] = -1

    uxo_mask = np.isin(labels, uxos).astype(int)
    uxo_mask[uxo_mask == 0] = -1

    return apply_mask(img, uxo_mask, mode='highlight')


if __name__ == "__main__":
    # Load configuration
    with open('config.yaml') as f:
        config = yaml.safe_load(f)
    
    # Set directories from config
    source_dir = config['directories']['source_dir']
    tiles_dir = f"{source_dir}/{config['directories']['tiles_dir']}"
    dataset_dir = f"{source_dir}/{config['directories']['dataset_dir']}"
    results_dir = f"{source_dir}/{config['directories']['results_dir']}"
    models_dir = f"{source_dir}/{config['directories']['models_dir']}"
    features_dir = f"{source_dir}/{config['directories']['features_dir']}"
    
    if not os.path.exists(tiles_dir) or not os.path.isdir(tiles_dir):
        print("The tiles folder doesn't exist. Please create the tiles folder as explained in the README file.")
        exit()

    # Create directories if they don't exist
    for dir_path in [dataset_dir, results_dir, models_dir, features_dir, f"{dataset_dir}/2D/", f"{dataset_dir}/3D/"]:
        if not os.path.exists(dir_path) or not os.path.isdir(dir_path):
            os.mkdir(dir_path)

    # Process according to config modes
    if config['create_dataset']['enabled']:
        print("Creating dataset...")

        for dtset in os.listdir(tiles_dir):
            image = reconstruct_mosaic(f"{tiles_dir}/{dtset}/images")
            depth = reconstruct_mosaic(f"{tiles_dir}/{dtset}/depths")
            mask = reconstruct_mosaic(f"{tiles_dir}/{dtset}/masks").astype(np.int16)

            create_dataset(
                image, 
                depth, 
                mask, 
                dataset_dir=dataset_dir,
                prefix=dtset,
                bg_per_img=config['create_dataset']['bg_per_img'],
                thread_count=config['create_dataset']['thread_count'],
                uxo_sample_rate=config['create_dataset']['uxo_sample_rate'],
                uxo_threshold=config['create_dataset']['uxo_threshold'],
                invalid_threshold=config['create_dataset']['invalid_threshold'],
                window_size=config['create_dataset']['window_size'],
                patch_size=config['create_dataset']['patch_size'],
                angles=config['create_dataset']['angles']
            )


    if config['train_model']['enabled']:
        print("Training model...")
        train_model(
            dataset_dir=dataset_dir,
            features_dir=features_dir,
            models_dir=models_dir,
            results_dir=results_dir,
            test_size=config['train_model']['test_size'],
            n_components=config['train_model']['n_components'],
            dimension=config['train_model']['dimension'],
            use_saved_features=config['train_model']['use_saved_features'],
            subset_size=config['train_model']['subset_size']
        )


    if config['run_inference']['enabled']:
        print("Running inference...")
        run_inference(
            image_path=config['run_inference']['image_path'],
            depth_path=config['run_inference']['depth_path'],
            models_dir=models_dir,
            model_name=config['run_inference']['model_name'],
            region_size=config['run_inference']['region_size'],
            window_size=config['run_inference']['window_size'],
            patch_size=config['run_inference']['patch_size'],
            subdivide_axis=config['run_inference']['subdivide_axis'],
            threshold=config['run_inference']['threshold'],
            dimension=config['run_inference']['dimension']
        )