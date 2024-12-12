from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.kernel_approximation import Nystroem
import numpy as np
import matplotlib.pyplot as plt
from typing import Literal
import datetime, cv2, os, pickle, threading, gc, msgpack

from utils.data_loader import load_features, save_features, extract_features
from utils.image_processing import process_images, superpixel_segmentation, apply_superpixel_masks, calculate_centroids
from classification.SVMLinear import SVMLinearModel


SOURCE_DIR = '/home/lucky/Development/CIRS/UXOBaselineModel/'

# Constants for directory paths
IMAGES_DIR = f'{SOURCE_DIR}/data/images/'
FEATURES_2D_DIR = f'{SOURCE_DIR}/data/datasets/miami/dataset_2d/'
DEPTH_DIR = f'{SOURCE_DIR}/data/datasets/miami/dataset_3d/'
RESULTS_DIR = f'{SOURCE_DIR}/data/results/'
MODELS_DIR = f'{SOURCE_DIR}/data/models/'
FEATURES_DIR = f'{SOURCE_DIR}/data/features/'


def train_model(n_components: int = 100, dimension: Literal['2', '25', '3'] = '25'):
    get_pipeline = lambda: Pipeline([
        ('scaling', StandardScaler()),
        ('kernel', Nystroem(n_jobs=-1, n_components=n_components)),
        ('pca', PCA()),
    ], verbose=True)

    # Load features and encode labels
    X_data, labels = load_features(FEATURES_DIR, dimension=dimension)
    le = LabelEncoder()
    y_data = le.fit_transform(labels)

    print(f"Training start time: {datetime.datetime.now().isoformat()}")

    pipeline = get_pipeline().fit(X_data)
    with open(f"{FEATURES_DIR}/pipeline_{dimension}D.pkl", 'wb') as f:
        pickle.dump(pipeline, f)

    # Transform and split the data into training and testing sets
    X_data = pipeline.transform(X_data)
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, stratify=y_data, test_size=0.1)

    # Train model on full dataset and save it
    model = SVMLinearModel(model_dir=MODELS_DIR)
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


def get_masks_centroids(img, depth, mask, centroids, window_size=400, dimension='25'):
    # Cut out batches that are centered around the centroids
    patches = []
    depths = []
    for c_y, c_x in centroids:
        r = window_size//2

        x_s, x_e = c_x - r, c_x + r
        y_s, y_e = c_y - r, c_y + r

        # Shift the batch so that it is within image bounds
        if x_s < 0:
            x_e += -x_s
            x_s = 0
        elif x_e >= img.shape[0]:
            x_s -= x_e - img.shape[0] + 1
            x_e = img.shape[0] - 1

        if y_s < 0:
            y_e += -y_s
            y_s = 0
        elif y_e >= img.shape[1]:
            y_s -= y_e - img.shape[1] + 1
            y_e = img.shape[1] - 1

        patches.append(cv2.resize(img[x_s:x_e, y_s:y_e, :], (128, 128), interpolation=cv2.INTER_AREA))

        d = cv2.resize(depth[x_s:x_e, y_s:y_e], (128, 128), interpolation=cv2.INTER_AREA)
        d = d.astype(np.double)
        d -= np.min(d)
        d /= np.max(d)
        d = ((255*d).astype(np.uint8)).astype(np.double)

        depths.append(d)

    print("Processing patches")

    gray_images, hsv_images = process_images(patches)
    features_2d, features_3d = extract_features(gray_images, hsv_images, None if dimension == '2' else depths)
    
    if dimension == '3':
        features = np.concatenate(features_3d, axis=1)
    else:
        features = np.concatenate(features_2d + features_3d, axis=1)

    with open(f"{FEATURES_DIR}/pipeline_{dimension}D.pkl", 'rb') as f:
        pipeline = pickle.load(f)

    features = pipeline.transform(features)

    print("Loading model and running inference")

    # SVM Model
    model = SVMLinearModel(model_dir=MODELS_DIR)
    model.load_model(f"SVMLinear_{dimension}D.pkl")

    y_pred = model.evaluate(features)

    uxo_centroids = np.where(y_pred == 1)

    uxo_masks = np.isin(mask, uxo_centroids).astype(int)
    uxo_masks[uxo_masks == 0] = -1
    return uxo_masks, centroids[uxo_centroids]

def _run_inference(img, depth, mask, centroids, window_size=400, dimension='25'):
    uxo_masks, _ = get_masks_centroids(img, depth, mask, centroids, window_size=window_size, dimension=dimension)
    return apply_superpixel_masks(img, uxo_masks, mode='highlight')


def run_inference(image_path, depth_path, region_size=400, window_size=400, dimension='25'):
    print(f"Running inference ({dimension}D) on:\n{image_path}\n")

    img = cv2.imread(image_path)
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

    print("Applying Superpixel Segmentation")
    mask = superpixel_segmentation(img, ruler=1, region_size=region_size)
    centroids = calculate_centroids(mask)

    return _run_inference(img, depth, mask, centroids, window_size=window_size, dimension=dimension)


def run_inference_neighbor_polling(image_path: str, depth_path: str, mini_mask: np.ndarray, mini_centroids: np.ndarray, large_mask: np.ndarray, large_centroids: np.ndarray, window_size: int = 400, dimension: Literal['2', '25', '3'] = '25', neighbor_threshold: int = 4):
    print(f"Running neighbor polling inference ({dimension}D) on:\n{image_path}\n")

    img = cv2.imread(image_path)
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

    uxo_masks, uxo_centroids = get_masks_centroids(img, depth, mini_mask, mini_centroids, window_size=window_size, dimension=dimension)

    distances = []
    for c in large_centroids:
        distances.append(np.linalg.norm(c - uxo_centroids, axis=1))

    distances = np.array(distances)
    neighbors_count = np.sum(distances <= window_size//2, axis=1)  # Radius is half of the diameter

    uxo_masks = np.isin(large_mask, np.where(neighbors_count >= neighbor_threshold)).astype(int)
    uxo_masks[uxo_masks == 0] = -1

    return apply_superpixel_masks(img, uxo_masks, mode='highlight')


if __name__ == "__main__":
    # save_features(FEATURES_2D_DIR, DEPTH_DIR, FEATURES_DIR, subset=100)
    # train_model(dimension='2')
    # train_model(dimension='25')
    # train_model(dimension='3')
    # exit()

    # image_paths = (
    #     f"{SOURCE_DIR}/data/images/ortho_maps/plot1_tiles/plot1_18_240424_t2_ortho_r01_c03.png",
    #     f"{SOURCE_DIR}/data/images/ortho_maps/plot1_tiles/plot1_18_240424_t2_ortho_r01_c04.png",
    #     f"{SOURCE_DIR}/data/images/ortho_maps/plot1_tiles/plot1_18_240424_t2_ortho_r02_c03.png",
    #     f"{SOURCE_DIR}/data/images/ortho_maps/plot1_tiles/plot1_18_240424_t2_ortho_r03_c03.png"
    # )

    # depth_paths = (
    #     f"{SOURCE_DIR}/data/images/ortho_maps/plot1_depth/plot1_18_240424_t2_ortho_r01_c03.png",
    #     f"{SOURCE_DIR}/data/images/ortho_maps/plot1_depth/plot1_18_240424_t2_ortho_r01_c04.png",
    #     f"{SOURCE_DIR}/data/images/ortho_maps/plot1_depth/plot1_18_240424_t2_ortho_r02_c03.png",
    #     f"{SOURCE_DIR}/data/images/ortho_maps/plot1_depth/plot1_18_240424_t2_ortho_r03_c03.png"
    # )

    # ts: list[threading.Thread] = []
    # for i, (image_path, depth_path) in enumerate(zip(image_paths, depth_paths)):
    #     func = lambda: cv2.imwrite(f"./inf_{i}.png", run_inference(image_path, depth_path, region_size=50, dimension='3'))
    #     t = threading.Thread(target=func)
    #     ts.append(t)
    #     ts[-1].start()

    # for t in ts:
    #     t.join()
    # exit()

    ortho_path = f"{SOURCE_DIR}/data/images/ortho_maps/plot1_mosaic.large.png"
    depth_path = f"{SOURCE_DIR}/data/images/ortho_maps/plot1_depth.large.png"
    mini_mask_path = f"{SOURCE_DIR}/data/images/ortho_maps/mask_50_plot1.msgpack"
    mini_centroids_path = f"{SOURCE_DIR}/data/images/ortho_maps/centroids_50_plot1.msgpack"
    large_mask_path = f"{SOURCE_DIR}/data/images/ortho_maps/mask_400_plot1.msgpack"
    large_centroids_path = f"{SOURCE_DIR}/data/images/ortho_maps/centroids_400_plot1.msgpack"

    ortho_inf = run_inference(ortho_path, depth_path, region_size=400, dimension='25')
    cv2.imwrite("./ortho_inference.png", ortho_inf)

    with open(mini_mask_path, "rb") as f:
        mini_mask = np.array(msgpack.unpackb(f.read()))

    with open(mini_centroids_path, "rb") as f:
        mini_centroids = np.array(msgpack.unpackb(f.read()))

    with open(large_mask_path, "rb") as f:
        large_mask = np.array(msgpack.unpackb(f.read()))

    with open(large_centroids_path, "rb") as f:
        large_centroids = np.array(msgpack.unpackb(f.read()))

    ortho_inf = run_inference_neighbor_polling(ortho_path, depth_path, mini_mask, mini_centroids, large_mask, large_centroids, dimension='25', neighbor_threshold=16)
    cv2.imwrite("./ortho_inference.png", ortho_inf)
