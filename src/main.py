from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import datetime, cv2, os, gc, threading, pickle

from utils.data_loader import load_features, save_features, extract_features
from utils.image_processing import process_images, superpixel_segmentation, apply_superpixel_masks, calculate_centroids
from classification.KNN import KNNModel
from classification.SVM import SVMModel
from classification.SVMLinear import SVMLinearModel


# Constants for directory paths
IMAGES_DIR = '/home/lucky/Development/CIRS/BaselineModel/data/images/'
FEATURES_2D_DIR = '/home/lucky/Development/CIRS/BaselineModel/data/datasets/miami/dataset_2d/'
DEPTH_DIR = '/home/lucky/Development/CIRS/BaselineModel/data/datasets/miami/dataset_3d/'
RESULTS_DIR = '/home/lucky/Development/CIRS/BaselineModel/data/results/'
# MODELS_DIR = '/home/lucky/Development/CIRS/BaselineModel/data/models/'
# FEATURES_DIR = '/home/lucky/Development/CIRS/BaselineModel/data/features/'

FEATURES_DIR = '/home/lucky/Development/CIRS/BaselineModel/jaguar/features'
MODELS_DIR = '/home/lucky/Development/CIRS/BaselineModel/jaguar/models'

def train_model(dimension='25'):
    # Load features and encode labels
    X_data, labels = load_features(FEATURES_DIR, dim=dimension)
    le = LabelEncoder()
    y_data = le.fit_transform(labels)

    print(f"Training start time: {datetime.datetime.now().isoformat()}")

    # Initialize the model and kernel transform
    model = SVMLinearModel(model_dir=MODELS_DIR)
    X_data = model.fit_transform(X_data, y_data)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, stratify=y_data, test_size=0.1)

    # Train model on full dataset
    model.train(X_train, y_train)

    # Save the trained model
    model.save_model()

    # Evaluate on the test set
    y_pred = model.evaluate(X_test)

    # Save the classification report to a file
    if not os.path.exists(RESULTS_DIR) or not os.path.isdir(RESULTS_DIR):
        os.mkdir(RESULTS_DIR)

    with open(f"{RESULTS_DIR}/{model.model_name}_{dimension}D.txt", 'w') as f:
        y_test_original = le.inverse_transform(y_test)
        y_pred_original = le.inverse_transform(y_pred)
        print(classification_report(y_test_original, y_pred_original, zero_division=0), file=f)


def run_inference(image_path, depth_path, region_size=400, window_size=400, dimension='25'):
    print(f"Running inference ({dimension}D) on:\n{image_path}\n")

    img = cv2.imread(image_path)
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

    print("Applying Superpixel Segmentation")
    mask = superpixel_segmentation(img, ruler=1, region_size=region_size)
    centroids = calculate_centroids(mask)

    # segmented = apply_superpixel_masks(img, mask, mode='contours')
    # plt.imshow(cv2.cvtColor(segmented, cv2.COLOR_BGR2RGB))
    # plt.scatter(centroids[:, 0], centroids[:, 1], s=5, c='red')
    # plt.show()

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

    del centroids
    gc.collect()

    print("Processing patches")

    gray_images, hsv_images = process_images(patches)

    features_2d, features_3d = extract_features(gray_images, hsv_images, None if dimension == '2' else depths)
    
    if dimension == '3':
        features = np.concatenate(features_3d, axis=1)
    else:
        features = np.concatenate(features_2d + features_3d, axis=1)

    del patches, gray_images, hsv_images
    gc.collect()

    with open(f"{FEATURES_DIR}/pipeline_{dimension}D.pkl", 'rb') as f:
        pipeline = pickle.load(f)

    features = pipeline.transform(features)

    print("Loading model and running inference")

    # SVM Model
    model = SVMLinearModel(model_dir=MODELS_DIR)
    model.load_model(f"SVMLinear_{dimension}D.pkl")

    y_pred = model.evaluate(features)

    uxo_masks = np.isin(mask, np.where(y_pred == 1)).astype(int)
    uxo_masks[uxo_masks == 0] = -1

    return apply_superpixel_masks(img, uxo_masks, mode='highlight')


if __name__ == "__main__":
    # save_features(FEATURES_2D_DIR, DEPTH_DIR, FEATURES_DIR)
    # train_model(dimension='3')
    # exit()

    image_paths = (
        "/home/lucky/Development/CIRS/BaselineModel/data/images/ortho_maps/plot1_tiles/plot1_18_240424_t2_ortho_r01_c03.png",
        "/home/lucky/Development/CIRS/BaselineModel/data/images/ortho_maps/plot1_tiles/plot1_18_240424_t2_ortho_r01_c04.png",
        "/home/lucky/Development/CIRS/BaselineModel/data/images/ortho_maps/plot1_tiles/plot1_18_240424_t2_ortho_r02_c03.png",
        "/home/lucky/Development/CIRS/BaselineModel/data/images/ortho_maps/plot1_tiles/plot1_18_240424_t2_ortho_r03_c03.png"
    )

    depth_paths = (
        "/home/lucky/Development/CIRS/BaselineModel/data/images/ortho_maps/plot1_depth/plot1_18_240424_t2_ortho_r01_c03.png",
        "/home/lucky/Development/CIRS/BaselineModel/data/images/ortho_maps/plot1_depth/plot1_18_240424_t2_ortho_r01_c04.png",
        "/home/lucky/Development/CIRS/BaselineModel/data/images/ortho_maps/plot1_depth/plot1_18_240424_t2_ortho_r02_c03.png",
        "/home/lucky/Development/CIRS/BaselineModel/data/images/ortho_maps/plot1_depth/plot1_18_240424_t2_ortho_r03_c03.png"
    )

    ts: list[threading.Thread] = []
    for i, (image_path, depth_path) in enumerate(zip(image_paths, depth_paths)):
        func = lambda: cv2.imwrite(f"./inf_{i}.png", run_inference(image_path, depth_path, region_size=50, dimension='3'))
        t = threading.Thread(target=func)
        ts.append(t)
        ts[-1].start()

    for t in ts:
        t.join()

    # ortho_path = "/home/lucky/Development/CIRS/BaselineModel/data/images/ortho_maps/plot1_mosaic.large.png"
    # depth_path = "/home/lucky/Development/CIRS/BaselineModel/data/images/ortho_maps/plot1_depth.large.png"

    # ortho_inf = run_inference(ortho_path, depth_path, region_size=200, dimension='25')
    # cv2.imwrite("./ortho_inference.png", ortho_inf)

