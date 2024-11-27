import numpy as np
import cv2, os, gc, threading, random

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

GET_ROW_COLUMN = lambda s: np.array(s.replace('.png', '').replace('r', '').replace('c', '').split('_')[-2:], int).tolist()

THREAD_COUNT = 64
WINDOW_SIZE = 400
ANGLES = (0, 90, 180, 270)
PATCH_SIZE = 128
UXO_THRESHHOLD = 0.4
INVALID_THRESHHOLD = 0.01
BG_PER_TILE = 2_000
UXO_SAMPLE_RATE = 0.01  # Percent of UXO pixels that will be used as centers for the patches.


def reconstruct_mosaic(tiles_path):
    row = []
    mosaic = []

    r_prev = 0
    for t_name in sorted(os.listdir(tiles_path), key=GET_ROW_COLUMN):
        r, _ = GET_ROW_COLUMN(t_name)
        if r != r_prev:
            mosaic.append(np.concatenate(row, axis=1))
            r_prev = r

            row.clear()

        t = cv2.imread(f"{tiles_path}/{t_name}", cv2.IMREAD_UNCHANGED)
        row.append(t)

    return np.concatenate(mosaic)


def split_into_tiles(ortho_path, masks_path, tiles_path):
    img = cv2.imread(ortho_path, cv2.IMREAD_UNCHANGED)

    if len(img.shape) != 3:
        img[img == -32767] = 0
        img *= -1
        img -= np.min(img)
        img = np.astype(255*img/np.max(img), np.uint8)

    masks = sorted(os.listdir(masks_path), key=GET_ROW_COLUMN)

    x_s, y_s, x_e, y_e = 0, 0, 0, 0
    for m in masks:
        _, c = GET_ROW_COLUMN(m)

        mask = cv2.imread(f"{masks_path}/{m}", cv2.IMREAD_UNCHANGED)

        if c == 0:
            x_s = x_e
            x_e += mask.shape[0]

            y_s = 0

        y_e = y_s + mask.shape[1]

        del mask
        gc.collect()

        if len(img.shape) == 3:
            cv2.imwrite(f"{tiles_path}/{m}", img[x_s:x_e, y_s:y_e, :3])
        else:
            cv2.imwrite(f"{tiles_path}/{m}", img[x_s:x_e, y_s:y_e])

        y_s = y_e

    del img
    gc.collect()


def process_tile(tile_name, tiles_path, depth_path, masks_path, dataset_2d_dir, dataset_3d_dir, prefix):
    print(f"Started processing tile {tile_name}")

    tile = cv2.imread(f"{tiles_path}/{tile_name}")
    depth = cv2.imread(f"{depth_path}/{tile_name}", cv2.IMREAD_UNCHANGED)
    mask = np.astype(cv2.imread(f"{masks_path}/{tile_name}", cv2.IMREAD_UNCHANGED), int)

    mask[mask < 3] = 0  # Set Non-UXO pixels to 0
    mask[mask == 99] = -1  # Set invalid areas to -1
    mask[np.average(tile, axis=2) == 0] = -1  # Set true black pixels to -1
    mask[mask >= 3] = 1  # Set UXO pixels to 1

    if np.all(np.unique(mask) == -1):
        del tile, mask, depth
        gc.collect()
        print(f"Finished processing tile {tile_name}")
        return

    w, h = mask.shape
    r, c = GET_ROW_COLUMN(tile_name)

    uxo_indices = np.where(mask == 1)
    uxo_indices = list(zip(uxo_indices[0], uxo_indices[1]))
    uxo_indices = random.sample(uxo_indices, int(len(uxo_indices)*UXO_SAMPLE_RATE))

    # Oversample so that there are enough valid patches
    bg_indices = np.where(mask == 0)
    bg_indices = list(zip(bg_indices[0], bg_indices[1]))
    bg_indices = random.sample(bg_indices, 2*BG_PER_TILE) if len(bg_indices) > 2*BG_PER_TILE else bg_indices

    indices = uxo_indices + bg_indices

    del uxo_indices, bg_indices
    gc.collect()

    i, bg_count = 0, 0
    t, m, d = None, None, None
    for c_y, c_x in indices:
        del t, m, d
        gc.collect()

        radius = WINDOW_SIZE // 2

        x_s, x_e = c_x - radius, c_x + radius
        y_s, y_e = c_y - radius, c_y + radius

        # Shift the batch so that it is within image bounds
        if x_s < 0:
            x_e += -x_s
            x_s = 0
        elif x_e >= w:
            x_s -= x_e - w + 1
            x_e = w - 1

        if y_s < 0:
            y_e += -y_s
            y_s = 0
        elif y_e >= h:
            y_s -= y_e - h + 1
            y_e = h - 1

        t = tile[y_s:y_e, x_s:x_e, :]
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
        d = np.astype(255*d, np.uint8)

        if np.sum(m == 1)/m.size >= UXO_THRESHHOLD:
            h_img, w_img = d.shape
            for angle in ANGLES:
                M = cv2.getRotationMatrix2D((w_img//2, h_img//2), angle, 1)  # Center, rotation angle, scale
                t_rot = cv2.warpAffine(t, M, (w_img, h_img))
                d_rot = cv2.warpAffine(d, M, (w_img, h_img))

                cv2.imwrite(f"{dataset_2d_dir}/uxo/{prefix}-{r}_{c}_{i}_{angle}.png", t_rot)
                cv2.imwrite(f"{dataset_3d_dir}/uxo/{prefix}-{r}_{c}_{i}_{angle}.png", d_rot)

            del t_rot, d_rot
            gc.collect()
        elif np.all(m == 0) and bg_count < BG_PER_TILE:
            cv2.imwrite(f"{dataset_2d_dir}/background/{prefix}-{r}_{c}_{i}.png", t)
            cv2.imwrite(f"{dataset_3d_dir}/background/{prefix}-{r}_{c}_{i}.png", d)
            bg_count += 1

        i += 1

    print(f"Finished processing tile {tile_name}")

    del tile, mask, depth, indices
    gc.collect()


def create_patches(masks_path, tiles_path, depth_path, dataset_path, prefix="image"):
    if not os.path.exists(dataset_path) or not os.path.isdir(dataset_path):
        os.mkdir(dataset_path)

    dataset_2d_dir = f"{dataset_path}/dataset_2d/"
    if not os.path.exists(dataset_2d_dir) or not os.path.isdir(dataset_2d_dir):
        os.mkdir(dataset_2d_dir)

    dataset_3d_dir = f"{dataset_path}/dataset_3d/"
    if not os.path.exists(dataset_3d_dir) or not os.path.isdir(dataset_3d_dir):
        os.mkdir(dataset_3d_dir)

    for dir in (dataset_2d_dir, dataset_3d_dir):
        background_dir = f"{dir}/background/"
        if not os.path.exists(background_dir) or not os.path.isdir(background_dir):
            os.mkdir(background_dir)

        uxo_dir = f"{dir}/uxo/"
        if not os.path.exists(uxo_dir) or not os.path.isdir(uxo_dir):
            os.mkdir(uxo_dir)

    f_names = sorted(os.listdir(masks_path), key=GET_ROW_COLUMN)

    ts: list[threading.Thread] = []
    for f_name in f_names:
        ts.append(threading.Thread(target=process_tile, args=(f_name, tiles_path, depth_path, masks_path, dataset_2d_dir, dataset_3d_dir, prefix)))

    curr_threads = 0
    t_started: list[threading.Thread] = []
    while len(ts) > 0:
        if curr_threads >= THREAD_COUNT:
            t_started.pop(0).join()
            curr_threads -= 1

        t_started.append(ts.pop(0))
        t_started[-1].start()

        curr_threads += 1

    for t in t_started:
        t.join()


if __name__ == "__main__":
    # split_into_tiles(DEM1_PATH, MASKS1_PATH, DEPTH1_PATH)
    # split_into_tiles(DEM3_PATH, MASKS3_PATH, DEPTH3_PATH)

    # create_patches(MASKS1_PATH, TILES1_PATH, DEPTH1_PATH, DATASET_PATH, prefix="plot1")
    # create_patches(MASKS3_PATH, TILES3_PATH, DEPTH3_PATH, DATASET_PATH, prefix="plot3")

    cv2.imwrite('./mosaic1.png', reconstruct_mosaic(TILES1_PATH))
    cv2.imwrite('./depth1.png', reconstruct_mosaic(DEPTH1_PATH))
    cv2.imwrite('./mosaic3.png', reconstruct_mosaic(TILES3_PATH))
    cv2.imwrite('./depth3.png', reconstruct_mosaic(DEPTH3_PATH))

