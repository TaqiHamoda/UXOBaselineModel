import numpy as np
from typing import Literal
import cv2, gc


def color_correct(image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: tuple[int, int] = (8, 8)) -> np.ndarray:
    # Convert to YUV color space
    yuv_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    yuv_image[:, :, 0] = clahe.apply(yuv_image[:, :, 0])  # The luma (Y channel) is the intensity of the image (gray scale)
    corrected_image = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR)

    return corrected_image


def contrast_stretch(image: np.ndarray) -> np.ndarray:
    # Convert to float to avoid overflow during calculations
    image_float = image.astype(np.double)

    # Apply contrast stretching formula
    min_vals = np.percentile(image_float, 1.5, axis=(0, 1))
    max_vals = np.percentile(image_float, 98.5, axis=(0, 1))

    # If a channel has a uniform color, avoid division by 0
    if np.any(max_vals - min_vals == 0):
        index = max_vals - min_vals == 0
        max_vals[index] = min_vals[index] + 1

    # Stretch image and clip values
    stretched_image = (image_float - min_vals) / (max_vals - min_vals)
    return np.clip(255 * stretched_image, 0, 255).astype(np.uint8)


def superpixel_segmentation(image: np.ndarray, region_size: int = 40, ruler: float = 10.0, n_iters: int = 10) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # OpenCV Documentation: https://docs.opencv.org/3.4/df/d6c/group__ximgproc__superpixel.html#gacf29df20eaca242645a8d3a2d88e6c06

    # Create SLIC superpixels
    slic = cv2.ximgproc.createSuperpixelSLIC(image, region_size=region_size, ruler=ruler, algorithm=cv2.ximgproc.SLICO)
    slic.iterate(n_iters)  # Number of iterations

    # Get the labels and centroids of the segments
    labels = slic.getLabels()
    centroids = np.array([np.median(np.where(labels == l), axis=1)[::-1] for l in np.unique(labels)], dtype=int)

    return labels, centroids, slic.getLabelContourMask()


def apply_mask(image: np.ndarray, mask: np.ndarray, mode: Literal['contours', 'highlight'] = 'highlight', border_color: tuple[np.uint8, np.uint8, np.uint8] = (255, 255, 255), border_thickness: int = 2, highlight_color: tuple[np.uint8, np.uint8, np.uint8] = (0, 0, 255), alpha: float = 0.3) -> np.ndarray:
    output_image = image.copy()

    if mode == 'contours':
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(output_image, contours, -1, border_color, border_thickness)
    else:
        output_image[mask >= 0] = highlight_color
        output_image = cv2.addWeighted(image, 1 - alpha, output_image, alpha, 0)

    return output_image


def process_images(images: list[np.ndarray], correct_color: bool = True, stretch_contrast: bool = True, clip_limit: float = 40.0, tile_grid_size: tuple[int, int] = (8, 8)) -> tuple[np.ndarray, np.ndarray]:
    images_gray = []
    images_hsv = []
    
    for image in images:
        if correct_color:
            image = color_correct(image, clip_limit=clip_limit, tile_grid_size=tile_grid_size)

        if stretch_contrast:
            image = contrast_stretch(image)

        images_gray.append(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
        images_hsv.append(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))

    return np.array(images_gray), np.array(images_hsv)

