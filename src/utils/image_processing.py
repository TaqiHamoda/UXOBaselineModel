import numpy as np
from typing import Literal
import cv2, gc


def color_correct(image: np.ndarray, method: Literal['clahe', 'saturation', 'histogram', 'manual']='clahe', percent=50, alpha=1.0, beta=0, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Color correct an image using the specified method.

    Parameters:
        image (numpy.ndarray): Input image in BGR format.
        method (str): Method of color correction ('saturation', 'histogram', 'clahe', 'manual').
        alpha (float): Contrast control for manual adjustment (1.0-3.0).
        beta (int): Brightness control for manual adjustment (0-100).
        clip_limit (float): Clip limit for CLAHE.
        tile_grid_size (tuple): Tile grid size for CLAHE.

    Returns:
        numpy.ndarray: Color corrected image.
    """
    if method == 'saturation':
        # Convert to HSV color space
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv_image)
        s = cv2.add(s, percent)  # Increase saturation
        hsv_image = cv2.merge((h, s, v))
        corrected_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
    elif method == 'histogram':
        # Convert to YUV color space
        yuv_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        yuv_image[:, :, 0] = cv2.equalizeHist(yuv_image[:, :, 0])  # The luma (Y channel) is the intensity of the image (gray scale)
        corrected_image = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR)
    elif method == 'clahe':
        # Convert to YUV color space
        yuv_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        yuv_image[:, :, 0] = clahe.apply(yuv_image[:, :, 0])  # The luma (Y channel) is the intensity of the image (gray scale)
        corrected_image = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR)
    else:
        # Adjust brightness and contrast manually
        corrected_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    return corrected_image


def contrast_stretch(image: np.ndarray):
    """
    Apply contrast stretching to an image.

    Parameters:
        image (numpy.ndarray): Input image in grayscale format.

    Returns:
        numpy.ndarray: Contrast stretched image.
    """
    # Convert to float to avoid overflow during calculations
    image_float = image.astype(np.float32)

    # Apply contrast stretching formula
    min_val = np.min(image_float)
    max_val = np.max(image_float)

    stretched_image = (image_float - min_val) * (255 / (max_val - min_val))

    return np.clip(stretched_image, 0, 255).astype(np.uint8)


def superpixel_segmentation(image: np.ndarray, region_size=40, ruler=10.0, n_iters=10) -> np.ndarray:
    # OpenCV Documentation: https://docs.opencv.org/3.4/df/d6c/group__ximgproc__superpixel.html#gacf29df20eaca242645a8d3a2d88e6c06
    
    # Create SLIC superpixels
    slic = cv2.ximgproc.createSuperpixelSLIC(image, region_size=region_size, ruler=ruler, algorithm=cv2.ximgproc.SLICO)
    slic.iterate(n_iters)  # Number of iterations

    # Get the labels of the segments
    return slic.getLabels()


def apply_superpixel_masks(image: np.ndarray, mask: list[np.ndarray], mode: Literal['average', 'contours', 'highlight']='average', border_color: tuple[int, int, int]=(255, 255, 255), border_thickness: int=2, highlight_color: tuple[int, int, int]=(0, 0, 255), alpha: float=0.3) -> list[np.ndarray]:
    output_image = None

    if mode == 'average':
        output_image = np.zeros_like(image)
    elif mode == 'contours' or mode == 'highlight':
        output_image = image.copy()

    m = None
    for label in range(np.max(mask) + 1):
        del m
        gc.collect()

        # If the mask is empty, skip
        m = (mask == label)
        if not np.any(m):
            continue

        if mode == 'average':
            # Calculate the average color for the masked area
            if len(image.shape) == 2:  # Grayscale image
                color = np.average(image[m])
            else:  # Color image
                color = np.average(image[m], axis=0)

            # Apply the average color to the output image
            output_image[m] = color
        elif mode == 'contours':
            # Find contours of the superpixel and draw it
            contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(output_image, contours, -1, border_color, border_thickness)

            del contours
            gc.collect()
        elif mode == 'highlight':
            # Create a colored overlay for the mask
            output_image[m] = highlight_color  # Set the highlight color where the mask is

    if mode == 'highlight':
        # Blend the colored mask with the original image
        output_image = cv2.addWeighted(image, 1 - alpha, output_image, alpha, 0)

    return output_image


def calculate_centroids(mask: list[np.ndarray]) -> list[np.ndarray]:
    centroids = []

    m = None
    for label in range(np.max(mask) + 1):
        del m
        gc.collect()

        # If the mask is empty, skip
        m = (mask == label)
        if not np.any(m):
            continue

        # Get the coordinates of the pixels in the mask
        y_indices, x_indices = np.where(m)  # Get the indices of the pixels in the mask

        if len(y_indices) > 0:  # Ensure the mask is not empty
            # Calculate the centroid as the median of the coordinates
            centroid_x = np.median(x_indices)
            centroid_y = np.median(y_indices)
            centroids.append((centroid_x, centroid_y))

    return np.array(centroids, dtype=int)


def process_images(images: list[np.ndarray], correct_color=True, stretch_contrast=True, clip_limit=40.0, tile_grid_size=(8, 8)) -> tuple[np.ndarray, np.ndarray]:
    images_gray = []
    images_hsv = []
    
    for image in images:
        if correct_color:
            image = color_correct(image, method='clahe', clip_limit=clip_limit, tile_grid_size=tile_grid_size)

        if stretch_contrast:
            image = contrast_stretch(image)

        images_gray.append(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
        images_hsv.append(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))

    return np.array(images_gray), np.array(images_hsv)

