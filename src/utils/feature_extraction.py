from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from skimage.filters import gabor_kernel
from scipy import ndimage as ndi
from scipy.stats import skew, kurtosis
import numpy as np
import cv2
from typing import List, Literal
from threading import Thread


# TODO: Implement the log-gabor filter: https://peterkovesi.com/matlabfns/PhaseCongruency/Docs/convexpl.html
# TODO: Use the local energy instead of variance: https://stackoverflow.com/questions/20608458/gabor-feature-extraction
KERNELS = [
    gabor_kernel(frequency, theta=np.pi * theta / 4, sigma_x=sigma, sigma_y=sigma)
    for theta in range(4)
    for sigma in (1, 3)
    for frequency in (0.05, 0.25)
]


def extract_color_features(images, bins=8):
    color_features = [np.histogram(cv2.calcHist([img], [0], None, [bins], [0, 256]), bins=bins, range=(0, 256))[0] for img in images]
    return np.nan_to_num([cf / (np.linalg.norm(cf) + 1e-8) for cf in color_features])


def extract_lbp_features(images, radius=3, n_points=24):
    lbp_features = [local_binary_pattern(img, n_points, radius, method='uniform').flatten() for img in images]
    return np.nan_to_num(lbp_features)


def extract_glcm_features(images, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], properties: List[Literal['contrast', 'dissimilarity', 'energy', 'homogeneity', 'correlation', 'ASM']] = ['energy', 'homogeneity']):
    glcm_features = []
    for img in images:
        glcm = graycomatrix(img, distances=distances, angles=angles, symmetric=True, normed=True)
        feats = [graycoprops(glcm, prop).flatten() for prop in properties]
        glcm_features.append(feats)
    return np.nan_to_num(glcm_features).reshape(len(images), -1)


def extract_gabor_features(images):
    feats = []
    for image in images:
        gabor_features = np.zeros((len(KERNELS), 2), dtype=np.double)
        for k, kernel in enumerate(KERNELS):
            filtered = ndi.convolve(image, kernel, mode='wrap')
            gabor_features[k, 0] = filtered.mean()
            gabor_features[k, 1] = filtered.var()
        feats.append(gabor_features.flatten())

    return np.nan_to_num(feats)


# 3D Features
def extract_principal_plane_features(depth_patch):
    triangle_area = lambda p_1, p_2, p_3: np.linalg.norm(np.cross(p_2 - p_1, p_3 - p_1), axis=2) / 2 

    # Create a grid of x and y coordinates
    h, w = depth_patch.shape
    x, y = np.meshgrid(np.arange(w), np.arange(h))

    # Create a 3D array of points (x, y, depth)
    points = np.dstack((x, y, depth_patch))

    # Prepare arrays for the vertices of the triangles
    p1 = points[:-1, :-1]  # Top-left
    p2 = points[1:, :-1]   # Bottom-left
    p3 = points[:-1, 1:]   # Top-right
    p4 = points[1:, 1:]    # Bottom-right

    # Calculate the area of the two triangles for all pixels at once
    area1 = triangle_area(p1, p2, p3)
    area2 = triangle_area(p2, p4, p3)

    # Sum the areas
    As = np.sum(area1 + area2)
    Ap = (w - 1) * (h - 1)

    rugosity = As / Ap

    # Flatten the arrays
    x = x.flatten()
    y = y.flatten()
    z = depth_patch.flatten()  # Flatten the depth values

    # Create the design matrix for the polynomial terms
    A = np.vstack([
        np.ones_like(z),  # p1
        x,                # p2 * x
        y,                # p3 * y
        x**2,             # p4 * x^2
        x * y,            # p5 * x * y
        y**2,             # p6 * y^2
        x**2 * y,         # p7 * x^2 * y
        x * y**2,         # p8 * x * y^2
        y**3              # p9 * y^3
    ]).T

    # Perform least squares fitting
    # coeffs are p1, p2, p3, p4, p5, p6, p7, p8, and p9
    coeffs, residuals, rank, singular_values = np.linalg.lstsq(A, z, rcond=None)

    # Calculate the fitted z values using the polynomial equation
    z_fitted = (
        coeffs[0] +
        coeffs[1] * x +
        coeffs[2] * y +
        coeffs[3] * x**2 +
        coeffs[4] * x * y +
        coeffs[5] * y**2 +
        coeffs[6] * x**2 * y +
        coeffs[7] * x * y**2 +
        coeffs[8] * y**3
    )

    # Calculate the distances from each point to the fitted plane
    distances = np.abs(z - z_fitted)

    # Calculate mean and standard deviation of the distances
    mean_distance = np.mean(distances)
    std_distance = np.std(distances)

    # Normal vector components
    n_z = -1
    n_x = coeffs[1]  # Coefficient for x
    n_y = coeffs[2]  # Coefficient for y

    # Calculate the magnitude of the normal vector
    normal_magnitude = np.sqrt(n_x**2 + n_y**2 + n_z**2)

    # Calculate the angle with respect to the vertical
    theta = np.degrees(np.arccos(n_z / normal_magnitude))

    return [np.std(z), skew(z), kurtosis(z)] + coeffs.tolist() + [theta, mean_distance, std_distance, rugosity]


def extract_curvatures_and_surface_normals(depth_patch):
    dx, dy = np.gradient(depth_patch)
    dxdx, dxdy = np.gradient(dx)
    dydx, dydy = np.gradient(dy)

    G = (dxdx * dydy - dxdy * dydx) / (1 + dx**2 + dy**2)**2
    M = (dydy + dxdx) / (2 * (1 + dx**2 + dy**2)**(1.5))

    discriminant = np.sqrt(np.maximum(M**2 - G, 0))
    k1 = M + discriminant
    k2 = M - discriminant

    S = (2 / np.pi) * np.arctan2(k2 + k1, k2 - k1)
    C = np.sqrt((k1**2 + k2**2) / 2)

    # Calculate the surface normals
    nx = -dx
    ny = -dy
    nz = np.ones_like(depth_patch)  # Assuming z = depth_patch
    
    # Normalize the surface normals
    norm = np.sqrt(nx**2 + ny**2 + nz**2)
    nx /= norm
    ny /= norm
    nz /= norm

    alpha = np.arctan2(ny, nx)  # Angle in the xy-plane
    beta = np.arctan2(nz, np.sqrt(nx**2 + ny**2))  # Angle from the z-axis

    return (
        np.mean(G),     np.std(G),
        np.mean(M),     np.std(M),
        np.mean(k1),    np.std(k1),
        np.mean(k2),    np.std(k2),
        np.mean(S),     np.std(S),
        np.mean(C),     np.std(C),
        np.mean(alpha), np.std(alpha),
        np.mean(beta),  np.std(beta)
    )


def extract_symmetry_features(depth_patch):
    """Calculate the mean and variance of the symmetry image."""

    # Apply Log Gabor filter to get the symmetry image
    feats = []
    for kernel in KERNELS:
        real =  ndi.convolve(depth_patch, np.real(kernel), mode='wrap')
        imag = ndi.convolve(depth_patch, np.imag(kernel), mode='wrap')

        phase = np.arctan2(imag, real)
        magnitude = np.sqrt(real**2 + imag**2)

        # Calculate mean and variance of the symmetry image
        feats.extend((np.mean(phase), np.var(phase), np.mean(magnitude), np.var(magnitude)))

    return feats


def extract_features(images_gray, images_hsv, depth: np.ndarray | None=None) -> tuple[list[np.ndarray], list[np.ndarray]]:
    logger = lambda name, f: (print(f"Started processing {name}"), f(), print(f"Finished processing {name}"))

    features = {}

    ts: list[Thread] = []

    # Add color features
    func = lambda: features.update({'gray': extract_color_features(images_gray)})
    ts.append(Thread(target=logger, args=("Gray Color Features", func)))

    # Add color HSV features
    func = lambda: features.update({'hsv': extract_color_features(images_hsv)})
    ts.append(Thread(target=logger, args=("HSV Color Features", func)))

    func = lambda: features.update({'lbp': extract_lbp_features(images_gray)})
    ts.append(Thread(target=logger, args=("LBP Features", func)))

    func = lambda: features.update({'glcm': extract_glcm_features(images_gray)})
    ts.append(Thread(target=logger, args=("GLCM Features", func)))

    func = lambda: features.update({'gabor': extract_gabor_features(images_gray)})
    ts.append(Thread(target=logger, args=("Gabor Features", func)))

    if depth is not None:
        func = lambda: features.update({'principal plane': np.nan_to_num([extract_principal_plane_features(d) for d in depth])})
        ts.append(Thread(target=logger, args=("Principal Plane Features", func)))

        func = lambda: features.update({'curvatures': np.nan_to_num([extract_curvatures_and_surface_normals(d) for d in depth])})
        ts.append(Thread(target=logger, args=("Curvature and Surface Normal Features", func)))

        func = lambda: features.update({'symmetry': np.nan_to_num([extract_symmetry_features(d) for d in depth])})
        ts.append(Thread(target=logger, args=("Symmetry Features", func)))

    for t in ts:
        t.start()

    for t in ts:
        t.join()

    features_2d = [features['gray'], features['hsv'], features['lbp'], features['glcm'], features['gabor']]
    features_3d = []

    if depth is not None:
        features_3d = [features['principal plane'], features['curvatures'], features['symmetry']]

    return features_2d, features_3d
