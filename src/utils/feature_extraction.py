from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from scipy.stats import skew, kurtosis
from typing import List, Literal
from threading import Thread
import cv2, numpy as np, time


# https://stackoverflow.com/questions/33781502/how-to-get-the-real-and-imaginary-parts-of-a-gabor-kernel-matrix-in-opencv
KERNELS = [
    cv2.getGaborKernel(ksize=(5, 5), sigma=sigma, theta=np.pi * theta / 4, lambd=1/frequency, gamma=1, psi=0) +  # Real part (the cosine)
    1j * cv2.getGaborKernel(ksize=(5, 5), sigma=sigma, theta=np.pi * theta / 4, lambd=1/frequency, gamma=1, psi=np.pi/2)  # Complex part (the sine)
    for theta in range(4)
    for sigma in (1, 3)
    for frequency in (0.05, 0.25)
]


def extract_color_features(image: np.ndarray, bins: int = 8, range: tuple[int, int] = (0, 256)) -> np.ndarray:
    color_features, _ = np.histogram(image.flatten(), bins=bins, range=range, density=True)
    return color_features


def extract_lbp_features(image: np.ndarray, n_points: int = 24, radius: float = 3.0) -> np.ndarray:
    # https://scikit-image.org/docs/stable/auto_examples/features_detection/plot_local_binary_pattern.html
    lbp_image = local_binary_pattern(image, n_points, radius, method='uniform')

    n_bins = n_points + 2  # Uniform lbp has max(lbp_image) = n_points + 1, and n_bins = max(lbp_image) + 1
    return extract_color_features(lbp_image, bins=n_bins, range=(0, n_bins))


def extract_glcm_features(image: np.ndarray, distances: list[int] = [1], angles: list[float] = [0, np.pi/4, np.pi/2, 3*np.pi/4], properties: List[Literal['contrast', 'dissimilarity', 'energy', 'homogeneity', 'correlation', 'ASM']] = ['dissimilarity', 'energy', 'homogeneity', 'correlation']) -> np.ndarray:
    # https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.graycoprops
    # https://scikit-image.org/docs/stable/auto_examples/features_detection/plot_glcm.html#sphx-glr-auto-examples-features-detection-plot-glcm-py
    glcm = graycomatrix(image, distances=distances, angles=angles, symmetric=True, normed=True)
    return np.concatenate([graycoprops(glcm, prop).flatten() for prop in properties])


def extract_gabor_features(image: np.ndarray) -> list[float]:
    # TODO: Use log gabor filters to speed up runtime (https://peterkovesi.com/matlabfns/PhaseCongruency/Docs/convexpl.html)
    feats = []
    for kernel in KERNELS:
        response_real = cv2.filter2D(src=image, ddepth=-1, kernel=np.real(kernel))
        response_imag = cv2.filter2D(src=image, ddepth=-1, kernel=np.imag(kernel))

        # Calculate features
        # Source: https://stackoverflow.com/questions/20608458/gabor-feature-extraction
        response_squared = response_real**2 + response_imag**2
        local_energy = np.sum(response_squared)
        mean_amplitude = np.mean(np.sqrt(response_squared))
        phase_amplitude = np.atan2(response_imag, response_real).flatten()

        feats.extend((
            local_energy,             mean_amplitude,
            np.mean(phase_amplitude), np.std(phase_amplitude),
            skew(phase_amplitude),    kurtosis(phase_amplitude)
        ))

    return feats


# 3D Features
def extract_principal_plane_features(depth_patch: np.ndarray) -> list[float]:
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
    coeffs, _, _, _ = np.linalg.lstsq(A, z, rcond=None)

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
    mean_distance = np.mean(distances)
    std_distance = np.std(distances)

    # Normal vector components
    n_z = -1
    n_x = coeffs[1]  # Coefficient for x
    n_y = coeffs[2]  # Coefficient for y

    # Calculate angle with repsect to the vertical using the magnitude of the normal vector
    normal_magnitude = np.sqrt(n_x**2 + n_y**2 + n_z**2)
    theta = np.degrees(np.arccos(n_z / normal_magnitude))

    return [np.std(z), skew(z), kurtosis(z)] + coeffs.tolist() + [theta, mean_distance, std_distance, rugosity]


def extract_curvatures_and_surface_normals(depth_patch: np.ndarray) -> list[float]:
    # TODO: Double check and validate results
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

    return [
        np.mean(G),     np.std(G),
        np.mean(M),     np.std(M),
        np.mean(k1),    np.std(k1),
        np.mean(k2),    np.std(k2),
        np.mean(S),     np.std(S),
        np.mean(C),     np.std(C),
        np.mean(alpha), np.std(alpha),
        np.mean(beta),  np.std(beta)
    ]


def extract_features(images_gray: list[np.ndarray], images_hsv: list[np.ndarray], depth: list[np.ndarray] | None=None) -> tuple[np.ndarray, np.ndarray]:
    logger = lambda name, t, f: (print(f"Started processing {name}"), f(), print(f"Finished processing {name}: {round(time.perf_counter() - t, 2)} seconds"))

    features = {}

    ts: list[Thread] = []

    # Add color features
    func = lambda: features.update({'gray': [extract_color_features(img) for img in images_gray]})
    ts.append(Thread(target=logger, args=("Gray Color Features", time.perf_counter(), func)))

    # Add color HSV features
    func = lambda: features.update({'hsv': [extract_color_features(img[:, :, 0]) for img in images_hsv]})
    ts.append(Thread(target=logger, args=("HSV Color Features", time.perf_counter(), func)))

    func = lambda: features.update({'lbp': [extract_lbp_features(img) for img in images_gray]})
    ts.append(Thread(target=logger, args=("LBP Features", time.perf_counter(), func)))

    func = lambda: features.update({'glcm': [extract_glcm_features(img) for img in images_gray]})
    ts.append(Thread(target=logger, args=("GLCM Features", time.perf_counter(), func)))

    func = lambda: features.update({'gabor': [extract_gabor_features(img) for img in images_gray]})
    ts.append(Thread(target=logger, args=("Gabor Features", time.perf_counter(), func)))

    if depth is not None:
        func = lambda: features.update({'principal plane': [extract_principal_plane_features(d) for d in depth]})
        ts.append(Thread(target=logger, args=("Principal Plane Features", time.perf_counter(), func)))

        func = lambda: features.update({'curvatures': [extract_curvatures_and_surface_normals(d) for d in depth]})
        ts.append(Thread(target=logger, args=("Curvature and Surface Normal Features", time.perf_counter(), func)))

        func = lambda: features.update({'symmetry': [extract_gabor_features(d) for d in depth]})
        ts.append(Thread(target=logger, args=("Symmetry Features", time.perf_counter(), func)))

    for t in ts:
        t.start()

    for t in ts:
        t.join()

    features_2d = np.concatenate([features['gray'], features['hsv'], features['lbp'], features['glcm'], features['gabor']], axis=1)

    features_3d = []
    if depth is not None:
        features_3d = np.concatenate([features['principal plane'], features['curvatures'], features['symmetry']], axis=1)

    return np.nan_to_num(features_2d), np.nan_to_num(features_3d)
