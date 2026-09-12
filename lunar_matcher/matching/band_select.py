"""Band reduction and multi-band transform propagation for IIRS hyperspectral cubes."""

from typing import Tuple, Union, Optional
import numpy as np
import cv2
from sklearn.decomposition import PCA


def select_reference_band(
    hyperspectral_cube: np.ndarray,
    method: str = "pca",
    band_idx: int = 42,
) -> np.ndarray:
    """
    Selects or synthesizes a 2D reference band from an IIRS hyperspectral cube (H, W, B)
    to enable feature matching against panchromatic OHRC/TMC imagery.

    Parameters:
        hyperspectral_cube: 3D numpy array of shape (H, W, B) or (B, H, W)
        method: "pca" (1st principal component) or "single_band" (specific band slice)
        band_idx: Index of band to select if method == "single_band" (default: 42, mid-VNIR ~1000nm)

    Returns:
        2D uint8 numpy array (H, W) normalized to [0, 255]
    """
    cube = hyperspectral_cube
    # Ensure (H, W, B) orientation
    if cube.ndim != 3:
        raise ValueError(f"Expected 3D hyperspectral cube, got ndim={cube.ndim} with shape {cube.shape}")

    if cube.shape[0] < cube.shape[1] and cube.shape[0] < cube.shape[2]:
        # Likely (B, H, W) -> transpose to (H, W, B)
        cube = np.transpose(cube, (1, 2, 0))

    h, w, b = cube.shape

    if method.lower() == "pca":
        # Flatten spatial dimensions: (H*W, B)
        flat = cube.reshape(-1, b).astype(np.float32)
        # Handle possible NaN/Inf values from detector dropouts
        np.nan_to_num(flat, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        pca = PCA(n_components=1)
        pc1 = pca.fit_transform(flat).reshape(h, w)

        # Enforce positive correlation with mean spectrum so craters/albedo aren't inverted
        mean_spectrum = np.mean(flat, axis=1).reshape(h, w)
        if np.corrcoef(pc1.ravel(), mean_spectrum.ravel())[0, 1] < 0:
            pc1 = -pc1

        # Normalize to uint8 [0, 255]
        min_val, max_val = pc1.min(), pc1.max()
        if max_val > min_val:
            comp_u8 = ((pc1 - min_val) / (max_val - min_val) * 255.0).astype(np.uint8)
        else:
            comp_u8 = np.zeros((h, w), dtype=np.uint8)
        return comp_u8

    elif method.lower() == "single_band":
        clamped_idx = max(0, min(band_idx, b - 1))
        band_data = cube[:, :, clamped_idx].astype(np.float32)
        np.nan_to_num(band_data, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        min_val, max_val = band_data.min(), band_data.max()
        if max_val > min_val:
            band_u8 = ((band_data - min_val) / (max_val - min_val) * 255.0).astype(np.uint8)
        else:
            band_u8 = np.zeros((h, w), dtype=np.uint8)
        return band_u8

    else:
        raise ValueError(f"Unknown band selection method '{method}'. Supported: 'pca', 'single_band'")


def apply_transform_to_cube(
    cube: np.ndarray,
    transform: np.ndarray,
    output_shape: Optional[Tuple[int, int]] = None,
    interpolation: int = cv2.INTER_LINEAR,
) -> np.ndarray:
    """
    Applies a single solved geometric transform to all bands of a hyperspectral cube.
    Efficiently broadcasts the warp across hundreds of bands.

    Parameters:
        cube: 3D array (H, W, B)
        transform: 2x3 Affine matrix or 3x3 Homography matrix
        output_shape: Tuple (out_h, out_w). If None, defaults to input (H, W)
        interpolation: cv2 interpolation flag (default: INTER_LINEAR)

    Returns:
        Warped hyperspectral cube of shape (out_h, out_w, B)
    """
    if cube.ndim != 3:
        raise ValueError(f"Expected 3D cube, got {cube.shape}")

    orig_h, orig_w, n_bands = cube.shape
    out_h, out_w = output_shape if output_shape is not None else (orig_h, orig_w)
    dsize = (out_w, out_h)

    # Determine transform type
    is_affine = (transform.shape == (2, 3))
    is_homography = (transform.shape == (3, 3))

    if not (is_affine or is_homography):
        raise ValueError(f"Expected 2x3 affine or 3x3 homography matrix, got shape {transform.shape}")

    # Preallocate output cube with same dtype
    warped_cube = np.empty((out_h, out_w, n_bands), dtype=cube.dtype)

    # Batch process in 4-channel chunks (native OpenCV SIMD acceleration)
    chunk_size = 4
    for start_b in range(0, n_bands, chunk_size):
        end_b = min(start_b + chunk_size, n_bands)
        sub_chunk = cube[:, :, start_b:end_b]

        if is_homography:
            warped_sub = cv2.warpPerspective(
                sub_chunk, transform, dsize, flags=interpolation, borderMode=cv2.BORDER_CONSTANT, borderValue=0
            )
        else:
            warped_sub = cv2.warpAffine(
                sub_chunk, transform, dsize, flags=interpolation, borderMode=cv2.BORDER_CONSTANT, borderValue=0
            )

        if sub_chunk.shape[2] == 1:
            warped_cube[:, :, start_b] = warped_sub
        else:
            warped_cube[:, :, start_b:end_b] = warped_sub

    return warped_cube
