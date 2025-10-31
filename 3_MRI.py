#!/usr/bin/env python3
"""
MRI-like 2D segmentation: Otsu thresholding -> distance transform -> watershed.
Works for (1) URL input, (2) local path, (3) synthetic fallback. Handles different
scikit-image peak_local_max behaviors (coordinates vs boolean mask).
"""

import os
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from skimage import io, filters, segmentation, feature, color, morphology
from scipy import ndimage as ndi

# ---------------------------
# Utility / core functions
# ---------------------------

def load_image(source: Optional[str] = None, crop: Optional[Tuple[int,int,int,int]] = None):
    """
    Load image from URL or local path. If source is None or load fails, create synthetic image.
    crop: (r0, r1, c0, c1) to slice image[r0:r1, c0:c1]
    Returns: image as 2D float array normalized to 0..1 and a string describing the source.
    """
    img = None
    source_desc = "synthetic"
    if source:
        # Try local file first
        if os.path.exists(source):
            try:
                img = io.imread(source, as_gray=True)
                source_desc = f"local:{source}"
            except Exception as e:
                print(f"Could not load local file '{source}': {e}")
        else:
            # Try URL
            try:
                img = io.imread(source, as_gray=True)
                source_desc = f"url:{source}"
            except Exception as e:
                print(f"Could not load URL '{source}': {e}")

    if img is None:
        # Synthetic two-blob image (MRI-like) as fallback
        x, y = np.ogrid[0:200, 0:200]
        img = 100 + 50 * np.exp(-((x - 100)**2 + (y - 100)**2) / (2 * 20**2))
        img += 30 * np.exp(-((x - 50)**2 + (y - 150)**2) / (2 * 10**2))
        img = img.astype(float)
        source_desc = "synthetic"

    # Optionally crop (r0, r1, c0, c1)
    if crop:
        r0, r1, c0, c1 = crop
        img = img[r0:r1, c0:c1]

    # Normalize to 0..1 floats
    img = img.astype(float)
    if img.max() > img.min():
        img = (img - img.min()) / (img.max() - img.min())

    return img, source_desc

def compute_binary_mask(image: np.ndarray, method: str = "otsu"):
    """
    Compute binary foreground mask. Default: Otsu threshold.
    Returns binary_image (bool) and threshold value.
    """
    if method == "otsu":
        thresh = filters.threshold_otsu(image)
        binary = image > thresh
    else:
        # fallback: mean-based threshold
        thresh = image.mean()
        binary = image > thresh
    return binary, float(thresh)

def compute_distance_and_markers(binary_image: np.ndarray,
                                 footprint_size: int = 3,
                                 min_distance: Optional[int] = None,
                                 gaussian_sigma_for_distance: Optional[float] = None,
                                 fallback_percentile: float = 90):
    """
    Compute distance transform and produce markers robustly across skimage versions.
    - footprint_size: size of local footprint for peak finding (odd integer).
    - min_distance: if provided, will be used to exclude very close peaks (applied after finding peaks).
    - gaussian_sigma_for_distance: optional smoothing on distance map before peak detection.
    - fallback_percentile: if no peaks found, create markers by thresholding distance at this percentile.
    Returns: distance, markers (integer-labeled array), n_initial_peaks, n_filtered_peaks
    """
    # Distance transform (on foreground)
    distance = ndi.distance_transform_edt(binary_image)

    # Optional smoothing to reduce tiny spurious peaks
    if gaussian_sigma_for_distance is not None and gaussian_sigma_for_distance > 0:
        distance = ndi.gaussian_filter(distance, sigma=gaussian_sigma_for_distance)

    # Build footprint for local maxima detection
    footprint = np.ones((footprint_size, footprint_size), dtype=bool)

    # Call peak_local_max and handle different skimage signatures.
    # Newer versions return coordinates array; older versions with indices=False returned a boolean mask.
    peaks_coords = None
    peaks_mask = None
    try:
        # Try call that returns coordinates (newer API)
        coords = feature.peak_local_max(distance, footprint=footprint, labels=binary_image)
        # If coords is None or empty, coords may be an empty array
        if coords is None:
            coords = np.empty((0, 2), dtype=int)
        if coords.ndim == 2 and coords.shape[1] == 2:
            peaks_coords = coords
        else:
            # Not coordinates: maybe older API returned boolean mask; treat as mask
            peaks_mask = np.asarray(coords, dtype=bool)
    except TypeError:
        # Older scikit-image may use indices=False to return boolean mask; try that
        try:
            mask = feature.peak_local_max(distance, indices=False, footprint=footprint, labels=binary_image)
            peaks_mask = np.asarray(mask, dtype=bool)
        except Exception:
            # Give up on peak_local_max; we'll fallback below
            peaks_coords = np.empty((0, 2), dtype=int)

    # If we got coordinates -> convert to mask
    if peaks_coords is not None:
        mask = np.zeros_like(distance, dtype=bool)
        if peaks_coords.size > 0:
            mask[tuple(peaks_coords.T)] = True
        peaks_mask = mask

    # Count initial markers
    n_initial = int(peaks_mask.sum()) if peaks_mask is not None else 0

    # If no peaks found, fallback strategy: threshold distance at percentile, then erode to get seed points
    if peaks_mask is None or peaks_mask.sum() == 0:
        # fallback: threshold
        valid = distance[binary_image]
        if valid.size > 0:
            cutoff = np.percentile(valid, fallback_percentile)
            fallback_mask = (distance >= cutoff) & binary_image
            # Erode to separate regions and reduce marker size
            fallback_mask = morphology.binary_erosion(fallback_mask, morphology.disk(max(1, footprint_size//2)))
            # Label small connected components and take centroids as markers
            labeled_temp, _ = ndi.label(fallback_mask)
            peaks_mask = np.zeros_like(distance, dtype=bool)
            # Use centers of connected components as markers
            for region_label in range(1, labeled_temp.max() + 1):
                coords_region = np.array(np.where(labeled_temp == region_label)).T
                if coords_region.size == 0:
                    continue
                center = coords_region.mean(axis=0).round().astype(int)
                peaks_mask[center[0], center[1]] = True
        else:
            # No foreground pixels -> empty markers
            peaks_mask = np.zeros_like(distance, dtype=bool)

    # Optionally filter peaks by minimal distance: remove peaks that are too close (simple heuristic)
    if min_distance is not None and min_distance > 0:
        # compute distance to nearest marker for each marker and remove those too close
        marker_coords = np.array(np.where(peaks_mask)).T
        keep_mask = np.ones(len(marker_coords), dtype=bool)
        if marker_coords.shape[0] > 1:
            # pairwise distances
            from scipy.spatial.distance import cdist
            dists = cdist(marker_coords, marker_coords)
            np.fill_diagonal(dists, np.inf)
            # For any pair closer than min_distance, remove the second
            for i in range(dists.shape[0]):
                close = np.where(dists[i] < min_distance)[0]
                for j in close:
                    if keep_mask[j]:
                        keep_mask[j] = False
        filtered_mask = np.zeros_like(peaks_mask, dtype=bool)
        filtered_mask[tuple(marker_coords[keep_mask].T)] = True
        filtered_count = int(filtered_mask.sum())
        peaks_mask = filtered_mask
    else:
        filtered_count = int(peaks_mask.sum())

    # Convert boolean marker mask to labeled markers for watershed
    markers, _ = ndi.label(peaks_mask)

    # Final counts
    n_initial_peaks = n_initial
    n_filtered_peaks = filtered_count

    return distance, markers, n_initial_peaks, n_filtered_peaks

def run_watershed(image: np.ndarray,
                  binary_mask: np.ndarray,
                  markers: np.ndarray):
    """
    Run watershed on -distance (so peaks become basins).
    Returns labels (integer-labeled array) and number of regions.
    """
    distance = ndi.distance_transform_edt(binary_mask)
    labels = segmentation.watershed(-distance, markers=markers, mask=binary_mask)
    return labels

# ---------------------------
# High-level pipeline wrapper
# ---------------------------

def segment_image(source: Optional[str] = None,
                  crop: Optional[Tuple[int,int,int,int]] = None,
                  footprint_size: int = 3,
                  min_distance: Optional[int] = None,
                  gaussian_sigma_for_distance: Optional[float] = 0.0,
                  fallback_percentile: float = 90.0,
                  figsize: Tuple[int,int] = (12, 4),
                  save_png: Optional[str] = None,
                  top_n_region_areas: int = 5):
    """
    Full pipeline. Prints diagnostics, displays 3-panel figure, optionally saves PNG.
    """
    image, src_desc = load_image(source, crop=crop)
    print(f"Image source used: {src_desc}")
    print(f"Image shape: {image.shape}")

    binary, thresh = compute_binary_mask(image, method="otsu")
    print(f"Otsu threshold value: {thresh:.6f}")
    print(f"Foreground pixel count: {int(binary.sum())}")

    distance, markers, n_initial, n_filtered = compute_distance_and_markers(
        binary_image=binary,
        footprint_size=footprint_size,
        min_distance=min_distance,
        gaussian_sigma_for_distance=gaussian_sigma_for_distance if gaussian_sigma_for_distance > 0 else None,
        fallback_percentile=fallback_percentile
    )

    print(f"Initial markers found (boolean count) before filtering: {n_initial}")
    print(f"Markers after filtering / fallback: {n_filtered}")

    # Run watershed using the markers
    labels = run_watershed(image, binary, markers)
    n_regions = int(labels.max())
    print(f"Watershed produced {n_regions} region(s) (labels.max()).")

    # Region areas (pixel counts) for output filtering/reporting
    region_areas = []
    for lbl in range(1, labels.max() + 1):
        area = int((labels == lbl).sum())
        region_areas.append((lbl, area))
    # sort by area desc
    region_areas_sorted = sorted(region_areas, key=lambda x: x[1], reverse=True)
    print(f"Top {min(top_n_region_areas, len(region_areas_sorted))} region areas (label, area):")
    for lbl, area in region_areas_sorted[:top_n_region_areas]:
        print(f"  - Label {lbl}: {area} pixels")

    # Visualization: original | binary | colorized watershed
    fig, axs = plt.subplots(1, 3, figsize=figsize)
    axs[0].imshow(image, cmap='gray')
    axs[0].set_title('Original')
    axs[0].axis('off')

    axs[1].imshow(binary, cmap='gray')
    axs[1].set_title('Binary (Otsu)')
    axs[1].axis('off')

    colored = color.label2rgb(labels, image=image, bg_label=0)
    axs[2].imshow(colored)
    axs[2].set_title(f'Watershed (regions={n_regions})')
    axs[2].axis('off')

    plt.tight_layout()

    if save_png:
        try:
            fig.savefig(save_png, dpi=150)
            print(f"Saved visualization to: {save_png}")
        except Exception as e:
            print(f"Could not save PNG: {e}")

    # In a script or notebook, show() will render. Return results for programmatic use.
    plt.show()
    return {
        'image': image,
        'binary': binary,
        'distance': distance,
        'markers': markers,
        'labels': labels,
        'region_areas_sorted': region_areas_sorted,
        'n_regions': n_regions,
        'src_desc': src_desc,
        'otsu_threshold': thresh,
        'n_initial_markers': n_initial,
        'n_filtered_markers': n_filtered
    }

# ---------------------------
# Example main block
# ---------------------------

if __name__ == "__main__":
    # Example usage:
    # - To use a URL: set source_url to a direct grayscale image URL
    # - To use local path: set source_url to a local filename
    # - To use synthetic image: set source_url = None
    source_url = None  # e.g., "https://example.com/my_mri_slice.png" or "data/mri_slice.png"
    # Optional crop example (r0, r1, c0, c1) or None
    crop_region = None  # (50, 200, 50, 200)
    results = segment_image(
        source=source_url,
        crop=crop_region,
        footprint_size=3,
        min_distance=None,
        gaussian_sigma_for_distance=1.0,  # smooth distance a bit to reduce tiny peaks
        fallback_percentile=92.0,
        figsize=(14, 5),
        save_png="watershed_result.png",  # set to None if you don't want to save
        top_n_region_areas=6
    )

    # Example: access returned labels and do something else programmatically
    labels = results['labels']
    # Print a short sample of labels (center 5x5)
    h, w = labels.shape
    ch, cw = h//2, w//2
    sample = labels[max(0, ch-2):min(h, ch+3), max(0, cw-2):min(w, cw+3)]
    print("Center label sample (5x5):")
    print(sample)