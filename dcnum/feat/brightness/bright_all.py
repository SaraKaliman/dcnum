from numba import njit, prange, bool_, float64, int16, uint8
import numpy as np


def brightness_features(image,
                        mask,
                        image_bg=None,
                        image_corr=None):
    mask = np.array(mask, dtype=bool)

    data = {}
    avg_sd = compute_avg_sd_masked_uint8(image, mask)
    data["bright_avg"] = avg_sd[:, 0]
    data["bright_sd"] = avg_sd[:, 1]

    if image_bg is not None:
        data["bg_med"] = compute_median(image_bg)

    if image_bg is not None or image_corr is not None:
        # Background-corrected brightness values
        if image_corr is None:
            image_corr = np.array(image, dtype=np.int16) - image_bg

        avg_sd_corr = compute_avg_sd_masked_int16(image_corr, mask)
        data["bright_bc_avg"] = avg_sd_corr[:, 0]
        data["bright_bc_sd"] = avg_sd_corr[:, 1]

        # Percentiles
        percentiles = compute_percentiles_10_90(image_corr, mask)
        data["bright_perc_10"] = percentiles[:, 0]
        data["bright_perc_90"] = percentiles[:, 1]

    return data


@njit(float64[:, :](uint8[:, :, :], bool_[:, :, :]), parallel=True, cache=True)
def compute_avg_sd_masked_uint8(image, mask):
    size = image.shape[0]
    avg_sd = np.zeros((size, 2), dtype=np.float64)
    for ii in prange(size):
        maski = np.where(mask[ii].ravel())[0]
        masked = image[ii].ravel()[maski]
        avg_sd[ii, 0] = np.mean(masked)
        avg_sd[ii, 1] = np.std(masked)
    return avg_sd


@njit(float64[:, :](int16[:, :, :], bool_[:, :, :]), parallel=True, cache=True)
def compute_avg_sd_masked_int16(image, mask):
    size = image.shape[0]
    avg_sd = np.zeros((size, 2), dtype=np.float64)
    for ii in prange(size):
        maski = np.where(mask[ii].ravel())[0]
        masked = image[ii].ravel()[maski]
        avg_sd[ii, 0] = np.mean(masked)
        avg_sd[ii, 1] = np.std(masked)
    return avg_sd


@njit(float64[:](uint8[:, :, :]), parallel=True, cache=True)
def compute_median(image):
    size = image.shape[0]
    image_med = np.zeros(size, dtype=np.float64)
    for ii in prange(size):
        image_med[ii] = np.median(image[ii].ravel())
    return image_med


@njit(float64[:, :](int16[:, :, :], bool_[:, :, :]), parallel=True, cache=True)
def compute_percentiles_10_90(image, mask):
    size = image.shape[0]
    percentiles = np.zeros((size, 2), dtype=np.float64)
    for ii in prange(size):
        maski = np.where(mask[ii].ravel())[0]
        masked = image[ii].ravel()[maski]
        peri = np.percentile(masked, q=(10, 90))
        percentiles[ii, 0] = peri[0]
        percentiles[ii, 1] = peri[1]
    return percentiles
