import numpy as np
from numpy import ma


def brightness_features(image,
                        mask,
                        image_bg=None):
    mask = np.asarray(mask, dtype=bool)

    size = image.shape[0]
    data = {}
    image_masked = image.view(ma.MaskedArray)
    image_masked.mask = ~mask

    data["bright_avg"] = ma.mean(image_masked, axis=(1, 2)).data
    data["bright_sd"] = ma.std(image_masked, axis=(1, 2)).data

    if image_bg is not None:
        data["bg_med"] = np.median(image_bg, axis=(1, 2))

        # Background-corrected brightness values
        image_corr = np.array(image, dtype=int) - image_bg
        image_corr_masked = image_corr.view(ma.MaskedArray)
        image_corr_masked.mask = ~mask
        data["bright_bc_avg"] = np.mean(image_corr_masked, axis=(1, 2))
        data["bright_bc_sd"] = np.std(image_corr_masked, axis=(1, 2))

        # Percentiles
        p10 = np.zeros(size, dtype=float)
        p90 = np.zeros(size, dtype=float)
        for ii in range(size):
            p10[ii], p90[ii] = \
                np.percentile(image_corr[ii][mask[ii]], q=(10, 90))
        data["bright_perc_10"] = p10
        data["bright_perc_90"] = p90

    return data
