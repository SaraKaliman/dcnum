from mahotas.features import haralick as mh_haralick
import numpy as np

from .common import haralick_names


def haralick_texture_features(image, mask, image_bg=None):
    if image_bg is not None:
        # We have to perform background subtraction (casting required).
        image_corr = np.array(image, dtype=int) - image_bg
    else:
        # Input is already background-corrected
        image_corr = image

    # make sure we have a boolean array
    mask = np.asarray(mask, dtype=bool)

    size = image.shape[0]

    ds_dt = np.dtype({'names': haralick_names,
                      'formats': [float] * len(haralick_names)})
    rec_arr = np.recarray(size, dtype=ds_dt)
    nan_result = np.nan * np.zeros(26, dtype=float)

    # This loop is parallelized with numba jit via `prange`.
    for ii in range(size):
        # Haralick texture features
        # https://gitlab.gwdg.de/blood_data_analysis/dcevent/-/issues/20
        # Preprocessing:
        # - create a copy of the array (don't edit `image_corr`)
        # - add grayscale values (negative values not supported)
        #   -> maximum value should be as small as possible
        # - set pixels outside contour to zero (ignored areas, see mahotas)
        minval = (image_corr[ii][mask[ii]]).min()
        imi = (image_corr[ii] - minval + 1) * mask[ii]
        try:
            ret = mh_haralick(imi,
                              ignore_zeros=True,
                              return_mean_ptp=True)
        except ValueError:
            # The error message looks like this:
            #    ValueError: mahotas.haralick_features: the input is empty.
            #    Cannot compute features! This can happen if you are
            #    using `ignore_zeros`.
            # The problem is that a co-occurrence matrix is all-zero (e.g.
            # if the mask is just a one-pixel horizontal line, then the
            # diagonal and vertical co-occurrence matrices do not have any
            # entries. We just catch the exception and return `nan`s.
            ret = nan_result
        # (1) Angular Second Moment
        rec_arr["tex_asm_avg"][ii] = ret[0]
        rec_arr["tex_asm_ptp"][ii] = ret[13]
        # (2) Contrast
        rec_arr["tex_con_avg"][ii] = ret[1]
        rec_arr["tex_con_ptp"][ii] = ret[14]
        # (3) Correlation
        rec_arr["tex_cor_avg"][ii] = ret[2]
        rec_arr["tex_cor_ptp"][ii] = ret[15]
        # (4) Variance
        rec_arr["tex_var_avg"][ii] = ret[3]
        rec_arr["tex_var_ptp"][ii] = ret[16]
        # (5) Inverse Difference Moment
        rec_arr["tex_idm_avg"][ii] = ret[4]
        rec_arr["tex_idm_ptp"][ii] = ret[17]
        # (6) Feature 6 "Sum Average", which is equivalent to
        # 2 * bright_bc_avg since dclab 0.44.0.
        # (7) Sum Variance
        rec_arr["tex_sva_avg"][ii] = ret[6]
        rec_arr["tex_sva_ptp"][ii] = ret[19]
        # (8) Sum Entropy
        rec_arr["tex_sen_avg"][ii] = ret[7]
        rec_arr["tex_sen_ptp"][ii] = ret[20]
        # (9) Entropy
        rec_arr["tex_ent_avg"][ii] = ret[8]
        rec_arr["tex_ent_ptp"][ii] = ret[21]
        # (10) Feature 10 "Difference Variance" is excluded, because it
        # has a functional dependency on the offset value (we use "1" here)
        # and thus is not really only describing texture.
        # (11) Difference Entropy
        rec_arr["tex_den_avg"][ii] = ret[10]
        rec_arr["tex_den_ptp"][ii] = ret[23]
        # (12) Information Measure of Correlation 1
        rec_arr["tex_f12_avg"][ii] = ret[11]
        rec_arr["tex_f12_ptp"][ii] = ret[24]
        # (13) Information Measure of Correlation 2
        rec_arr["tex_f13_avg"][ii] = ret[12]
        rec_arr["tex_f13_ptp"][ii] = ret[25]
        # (14) Feature 14 is excluded, because nobody is using it, it is
        # not understood by everyone what it actually is, and it is
        # computationally expensive.

    return rec_arr
