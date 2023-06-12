import pathlib

import numpy as np
from skimage import morphology

from dcnum import segm
import h5py

from helper_methods import retrieve_data

data_path = pathlib.Path(__file__).parent / "data"


def test_segm_thresh_basic():
    """Basic thresholding segmenter

    The segmenter is equivalent to the old dcevent legacy segmenter with
    the options legacy:t=-6^bl=0^bi=0^d=1:cle=1^f=1^clo=3
    (no blur, no binaryops, clear borders, fill holes, closing disk 3).
    Since in the dcevent pipeline, the data are gated and small objects
    are removed, we have to do this here manually before comparing mask
    images.
    """
    path = retrieve_data(
        data_path / "fmt-hdf5_cytoshot_full-features_legacy_allev_2023.zip")

    # Get all the relevant information
    with h5py.File(path) as h5:
        image = h5["events/image"][:]
        image_bg = h5["events/image_bg"][:]
        mask = h5["events/mask"][:]
        frame = h5["events/frame"][:]

    # Concatenate the masks
    frame_u, indices = np.unique(frame, return_index=True)
    image_u = image[indices]
    image_bg_u = image_bg[indices]
    mask_u = np.zeros_like(image_u, dtype=bool)
    for ii, fr in enumerate(frame):
        idx = np.where(frame_u == fr)[0]
        mask_u[idx] = np.logical_or(mask_u[idx], mask[ii])

    image_u_c = np.array(image_u, dtype=int) - image_bg_u

    sm = segm.SegmentThresh(thresh=-6, kwargs_mask={"closing_disk": 3})
    for ii in range(len(frame_u)):
        mask_seg = sm.segment_frame(image_u_c[ii])
        # Remove small objects, because this is not implemented in the
        # segmenter class as it would be part of gating.
        mask_seg = morphology.remove_small_objects(mask_seg, min_size=10)
        assert np.all(mask_seg == mask_u[ii]), f"masks not matching at {ii}"
