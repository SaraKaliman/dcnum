import multiprocessing as mp
import pathlib

from dcnum import segm
import h5py
import numpy as np
from skimage import morphology

import pytest

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

    sm = segm.segm_thresh.SegmentThresh(thresh=-6,
                                        kwargs_mask={"closing_disk": 3})
    for ii in range(len(frame_u)):
        labels_seg = sm.segment_frame(image_u_c[ii])
        mask_seg = np.array(labels_seg, dtype=bool)
        # Remove small objects, because this is not implemented in the
        # segmenter class as it would be part of gating.
        mask_seg = morphology.remove_small_objects(mask_seg, min_size=10)
        assert np.all(mask_seg == mask_u[ii]), f"masks not matching at {ii}"


@pytest.mark.parametrize("worker_type", ["thread", "process"])
def test_segm_thresh_segment_batch(worker_type):
    debug = worker_type == "thread"
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

    sm = segm.segm_thresh.SegmentThresh(thresh=-6,
                                        debug=debug,
                                        kwargs_mask={"closing_disk": 3})

    labels_seg = sm.segment_batch(image_u_c, start=0, stop=5)
    assert labels_seg is sm.labels_array
    assert np.all(np.array(labels_seg, dtype=bool) == sm.mask_array)
    # tell workers to stop
    sm.join_workers()

    for ii in range(len(frame_u)):
        mask_seg = np.array(labels_seg[ii], dtype=bool)
        # Remove small objects, because this is not implemented in the
        # segmenter class as it would be part of gating.
        mask_seg = morphology.remove_small_objects(mask_seg, min_size=10)
        assert np.all(mask_seg == mask_u[ii]), f"masks not matching at {ii}"


@pytest.mark.parametrize("worker_type", ["thread", "process"])
def test_segm_thresh_segment_batch_large(worker_type):
    debug = worker_type == "thread"

    # Create fake data
    mask = np.zeros((121, 80, 200), dtype=bool)
    mask[:, 10:71, 100:161] = morphology.disk(30).reshape(-1, 61, 61)
    image = -10 * mask

    sm = segm.segm_thresh.SegmentThresh(thresh=-6,
                                        kwargs_mask={"closing_disk": 3},
                                        debug=debug)

    labels_seg_1 = np.copy(
        sm.segment_batch(image, start=0, stop=101))

    assert labels_seg_1.dtype == np.uint16  # uint8 is not enough
    assert sm.mp_batch_index.value == 0
    if worker_type == "thread":
        assert len(sm._mp_workers) == 1
        assert sm.mp_batch_worker.value == 1
    else:
        # This will fail if you have too many CPUs in your system
        assert len(sm._mp_workers) == mp.cpu_count()
        # Check whether all processes did their deeds
        assert sm.mp_batch_worker.value == mp.cpu_count()

    labels_seg_2 = np.copy(
        sm.segment_batch(image, start=101, stop=121))

    # tell workers to stop
    sm.join_workers()

    for ii in range(101):
        mask_seg = np.array(labels_seg_1[ii], dtype=bool)
        assert np.all(mask_seg == mask[ii]), f"masks not matching at {ii}"

    for jj in range(101, 121):
        mask_seg = np.array(labels_seg_2[jj - 101], dtype=bool)
        assert np.all(mask_seg == mask[jj]), f"masks not matching at {jj}"


def test_segm_thresh_labeled_mask():
    mask = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0, 0],  # filled, 1
        [0, 0, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 1, 1, 1],  # border, 2
        [0, 0, 0, 0, 0, 1, 1, 1],
        [0, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 1, 1, 1, 0, 0, 0],  # other, 3
        [0, 0, 1, 1, 1, 0, 0, 0],
        [0, 0, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        ], dtype=bool)

    sm1 = segm.segm_thresh.SegmentThresh(thresh=-6,
                                         kwargs_mask={"clear_border": True,
                                                      "fill_holes": True,
                                                      "closing_disk": 0,
                                                      })
    labels1 = sm1.segment_frame(-10 * mask)
    assert np.sum(labels1 != 0) == 21
    assert len(np.unique(labels1)) == 3  # (bg, filled, other)
    assert np.sum(labels1 == 1) == 9
    # due to the relabeling done in `fill_holes`, the index of "other" is "3"
    assert np.sum(labels1 == 2) == 12

    sm2 = segm.segm_thresh.SegmentThresh(thresh=-6,
                                         kwargs_mask={"clear_border": True,
                                                      "fill_holes": False,
                                                      "closing_disk": 0,
                                                      })
    labels2 = sm2.segment_frame(-10 * mask)
    _, l2a, l2b = np.unique(labels2)
    assert np.sum(labels2 != 0) == 20
    assert len(np.unique(labels2)) == 3  # (bg, filled, other)
    assert np.sum(labels2 == l2a) == 8
    assert np.sum(labels2 == l2b) == 12

    sm3 = segm.segm_thresh.SegmentThresh(thresh=-6,
                                         kwargs_mask={"clear_border": False,
                                                      "fill_holes": False,
                                                      "closing_disk": 0,
                                                      })
    labels3 = sm3.segment_frame(-10 * mask)
    assert np.sum(labels3 != 0) == 30
    assert len(np.unique(labels3)) == 4  # (bg, filled, border, other)
    assert np.sum(labels3 == 1) == 8
    assert np.sum(labels3 == 2) == 10
    assert np.sum(labels3 == 3) == 12

    sm4 = segm.segm_thresh.SegmentThresh(thresh=-6,
                                         kwargs_mask={"clear_border": False,
                                                      "fill_holes": True,
                                                      "closing_disk": 0,
                                                      })
    labels4 = sm4.segment_frame(-10 * mask)
    assert np.sum(labels4 != 0) == 31
    assert len(np.unique(labels4)) == 4  # (bg, filled, border, other)
    assert np.sum(labels4 == 1) == 9
    assert np.sum(labels4 == 2) == 10
    assert np.sum(labels4 == 3) == 12


def test_segm_thresh_labeled_mask_closing_disk():
    mask = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0, 0, 0],  # filled, 1
        [0, 0, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 1, 1, 1],  # border, 2
        [0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 1, 1, 0, 0],  # other, 3
        [0, 0, 1, 0, 0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 0, 1, 0, 0],
        [0, 0, 1, 1, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        ], dtype=bool)

    sm1 = segm.segm_thresh.SegmentThresh(thresh=-6,
                                         kwargs_mask={"clear_border": True,
                                                      "fill_holes": True,
                                                      "closing_disk": 1,
                                                      })
    labels1 = sm1.segment_frame(-10 * mask)
    assert np.sum(labels1 != 0) == 32
    assert len(np.unique(labels1)) == 3  # (bg, filled, other)
    assert np.sum(labels1 == 1) == 9
    # due to the relabeling done in `fill_holes`, the index of "other" is "3"
    assert np.sum(labels1 == 2) == 23

    sm2 = segm.segm_thresh.SegmentThresh(thresh=-6,
                                         kwargs_mask={"clear_border": True,
                                                      "fill_holes": False,
                                                      "closing_disk": 1,
                                                      })
    labels2 = sm2.segment_frame(-10 * mask)
    _, l2a, l2b = np.unique(labels2)
    assert np.sum(labels2 != 0) == 27
    assert len(np.unique(labels2)) == 3  # (bg, filled, other)
    assert np.sum(labels2 == l2a) == 9
    assert np.sum(labels2 == l2b) == 18

    sm3 = segm.segm_thresh.SegmentThresh(thresh=-6,
                                         kwargs_mask={"clear_border": False,
                                                      "fill_holes": False,
                                                      "closing_disk": 1,
                                                      })
    labels3 = sm3.segment_frame(-10 * mask)
    assert np.sum(labels3 != 0) == 35
    assert len(np.unique(labels3)) == 4  # (bg, filled, border, other)
    assert np.sum(labels3 == 1) == 9
    assert np.sum(labels3 == 2) == 8
    assert np.sum(labels3 == 3) == 18

    sm4 = segm.segm_thresh.SegmentThresh(thresh=-6,
                                         kwargs_mask={"clear_border": False,
                                                      "fill_holes": True,
                                                      "closing_disk": 1,
                                                      })
    labels4 = sm4.segment_frame(-10 * mask)
    assert np.sum(labels4 != 0) == 40
    assert len(np.unique(labels4)) == 4  # (bg, filled, border, other)
    assert np.sum(labels4 == 1) == 9
    assert np.sum(labels4 == 2) == 8
    assert np.sum(labels4 == 3) == 23
