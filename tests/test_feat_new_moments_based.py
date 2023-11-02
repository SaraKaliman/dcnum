import pathlib

import h5py
import numpy as np

from dcnum.feat import feat_moments

data_path = pathlib.Path(__file__).parent / "data"


def test_moments_based_features():
    # This file has new cell features belonging to
    # fmt-hdf5_cytoshot_full-features_2023.zip
    file_name = ("new_feat_tests.hdf5")

    feats = [
            "area_um_raw",
            "per_um_raw",
            "deform_raw",
            "eccentr_prnc",
            "per_ratio",
            "s_x",
            "s_y",
    ]

    # Make data available
    with h5py.File(data_path / file_name) as h5:
        data = feat_moments.moments_based_features(
            mask=h5["events/mask"][:],
            pixel_size=0.2645
        )
        for feat in feats:
            if feat.count("inert"):
                rtol = 2e-5
                atol = 1e-8
            else:
                rtol = 1e-5
                atol = 1e-8
            assert np.allclose(h5["events"][feat][:],
                               data[feat],
                               rtol=rtol,
                               atol=atol), f"Feature {feat} mismatch!"


def test_mask_0d():
    masks = np.array([
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ], dtype=bool)[np.newaxis]
    data = feat_moments.moments_based_features(
                mask=masks,
                pixel_size=0.2645
            )
    assert data["deform_raw"].shape == (1,)
    assert np.isnan(data["deform_raw"][0])
    assert np.isnan(data["area_um_raw"][0])


def test_mask_1d():
    masks = np.array([
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ], dtype=bool)[np.newaxis]
    data = feat_moments.moments_based_features(
                mask=masks,
                pixel_size=0.2645
            )
    assert data["deform_raw"].shape == (1,)
    assert np.isnan(data["deform_raw"][0])
    assert np.isnan(data["area_um_raw"][0])


def test_mask_1d_large():
    masks = np.array([
        [0, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
    ], dtype=bool)[np.newaxis]
    data = feat_moments.moments_based_features(
                mask=masks,
                pixel_size=0.2645
            )
    assert data["deform_raw"].shape == (1,)
    assert np.isnan(data["deform_raw"][0])
    assert np.isnan(data["area_um_raw"][0])


def test_mask_1d_large_no_border():
    masks = np.array([
        [0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ], dtype=bool)[np.newaxis]
    data = feat_moments.moments_based_features(
                mask=masks,
                pixel_size=0.2645
            )
    assert data["deform_raw"].shape == (1,)
    assert np.isnan(data["deform_raw"][0])
    assert np.isnan(data["area_um_raw"][0])


def test_mask_2d():
    masks = np.array([
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 0],
        [0, 0, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ], dtype=bool)[np.newaxis]
    data = feat_moments.moments_based_features(
                mask=masks,
                pixel_size=0.2645
            )
    assert data["deform_raw"].shape == (1,)
    # This is the deformation of a square (compared to circle)
    assert np.allclose(data["deform_raw"][0], 0.11377307454724206)
    # Without moments-based computation, this would be 4*pxsize=0.066125
    assert np.allclose(data["area_um_raw"][0], 0.06996025)


def test_mask_mixed():
    mask_valid = np.array([
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 0],
        [0, 0, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ], dtype=bool)
    mask_invalid = np.array([
        [0, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
    ], dtype=bool)
    mixed_masks = np.append(mask_valid[None, ...],
                            mask_invalid[None, ...], axis=0)
    data = feat_moments.moments_based_features(
                mask=mixed_masks,
                pixel_size=0.2645)
    assert data["deform_raw"].shape == (2,)
    assert np.all(data["valid"][:] == np.array([True, False]))
    assert not np.isnan(data["deform_raw"][0])
    assert np.isnan(data["deform_raw"][1])
