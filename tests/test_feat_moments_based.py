import pathlib

import h5py
import numpy as np

from dcnum.feat import feat_moments

from helper_methods import retrieve_data

data_path = pathlib.Path(__file__).parent / "data"


def test_moments_based_features():
    # This original file was generated with dcevent for reference.
    path = retrieve_data(data_path /
                         "fmt-hdf5_cytoshot_full-features_2023.zip")
    feats = [
        "deform",
        "size_x",
        "size_y",
        "pos_x",
        "pos_y",
        "area_msd",
        "area_ratio",
        "area_um",
        "aspect",
        "tilt",
        "inert_ratio_cvx",
        "inert_ratio_raw",
        "inert_ratio_prnc",
    ]

    # Make data available
    with h5py.File(path) as h5:
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
        # control test
        assert not np.allclose(h5["events"]["inert_ratio_cvx"][:],
                               data["tilt"])


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
    assert data["deform"].shape == (1,)
    assert np.isnan(data["deform"][0])
    assert np.isnan(data["area_um"][0])


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
    assert data["deform"].shape == (1,)
    assert np.isnan(data["deform"][0])
    assert np.isnan(data["area_um"][0])


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
    assert data["deform"].shape == (1,)
    # This is the deformation of a square (compared to circle)
    assert np.allclose(data["deform"][0], 0.11377307454724206)
    # Without moments-based computation, this would be 4*pxsize=0.066125
    assert np.allclose(data["area_um"][0], 0.06996025)
