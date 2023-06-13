import pathlib

from dcnum import read
import h5py
import numpy as np
import pytest

from helper_methods import retrieve_data

data_path = pathlib.Path(__file__).parent / "data"


@pytest.mark.parametrize("path_out", [None, True])
def test_concat_basic(path_out):
    path = retrieve_data(data_path /
                         "fmt-hdf5_cytoshot_full-features_2023.zip")
    # create simple concatenated dataset, repeating a file
    data = read.concatenated_hdf5_data([path, path, path],
                                       path_out=path_out)
    assert len(data) == 120


@pytest.mark.parametrize("path_out", [None, True])
def test_concat_basic_frame(path_out):
    path = retrieve_data(data_path /
                         "fmt-hdf5_cytoshot_full-features_2023.zip")
    # create simple concatenated dataset, repeating a file
    data = read.concatenated_hdf5_data([path, path, path],
                                       path_out=path_out)
    with h5py.File(path) as h5:
        frame = h5["events/frame"][:]
    assert frame[0] == 101

    assert np.allclose(data["frame"][:frame.size],
                       frame - 101 + 1)
    offset1 = frame[-1] - 101 + 1
    assert np.allclose(offset1, data["frame"][frame.size-1])
    assert np.allclose(offset1 + 1, data["frame"][frame.size])
    assert np.allclose(data["frame"][frame.size:2*frame.size],
                       frame - 101 + offset1 + 1)
    diff = frame[-1] - frame[0]
    assert np.allclose(data["frame"][-1], 3 * (diff + 1))


def test_concat_basic_to_file(tmp_path):
    path = retrieve_data(data_path /
                         "fmt-hdf5_cytoshot_full-features_2023.zip")
    # create simple concatenated dataset, repeating a file
    path_out = tmp_path / "test.rtdc"
    assert not path_out.exists()
    data = read.concatenated_hdf5_data([path, path, path],
                                       path_out=path_out)
    assert len(data) == 120
    assert path_out.exists()
