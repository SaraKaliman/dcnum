import pathlib

import h5py

from dcnum import write

from helper_methods import retrieve_data

data_path = pathlib.Path(__file__).parent / "data"


def test_writer_basic():
    path = retrieve_data(data_path /
                         "fmt-hdf5_cytoshot_full-features_2023.zip")
    path_wrt = path.with_name("written.hdf5")
    with h5py.File(path) as h5, write.HDF5Writer(path_wrt) as hw:
        deform = h5["events"]["deform"][:]
        image = h5["events"]["image"][:]

        hw.store_feature_chunk(feat="deform", data=deform)
        hw.store_feature_chunk(feat="deform", data=deform)
        hw.store_feature_chunk(feat="deform", data=deform[:10])

        hw.store_feature_chunk(feat="image", data=image)
        hw.store_feature_chunk(feat="image", data=image)
        hw.store_feature_chunk(feat="image", data=image[:10])

    with h5py.File(path_wrt) as ho:
        events = ho["events"]
        size = deform.shape[0]
        assert events["deform"].shape[0] == 2*size + 10
        assert events["image"].shape[0] == 2 * size + 10
        assert events["image"].shape[1:] == image.shape[1:]
