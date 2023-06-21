import pathlib

import h5py
import numpy as np

from dcnum.feat import feat_texture

from helper_methods import retrieve_data

data_path = pathlib.Path(__file__).parent / "data"


def test_basic_haralick():
    # This original file was generated with dcevent for reference.
    path = retrieve_data(data_path /
                         "fmt-hdf5_cytoshot_full-features_2023.zip")
    # Make data available
    with h5py.File(path) as h5:
        ret_arr = feat_texture.haralick_texture_features(
            image=h5["events/image"][:],
            image_bg=h5["events/image_bg"][:],
            mask=h5["events/mask"][:],
        )

        assert np.allclose(ret_arr["tex_asm_avg"][1],
                           0.001514295993357114,
                           atol=0, rtol=1e-10)
        for feat in feat_texture.haralick_names:
            assert np.allclose(h5["events"][feat],
                               ret_arr[feat])
        # control test
        assert not np.allclose(h5["events"]["tex_asm_avg"],
                               ret_arr["tex_asm_ptp"])
