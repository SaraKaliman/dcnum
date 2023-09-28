import pathlib

import numpy as np

from dcnum import write

data_path = pathlib.Path(__file__).parent / "data"


def test_event_stash():
    feat_nevents = [1, 3, 1, 5]
    stash = write.EventStash(index_offset=0,
                             feat_nevents=feat_nevents)
    assert stash.size == 10
    assert np.all(stash.nev_idx == [1, 4, 5, 10])
    assert stash.num_frames == 4
    assert not stash.is_complete()
    stash.add_events(index=0,
                     events={"deform": np.array([.1]),
                             "area_um": np.array([100])})
    assert not stash.is_complete()
    stash.add_events(index=1,
                     events={"deform": np.array([.1, .2, .3]),
                             "area_um": np.array([100, 120, 150])})
    assert not stash.is_complete()
    stash.add_events(index=2,
                     events={"deform": np.array([.1]),
                             "area_um": np.array([100])})
    assert not stash.is_complete()
    stash.add_events(index=3,
                     events={"deform": np.array([.1, .2, .3, .4, .5]),
                             "area_um": np.array([100, 110, 120, 130, 140])})
    assert stash.is_complete()

    assert np.all(stash.events["deform"]
                  == [.1, .1, .2, .3, .1, .1, .2, .3, .4, .5])
    assert np.all(stash.events["area_um"]
                  == [100, 100, 120, 150, 100, 100, 110, 120, 130, 140])
