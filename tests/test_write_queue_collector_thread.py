import collections
import multiprocessing as mp
import pathlib

import numpy as np

from dcnum import read, write

from helper_methods import retrieve_data

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


def test_queue_collector_thread():
    # keyword arguments
    data = read.HDF5Data(
        retrieve_data("fmt-hdf5_cytoshot_full-features_2023.zip"))
    event_queue = mp.Queue()
    writer_dq = collections.deque()
    feat_nevents = np.array([1, 3, 1, 5])
    write_threshold = 2
    # queue collector thread
    qct = write.QueueCollectorThread(
        data=data,
        event_queue=event_queue,
        writer_dq=writer_dq,
        feat_nevents=feat_nevents,
        write_threshold=write_threshold
    )
    # put data into queue
    event_queue.put((0, {"deform": np.array([.1]),
                         "area_um": np.array([100])}))
    event_queue.put((1, {"deform": np.array([.1, .2, .3]),
                         "area_um": np.array([100, 120, 150])}))
    event_queue.put((2, {"deform": np.array([.1]),
                         "area_um": np.array([100])}))
    event_queue.put((3, {"deform": np.array([.1, .2, .3, .4, .5]),
                         "area_um": np.array([100, 110, 120, 130, 140])}))
    # collect information from queue (without the threading part)
    qct.run()
    # test whether everything is in order.
    # We have a write threshold of 2, so there should data in batches of two
    # frames stored n the writer_dq.

    # BATCH 1
    feat, deform1 = writer_dq.popleft()
    assert feat == "deform"
    assert np.all(deform1 == [.1, .1, .2, .3])
    feat, _ = writer_dq.popleft()
    assert feat == "area_um"
    for fexp in data.features_scalar_frame + ["image", "image_bg", "nevents"]:
        fact, _ = writer_dq.popleft()
        assert fexp == fact

    # BATCH 2
    feat, deform1 = writer_dq.popleft()
    assert feat == "deform"
    assert np.all(deform1 == [.1, .1, .2, .3, .4, .5])
    feat, _ = writer_dq.popleft()
    assert feat == "area_um"
    for fexp in data.features_scalar_frame + ["image", "image_bg", "nevents"]:
        fact, _ = writer_dq.popleft()
        assert fexp == fact

    assert len(writer_dq) == 0
