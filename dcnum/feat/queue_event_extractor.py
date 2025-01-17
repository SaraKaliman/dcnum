import collections
import logging
from logging.handlers import QueueHandler
import multiprocessing as mp
import os
import queue
import threading
import traceback

import numpy as np

from ..meta.ppid import kwargs_to_ppid
from ..read import HDF5Data

from .feat_brightness import brightness_features
from .feat_moments import moments_based_features
from .feat_texture import haralick_texture_features
from .gate import Gate


# All subprocesses should use 'spawn' to avoid issues with threads
# and 'fork' on POSIX systems.
mp_spawn = mp.get_context("spawn")


class QueueEventExtractor:
    def __init__(self,
                 data: HDF5Data,
                 gate: Gate,
                 preselect: bool,
                 ptp_median: float,
                 raw_queue: mp.Queue,
                 event_queue: mp.Queue,
                 log_queue: mp.Queue,
                 feat_nevents: mp.Array,
                 label_array: mp.Array,
                 finalize_extraction: mp.Value,
                 extract_kwargs: dict = None,
                 *args, **kwargs):
        """Base class for event extraction from label images

        This class is meant to be subclassed to run either in a
        :class:`threading.Thread` or a :class:`multiprocessing.Process`.

        Parameters
        ----------
        data:
            Data source.
        gate:
            Gating rules.
        preselect:
            Whether to perform data preselection based on peak-to-peak
            values in the images.
        ptp_median:
            Median peak-to-peak value in the images for preselction.
        raw_queue:
            Queue from which the worker obtains the chunks and
            indices of the labels to work on.
        event_queue:
            Queue in which the worker puts the extracted event feature
            data.
        log_queue:
            Logging queue, used for sending messages to the main Process.
        feat_nevents:
            Shared array of same length as data into which the number of
            events per input frame is written. This array must be initialized
            with -1 (all values minus one).
        label_array:
            Shared array containing the labels of one chunk from `data`.
        finalize_extraction:
            Shared value indicating whether this worker should stop as
            soon as the `raw_queue` is empty.
        extract_kwargs:
            Keyword arguments for the extraction process. See the
            keyword-only arguments in
            :func:`QueueEventExtractor.get_events_from_masks`.

        """
        super(QueueEventExtractor, self).__init__(*args, **kwargs)
        #: Data instance
        self.data = data
        #: Gating information
        self.gate = gate
        #: Whether to perform Preselection
        self.preselect = preselect
        #: Peak-to-peak median for preselection
        self.ptp_median = ptp_median
        #: queue containing sub-indices for `label_array`
        self.raw_queue = raw_queue
        #: queue with event-wise feature dictionaries
        self.event_queue = event_queue
        #: queue for logging
        self.log_queue = log_queue
        #: Shared array of length `len(data)` into which the number of
        #: events per frame is written.
        self.feat_nevents = feat_nevents
        #: Shared array containing the labels of one chunk from `data`.
        self.label_array = label_array
        #: Set to True to let worker join when `raw_queue` is empty.
        self.finalize_extraction = finalize_extraction
        # Keyword arguments for data extraction
        if extract_kwargs is None:
            extract_kwargs = {}
        extract_kwargs.setdefault("brightness", True)
        extract_kwargs.setdefault("haralick", True)
        #: Feature extraction keyword arguments.
        self.extract_kwargs = extract_kwargs
        # Logging needs to be set up after `start` is called, otherwise
        # it looks like we have the same PID as the parent process. We
        # are setting up logging in `run`.
        self.logger = None

    @staticmethod
    def get_init_kwargs(data, gate, preselect, ptp_median, log_queue):
        """You can pass `*args.values()` directly to __init__

        This method was created for convenience reasons:
        - It makes sure that the order of arguments is correct, since it
          is implemented in the same class.
        - It simplifies testing.
        """
        # queue with the raw (unsegmented) image data
        raw_queue = mp_spawn.Queue()
        # queue with event-wise feature dictionaries
        event_queue = mp_spawn.Queue()

        args = collections.OrderedDict()
        args["data"] = data
        args["gate"] = gate
        args["preselect"] = preselect
        args["ptp_median"] = ptp_median
        args["raw_queue"] = raw_queue
        args["event_queue"] = event_queue
        args["log_queue"] = log_queue
        args["feat_nevents"] = mp_spawn.Array("i", len(data))
        args["feat_nevents"][:] = np.full(len(data), -1)
        # "h" is signed short (np.int16)
        args["label_array"] = mp_spawn.RawArray(
            np.ctypeslib.ctypes.c_int16,
            int(np.product(data.image.chunk_shape)))
        args["finalize_extraction"] = mp_spawn.Value("b", False)
        return args

    @classmethod
    def get_ppid_from_kwargs(cls, kwargs):
        """Return the pipeline ID for this event extractor"""
        key = "legacy"
        cback = kwargs_to_ppid(cls, "get_events_from_masks", kwargs)
        return ":".join([key, cback])

    def get_events_from_masks(self, masks, data_index, *,
                              brightness: bool = True,
                              haralick: bool = True,
                              ):
        """Get events dictionary, performing event-based gating"""
        events = {"mask": masks}
        image = self.data.image[data_index][np.newaxis]
        image_bg = self.data.image_bg[data_index][np.newaxis]
        image_corr = self.data.image_corr[data_index][np.newaxis]

        events.update(
            moments_based_features(
                masks,
                pixel_size=self.data.pixel_size))
        if brightness:
            events.update(brightness_features(
                mask=masks, image=image, image_bg=image_bg,
                image_corr=image_corr
            ))
        if haralick:
            events.update(haralick_texture_features(
                mask=masks, image=image, image_corr=image_corr
            ))

        # gating on feature arrays
        if self.gate.box_gates:
            valid = self.gate.gate_events(events)
            gated_events = {}
            for key in events:
                gated_events[key] = events[key][valid]
        else:
            gated_events = events

        # removing events with invalid features
        valid_events = {}
        # the valid key-value pair was added in `moments_based_features` and
        # its only purpose is to mark events with invalid contours as such,
        # so they can be removed here. Resolves issue #9.
        valid = gated_events.pop("valid")
        invalid = ~valid
        # The following might lead to a computational overhead, if only a few
        # events are invalid, because then all 2d-features need to be copied
        # over from gated_events to valid_events. According to our experience
        # invalid events happen rarely though.
        if np.any(invalid):
            self.logger.info(f"Discarded {np.sum(invalid)} events due to "
                             "invalid segmentation.")
            for key in gated_events:
                valid_events[key] = gated_events[key][valid]
        else:
            valid_events = gated_events

        return valid_events

    def get_masks_from_label(self, label):
        """Get masks, performing mask-based gating"""
        # Using np.unique is a little slower than iterating over lmax
        # unu = np.unique(label)  # background is 0
        lmax = np.max(label)
        masks = []
        for jj in range(1, lmax+1):  # first item is 0
            mask_jj = label == jj
            mask_sum = np.sum(mask_jj)
            if mask_sum and self.gate.gate_mask(mask_jj, mask_sum=mask_sum):
                masks.append(mask_jj)
        return np.array(masks)

    def get_ppid(self):
        """Return a unique feature extractor pipeline identifier

        The pipeline identifier is universally applicable and must
        be backwards-compatible (future versions of dcevent will
        correctly acknowledge the ID).

        The feature extractor pipeline ID is defined as::

            KEY:KW_APPROACH

        Where KEY is e.g. "legacy", and KW_APPROACH is a
        list of keyword-only arguments for `get_events_from_masks`,
        e.g.::

            brightness=True^haralick=True

        which may be abbreviated to::

            b=1^h=1
        """
        return self.get_ppid_from_kwargs(self.extract_kwargs)

    def process_label(self, label, index):
        """Process one label image, extracting masks and features"""
        if (np.all(self.data.image[index - 1] == self.data.image[index])
                and index):
            # TODO: Do this before segmentation already?
            # skip events that have been analyzed already
            return None
        if self.preselect:
            # TODO: Do this before segmentation already?
            ptp = np.ptp(self.data.image_corr[index])
            if ptp < 0.1 * self.ptp_median:
                return None
        masks = self.get_masks_from_label(label)
        if masks.size:
            events = self.get_events_from_masks(
                masks=masks, data_index=index, **self.extract_kwargs)
        else:
            events = None
        return events

    def run(self):
        """Main loop of worker process"""
        # Don't wait for these two queues when joining workers
        self.raw_queue.cancel_join_thread()
        self.log_queue.cancel_join_thread()
        #: logger sends all logs to `self.log_queue`
        self.logger = logging.getLogger(
            f"dcnum.feat.EventExtractor.{os.getpid()}")
        queue_handler = QueueHandler(self.log_queue)
        self.logger.addHandler(queue_handler)
        self.logger.addFilter(DeduplicatingLoggingFilter())
        self.logger.debug(f"Running {self} in PID {os.getpid()}")

        mp_array = np.ctypeslib.as_array(
            self.label_array).reshape(self.data.image.chunk_shape)
        while True:
            try:
                chunk_index, label_index = self.raw_queue.get(timeout=.03)
                index = chunk_index * self.data.image.chunk_size + label_index
            except queue.Empty:
                if self.finalize_extraction.value:
                    # The manager told us that there is nothing more coming.
                    self.logger.debug(
                        f"Finalizing worker {self} with PID {os.getpid()}. "
                        f"{self.event_queue.qsize()} events are still queued.")
                    break
            else:
                try:
                    events = self.process_label(
                        label=mp_array[label_index],
                        index=index)
                except BaseException:
                    self.logger.error(traceback.format_exc())
                else:
                    if events:
                        key0 = list(events.keys())[0]
                        self.feat_nevents[index] = len(events[key0])
                    else:
                        self.feat_nevents[index] = 0
                    self.event_queue.put((index, events))

        self.logger.debug(f"Finalizing `run` for PID {os.getpid()}, {self}")
        # Explicitly close the event queue and join it
        self.event_queue.close()
        self.event_queue.join_thread()
        self.logger.debug(f"End of `run` for PID {os.getpid()}, {self}")
        # Also close the logging queue. Not that not all messages might
        # arrive in the logging queue, since we called `cancel_join_thread`
        # earlier.
        self.log_queue.close()


class EventExtractorProcess(QueueEventExtractor, mp_spawn.Process):
    """Multiprocessing worker for regular segmentation and extraction"""
    def __init__(self, *args, **kwargs):
        super(EventExtractorProcess, self).__init__(
            name="EventExtractorProcess", *args, **kwargs)


class EventExtractorThread(QueueEventExtractor, threading.Thread):
    """Threading worker for debugging (only one single thread)"""
    def __init__(self, *args, **kwargs):
        super(EventExtractorThread, self).__init__(
            name="EventExtractorThread", *args, **kwargs)


class DeduplicatingLoggingFilter(logging.Filter):
    def __init__(self, *args, **kwargs):
        super(DeduplicatingLoggingFilter, self).__init__(*args, **kwargs)
        self._records = []

    def filter(self, record):
        """Return True if the record should be logged"""
        msg = record.getMessage()
        logged = msg in self._records
        if not logged:
            self._records.append(msg)
        return not logged
