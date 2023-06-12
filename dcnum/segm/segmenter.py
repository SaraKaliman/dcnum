import abc
import functools
import inspect
import logging
import multiprocessing as mp
import time
import threading
from typing import Tuple

import cv2
import numpy as np
from skimage import measure, morphology

from ..meta.ppid import kwargs_to_ppid


class Segmenter(abc.ABC):
    #: Whether to enable mask post-processing. If disabled, you should
    #: make sure that your mask is properly defined and cleaned or you
    #: have to call `process_mask` in your `segment_approach` implementation.
    mask_postprocessing = False
    #: Default keyword arguments for mask post-processing. See `process_mask`
    #: for available options.
    mask_default_kwargs = {}
    #: If the segmenter requires a background image, set this to True
    requires_background = False

    def __init__(self, kwargs_mask=None, debug=False, **kwargs):
        """Base segemnter

        Parameters
        ----------
        data: HDF5Data
            Instance containing the raw data. Requires at least the
            `image` and `image_bg` attributes. Some segemnters require
            more properties, so make sure to use :class:`.HDF5Data`.
        kwargs_mask: dict
            Keyword arguments for mask post-processing (see `process_mask`)
        """
        self.debug = debug
        self.logger = logging.getLogger(__name__).getChild(
            self.__class__.__name__)
        spec = inspect.getfullargspec(self.segment_approach)
        #: custom keyword arguments for the subclassing segmenter
        self.kwargs = spec.kwonlydefaults or {}
        self.kwargs.update(kwargs)

        #: keyword arguments for mask post-processing
        self.kwargs_mask = {}
        if self.mask_postprocessing:
            spec_mask = inspect.getfullargspec(self.process_mask)
            self.kwargs_mask.update(spec_mask.kwonlydefaults or {})
            self.kwargs_mask.update(self.mask_default_kwargs)
            if kwargs_mask is not None:
                self.kwargs_mask.update(kwargs_mask)
        elif kwargs_mask:
            raise ValueError(
                "`kwargs_mask` has been specified, but mask post-processing "
                f"is disabled for segmenter {self.__class__}")

    @classmethod
    def key(cls):
        """The unique key/name of this segmenter class"""
        key = cls.__name__.lower()
        if key.startswith("segment"):
            key = key[7:]
        return key

    def get_ppid(self):
        """Return a unique segmentation pipeline identifier

        The pipeline identifier is universally applicable and must
        be backwards-compatible (future versions of dcevent will
        correctly acknowledge the ID).

        The segmenter pipeline ID is defined as::

            KEY:KW_APPROACH:KW_MASK

        Where KEY is e.g. "legacy" or "watershed", and KW_APPROACH is a
        list of keyword arguments for `segment_approach`, e.g.::

            thresh=-6^blur=0

        which may be abbreviated to::

            t=-6^b=0

        KW_MASK represents keyword arguments for `process_mask`.
        """
        return self.get_ppid_from_kwargs(self.kwargs, self.kwargs_mask)

    @classmethod
    def get_ppid_from_kwargs(cls, kwargs, kwargs_mask=None):
        """Return the pipeline ID from given keyword arguments

        See Also
        --------
        get_ppid: Same method for class instances
        """
        if kwargs_mask is None and kwargs.get("kwargs_mask", None) is None:
            raise KeyError("`kwargs_mask` must be either specified as "
                           "keyword argument to this method or as a key "
                           "in `kwargs`!")
        if kwargs_mask is None:
            # see check above (kwargs_mask may also be {})
            kwargs_mask = kwargs["kwargs_mask"]
        # Start with the default mask kwargs defined for this subclass
        kwargs_mask_used = cls.mask_default_kwargs
        kwargs_mask_used.update(kwargs_mask)
        key = cls.key()
        csegm = kwargs_to_ppid(cls, "segment_approach", kwargs)
        cmask = kwargs_to_ppid(cls, "process_mask", kwargs_mask_used)
        return ":".join([key, csegm, cmask])

    @staticmethod
    @functools.cache
    def get_disk(radius):
        """Cached `skimage.morphology.disk(radius)`"""
        return morphology.disk(radius)

    @staticmethod
    def process_mask(mask, *,
                     clear_border: bool = True,
                     fill_holes: bool = True,
                     closing_disk: int = 5):
        """Post-process retrieved mask image

        This is an optional convenience method that is called for each
        subclass individually. To enable mask post-processing, set
        `mask_postprocessing=True` in the subclass and specify default
        `mask_default_kwargs`.

        Parameters
        ----------
        mask: 2d boolean ndarray
        clear_border: bool
            clear the image boarder using
            :func:`skimage.segmentation.clear_border`
        fill_holes: bool
            binary-fill-holes in the binary mask image using
            :func:`scipy.ndimage.binary_fill_holes`
        closing_disk: int or None
            if > 0, perform a binary closing with a disk
            of that radius in pixels
        """
        if clear_border:
            #
            # from skimage import segmentation
            # segmentation.clear_border(mask, out=mask)
            #
            if (mask[0, :].sum() or mask[-1, :].sum()
                    or mask[:, 0].sum() or mask[:, -1].sum()):
                border = np.zeros_like(mask)
                border[0] = True
                border[-1] = True
                border[:, 0] = True
                border[:, -1] = True
                label = measure.label(mask)
                indices = sorted(np.unique(label[border]))
                for ii in indices[1:]:
                    mask[label == ii] = False

        # scikit-image is too slow for us here. So we use OpenCV.
        # https://github.com/scikit-image/scikit-image/issues/1190
        mask_int = np.array(mask, dtype=np.uint8) * 255

        if closing_disk:
            #
            # from skimage import morphology
            # morphology.binary_closing(
            #    mask,
            #    footprint=morphology.disk(closing_disk),
            #    out=mask)
            #
            element = Segmenter.get_disk(closing_disk)
            mask_dilated = cv2.dilate(mask_int, element)
            mask_int = cv2.erode(mask_dilated, element)

        if fill_holes:
            #
            # from scipy import ndimage
            # mask_old = ndimage.binary_fill_holes(mask)
            #
            # Floodfill algorithm fills the background image and
            # the resulting inversion is the image with holes filled.
            im_floodfill = mask_int.copy()
            h, w = mask_int.shape
            ff_mask = np.zeros((h + 2, w + 2), np.uint8)
            cv2.floodFill(im_floodfill, ff_mask, (0, 0), 255)
            im_floodfill_inv = cv2.bitwise_not(im_floodfill)
            mask_int = mask_int | im_floodfill_inv

        mask = mask_int > 0
        return mask

    def segment_frame(self, image):
        """Return the frame mask for one image at `index`"""
        segm_wrap = self.segment_frame_wrapper()
        # obtain mask
        mask = segm_wrap(image)
        # optional postprocessing
        if self.mask_postprocessing:
            mask = self.process_mask(mask, **self.kwargs_mask)
        return mask

    @functools.cache
    def segment_frame_wrapper(self):
        if self.kwargs:
            # For segmenters that accept keyword arguments.
            segm_wrap = functools.partial(self.segment_approach,
                                          **self.kwargs)
        else:
            # For segmenters that don't accept keyword arguments.
            segm_wrap = self.segment_approach
        return segm_wrap

    @staticmethod
    @abc.abstractmethod
    def segment_approach(image, image_bg):
        """Perform segmentation and return binary mask

        This is the approach the subclasses implements.
        """

    @abc.abstractmethod
    def segment_batch(self, data, start, stop):
        """Return the frame mask array for an entire batch"""


class CPUSegmenter(Segmenter, abc.ABC):
    #: Number of segmenter processes to use.
    num_processes = None

    def __init__(self, *args, **kwargs):
        super(CPUSegmenter, self).__init__(*args, **kwargs)
        self._image_raw = None
        self._image_np = None
        self._mask_raw = None
        self._mask_np = None
        self._workers = []
        # Processing control values
        # <-1: exit
        # -1: idle
        # 0: start
        # >0: this number workers finished a batch
        self._status_batch = mp.Value("i", -1)
        # The iteration of the process (increment to wake workers)
        self._status_iter = mp.Value("i", -1)

    def __getstate__(self):
        # Copy the object's state from self.__dict__ which contains
        # all our instance attributes. Always use the dict.copy()
        # method to avoid modifying the original state.
        state = self.__dict__.copy()
        # Remove the unpicklable entries.
        del state['_image_raw']
        del state['_image_np']
        del state['_mask_raw']
        del state['_mask_np']
        del state['_workers']
        del state['_status_batch']
        del state['_status_iter']
        return state

    def __setstate__(self, state):
        # Restore instance attributes (i.e., filename and lineno).
        self.__dict__.update(state)

    @staticmethod
    def _create_shared_array(image_shape, batch_size, dtype):
        """Return raw and numpy-view on shared array

        Parameters
        ----------
        image_shape: tuple of int
            Shape of one single image in the array
        batch_size: int
            Number of images in the array
        dtype:
            ctype, e.g. `np.ctypeslib.ctypes.c_int8`
            or `np.ctypeslib.ctypes.c_bool`
        """
        sx, sy = image_shape
        sa_raw = mp.RawArray(dtype, int(sx * sy * batch_size))
        # Convert the RawArray to something we can write to fast
        # (similar to memory view, but without having to cast) using
        # np.ctypeslib.as_array. See discussion in
        # https://stackoverflow.com/questions/37705974
        sa_np = np.ctypeslib.as_array(sa_raw).reshape(batch_size, sx, sy)
        return sa_raw, sa_np

    @property
    def image_array(self):
        return self._image_np

    @property
    def mask_array(self):
        return self._mask_np

    def segment_batch(self, image_data, start, stop, debug=False):
        batch_size = stop - start
        image_shape = image_data[0].shape
        size = np.prod(image_data[0]) * batch_size

        if self._image_np.size != size:
            # reset image data
            self._image_np = None
            self._mask_np = None

        if self._image_np is None:
            self._image_raw, self._image_np = self._create_shared_array(
                image_shape=image_shape,
                batch_size=batch_size,
                dtype=np.ctypeslib.ctypes.c_int8,
            )
        if self._mask_np is None:
            self._mask_raw, self._mask_np = self._create_shared_array(
                image_shape=image_shape,
                batch_size=batch_size,
                dtype=np.ctypeslib.ctypes.c_bool,
            )

        # populate image data
        self._image_np[:] = image_data[start:stop]

        # Create the workers
        if debug:
            worker_cls = CPUSegmenterWorkerThread
            num_workers = 1
        else:
            worker_cls = CPUSegmenterWorkerProcess
            num_workers = self.num_processes

        if not self._workers:
            worker_args = [self,
                           self._status_batch,
                           self._status_iter,
                           self._image_raw,
                           self._mask_raw,
                           image_shape,
                           ]

            step_size = size // num_workers
            rest = size % 20
            wstart = 0
            for ii in range(num_workers):
                # Give every worker the same-sized workload and add one
                # from the rest until there is no more.
                wstop = wstart + step_size
                if rest:
                    wstop += 1
                    rest -= 1
                args = worker_args + [wstart, wstop]
                w = worker_cls(*args)
                w.start()
                self._workers.append(w)
                wstart = wstop

        # match iteration number with workers
        self._status_iter.value += 1

        # Tell workers to get going
        self._status_batch.value = 0

        while self._status_batch.value != num_workers:
            time.sleep(.1)

        return self._mask_np


class CPUSegmenterWorker:
    def __init__(self,
                 segmenter: CPUSegmenter,
                 status_batch: mp.Value,
                 status_iter: mp.Value,
                 image_data: mp.RawArray,
                 mask_data: mp.RawArray,
                 image_shape: Tuple[int, int],
                 start: int,
                 stop: int,
                 ):
        self.segmenter = segmenter
        self.status_batch = status_batch
        self.status_iter = status_iter
        self.image_data = np.ctypeslib.as_array(image_data).reshape(
            -1, image_shape[0], image_shape[1])
        self.mask_data = np.ctypeslib.as_array(mask_data).reshape(
            -1, image_shape[0], image_shape[1])
        self.image_shape = image_shape
        self.start = start
        self.stop = stop

    def run(self):
        idx = self.start
        itr = 0  # current iteration
        bsize = self.stop - self.start
        while True:
            correct_iter = self.status_iter.value == itr
            status_batch = self.status_batch.value
            if status_batch.value >= 0 and correct_iter:
                self.mask_data[idx] = self.segmenter.segment_frame(
                    self.image_data[idx])
                idx += 1
                if bsize == idx:
                    # We finished processing everything.
                    itr += 1  # prevent running same things again
                    idx = self.start  # reset counter for next batch
                    with self.status_iter:
                        self.status_iter.value += 1
            elif status_batch == -1:
                # Wait for more data to arrive
                time.sleep(.03)
            elif status_batch < -1:
                break


class CPUSegmenterWorkerProcess(CPUSegmenterWorker, mp.Process):
    def __init__(self, *args):
        super(CPUSegmenterWorkerProcess, self).__init__(*args)


class CPUSegmenterWorkerThread(CPUSegmenterWorker, threading.Thread):
    def __init__(self, *args):
        super(CPUSegmenterWorkerThread, self).__init__(*args)
