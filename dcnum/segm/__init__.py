# flake8: noqa: F401
import functools

from .segmenter import Segmenter
from .segmenter_cpu import CPUSegmenter
from .segmenter_gpu import GPUSegmenter
from .segmenter_manager_thread import SegmenterManagerThread
from . import segm_thresh


@functools.cache
def get_available_segmenters():
    """Return dictionary of available segmenters"""
    segmenters = {}
    for scls in Segmenter.__subclasses__():
        for cls in scls.__subclasses__():
            segmenters[cls.key()] = cls
    return segmenters
