# flake8: noqa: F401
import functools

from .segmenter import Segmenter
from .segm_thresh import SegmentThresh


@functools.cache
def get_segmentation_methods():
    """Return dictionary of segmentation methods"""
    global_dict = globals()
    methods = {}
    for key in global_dict:
        if key == "Segmenter":
            continue
        if key.startswith("Segment"):
            meth = global_dict[key]
            methods[meth.key()] = meth
    return methods
