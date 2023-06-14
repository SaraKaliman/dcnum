# flake8: noqa: F401
import functools

from .base import Background
# Background methods are registered by importing them here.
from .bg_roll_median import BackgroundRollMed
from .bg_sparse_median import BackgroundSparseMed


@functools.cache
def get_available_background_methods():
    """Return dictionary of background computation methods"""
    methods = {}
    for cls in Background.__subclasses__():
        methods[cls.key()] = cls
    return methods
