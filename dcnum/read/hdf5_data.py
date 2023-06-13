import functools
import warnings

import h5py

from .cache import HDF5ImageCache, ImageCorrCache, md5sum
from .const import PROTECTED_FEATURES


class HDF5Data:
    """HDF5 (.rtdc) input file data instance"""
    def __init__(self, path, pixel_size=None,
                 md5_5m=None, meta=None, logs=None, tables=None):
        # Init is in __setstate__ so we can pickle this class
        # and use it for multiprocessing.
        self.h5 = None  # is set in __setstate__
        self._cache_scalar = {}
        self.__setstate__({"path": path,
                           "pixel_size": pixel_size,
                           "md5_5m": md5_5m,
                           "meta": meta,
                           "logs": logs,
                           "tables": tables})

    def __contains__(self, item):
        return item in self.keys()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.h5.close()

    def __getitem__(self, feat):
        if feat == "image":
            return self.image
        elif feat == "image_bg":
            return self.image_bg
        elif feat == "mask" and self.mask is not None:
            return self.mask
        elif feat in self._cache_scalar:
            return self._cache_scalar[feat]
        elif len(self.h5["events"][feat].shape) == 1:
            self._cache_scalar[feat] = self.h5["events"][feat][:]
            return self._cache_scalar[feat]
        else:
            # Not cached (possibly slow)
            warnings.warn(f"Feature {feat} not cached (possibly slow)")
            return self.h5["events"][feat]

    def __setstate__(self, state):
        self.path = state["path"]

        self.md5_5m = state["md5_5m"]
        if self.md5_5m is None:
            # 5MB md5sum of input file
            self.md5_5m = md5sum(self.path, count=80)
        self.logs = state["logs"]
        self.tables = state["tables"]
        self.meta = state["meta"]
        if self.meta is None or self.logs is None or self.tables is None:
            self.logs = {}
            self.tables = {}
            # get dataset configuration
            with h5py.File(self.path) as h5:
                self.meta = dict(h5.attrs)
                for key in h5.get("logs", []):
                    alog = list(h5["logs"][key])
                    if isinstance(alog[0], bytes):
                        alog = [ll.decode("utf") for ll in alog]
                    self.logs[key] = alog
                for tab in h5.get("tables", []):
                    self.tables[tab] = h5["tables"][key][:]

        if state["pixel_size"] is not None:
            self.pixel_size = state["pixel_size"]
        else:
            # Set known pixel size if possible
            did = self.meta.get("setup:identifier", "EMPTY")
            if (did.startswith("RC-")
                    and (self.pixel_size < 0.255 or self.pixel_size > 0.275)):
                warnings.warn(
                    f"Correcting for invalid pixel size in '{self.path}'!")
                # Set default pixel size for Rivercyte devices
                self.pixel_size = 0.2645

        self.h5 = h5py.File(self.path, libver="latest")
        self.image = HDF5ImageCache(self.h5["events/image"])

        if "events/image_bg" in self.h5:
            self.image_bg = HDF5ImageCache(self.h5["events/image_bg"])
        else:
            self.image_bg = None

        if "events/mask" in self.h5:
            self.mask = HDF5ImageCache(self.h5["events/mask"],
                                       boolean=True)
        else:
            self.mask = None

        self.image_corr = ImageCorrCache(self.image, self.image_bg)

    def __getstate__(self):
        return {"path": self.path,
                "pixel_size": self.pixel_size,
                "md5_5m": self.md5_5m,
                "meta": self.meta,
                "logs": self.logs,
                "tables": self.tables,
                }

    @functools.cache
    def __len__(self):
        return self.h5.attrs["experiment:event count"]

    @property
    def meta_nest(self):
        """Return `self.meta` as nested dicitonary

        This gets very close to the dclab `config` property of datasets.
        """
        md = {}
        for key in self.meta:
            sec, var = key.split(":")
            md.setdefault(sec, {})[var] = self.meta[key]
        return md

    @property
    def pixel_size(self):
        return self.meta.get("imaging:pixel size", 0)

    @pixel_size.setter
    def pixel_size(self, pixel_size):
        self.meta["imaging:pixel size"] = pixel_size

    @functools.cache
    def keys(self):
        return sorted(self.h5["/events"].keys())

    @property
    @functools.cache
    def features_scalar_frame(self):
        """Scalar features that apply to all events in a frame

        This is a convenience function for copying scalar features
        over to new processed datasets. Return a list of all features
        that describe a frame (e.g. temperature or time).
        """
        feats = []
        for feat in self.h5["events"]:
            if feat in PROTECTED_FEATURES:
                feats.append(feat)
        return feats
