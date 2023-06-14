import collections
import functools
import hashlib
import pathlib

import h5py
import numpy as np


class HDF5ImageCache:
    def __init__(self,
                 h5ds: h5py.Dataset,
                 chunk_size: int = 1000,
                 cache_size: int = 5,
                 boolean: bool = False):
        """An HDF5 image cache

        Deformability cytometry data files commonly contain image stacks
        that are chunked in various ways. Loading just a single image
        can be time-consuming, because an entire HDF5 chunk has to be
        loaded, decompressed and from that one image extracted. The
        `HDF5ImageCache` class caches the chunks from the HDF5 files
        into memory, making single-image-access very fast.
        """
        # TODO:
        # - adjust chunking to multiples of the chunks in the dataset
        #   (which will slightly speed up things)
        self.h5ds = h5ds
        self.chunk_size = chunk_size
        self.boolean = boolean
        self.cache_size = cache_size
        #: This is a FILO cache for the chunks
        self.cache = collections.OrderedDict()
        self.shape = h5ds.shape
        self._len = len(self.h5ds)

    def _get_chunk_index_for_index(self, index):
        if index < 0:
            index = len(self.h5ds) + index
        chunk_index = index // self.chunk_size
        sub_index = index % self.chunk_size
        return chunk_index, sub_index

    def __getitem__(self, index):
        chunk_index, sub_index = self._get_chunk_index_for_index(index)
        return self.get_chunk(chunk_index)[sub_index]

    def __len__(self):
        return self._len

    def get_chunk(self, chunk_index):
        """Get one chunk of images"""
        if chunk_index not in self.cache:
            fslice = slice(self.chunk_size * chunk_index,
                           self.chunk_size * (chunk_index + 1)
                           )
            data = self.h5ds[fslice]
            if self.boolean:
                data = np.array(data, dtype=bool)
            self.cache[chunk_index] = data
            if len(self.cache) > self.cache_size:
                # Remove the first item
                self.cache.popitem(last=False)
        return self.cache[chunk_index]

    def iter_chunks(self):
        size = self.h5ds.shape[0]
        index = 0
        chunk = 0
        while True:
            yield chunk
            chunk += 1
            index += self.chunk_size
            if index >= size:
                break


class ImageCorrCache:
    def __init__(self,
                 image: HDF5ImageCache,
                 image_bg: HDF5ImageCache):
        self.image = image
        self.image_bg = image_bg
        self.chunk_size = image.chunk_size
        self.h5ds = image.h5ds
        self.shape = image.shape
        #: This is a FILO cache for the corrected image chunks
        self.cache = collections.OrderedDict()
        self.cache_size = image.cache_size

    def _get_chunk_index_for_index(self, index):
        if index < 0:
            index = len(self.h5ds) + index
        chunk_index = index // self.chunk_size
        sub_index = index % self.chunk_size
        return chunk_index, sub_index

    def __getitem__(self, index):
        chunk_index, sub_index = self._get_chunk_index_for_index(index)
        return self.get_chunk(chunk_index)[sub_index]

    def __len__(self):
        return len(self.image)

    def get_chunk(self, chunk_index):
        if chunk_index not in self.cache:
            data = np.array(self.image.get_chunk(chunk_index), dtype=int) \
                - self.image_bg.get_chunk(chunk_index)
            self.cache[chunk_index] = data
            if len(self.cache) > self.cache_size:
                # Remove the first item
                self.cache.popitem(last=False)
        return self.cache[chunk_index]

    def iter_chunks(self):
        return self.image.iter_chunks()


@functools.cache
def md5sum(path, blocksize=65536, count=0):
    """Compute (partial) MD5 sum of a file

    Parameters
    ----------
    path: str or pathlib.Path
        path to the file
    blocksize: int
        block size in bytes read from the file
        (set to `0` to hash the entire file)
    count: int
        number of blocks read from the file
    """
    path = pathlib.Path(path)

    hasher = hashlib.md5()
    with path.open('rb') as fd:
        ii = 0
        while len(buf := fd.read(blocksize)) > 0:
            hasher.update(buf)
            ii += 1
            if count and ii == count:
                break
    return hasher.hexdigest()
