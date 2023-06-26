import abc
import pathlib

import numpy as np


from .segmenter import Segmenter


class GPUSegmenter(Segmenter, abc.ABC):
    mask_postprocessing = False

    def __init__(self, model_file, *args, **kwargs):
        super(GPUSegmenter, self).__init__(*args, **kwargs)
        self.model_path = self._get_model_path(model_file)

    @staticmethod
    def _get_model_path(model_file):
        """Custom hook that may be defined by subclasses"""
        return pathlib.Path(model_file)

    def segment_batch(self,
                      image_data: np.ndarray,
                      start: int = None,
                      stop: int = None):
        if stop is None or start is None:
            start = 0
            stop = len(image_data)

        image_slice = image_data[start:stop]
        segm = self.segment_frame_wrapper(self.model_path)

        return segm(image_slice)
