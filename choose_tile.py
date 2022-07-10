import numpy as np
import cv2

class ChooseTile:

    def __init__(self, big_image: np.array, small_images: list[np.array], tilingOption: str, gray):
        self.big_image = big_image
        self.small_images = small_images
        self.tilingOption = tilingOption
        self.gray = gray
        assert(tilingOption == "average" or tilingOption == "use all")
    
    def quantize2index(self):
        if self.gray:
            gray = cv2.cvtColor(self.big_image, cv2.COLOR_BGR2GRAY)
            # convert to numpy
            gray = np.float32(gray)
            if self.tilingOption == "average":
                return self.quantize2index_avg(gray)
            elif self.tilingOption == "use all":
                return self.quantize2index_use_all(gray)
        else: 
            raise NotImplementedError

    def quantize2index_avg(self, gray):
        values = np.array([np.average(img) for img in self.small_images])
        # scale values to between 0 and 255
        values = (values - values.min())*255 / (values.max() - values.min())
        quantized = np.abs(gray[np.newaxis, ...] - values.reshape(-1, 1, 1)).argmin(axis=0)
        return np.uint8(quantized)

    def quantize2index_use_all(self, gray):
        raise NotImplementedError