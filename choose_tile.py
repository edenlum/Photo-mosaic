import numpy as np
import cv2

class ChooseTile:

    def __init__(self, big_image: np.array, small_images: list[np.array], tilingOption: str, gray, upscale):
        self.big_image = big_image
        self.small_images = small_images
        self.tilingOption = tilingOption
        self.gray = gray
        self.upscale = upscale
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
        hs, ws = self.small_images[0].shape
        # calc avg of tiles in big image (resizing)
        h, w = gray.shape
        # new height and width are the sizes if every small image is 1 pixel. i.e 
        # if the big image is 10 small images wide, we resize to 10*(small image width)
        nh, nw = round(h * self.upscale / hs), round(w * self.upscale/ ws) 
        gray = cv2.resize(gray, (nw * ws, nh * hs))
        gray = gray.reshape(nh, hs, nw, ws).swapaxes(1, 2) # now (nh, nw, hs, ws)
        
        # average out the tiles in the image
        gray = np.average(gray.reshape(nh, nw, hs*ws), axis=2)
        # calc avg of small images
        values = np.array([np.average(img) for img in self.small_images])
        # scale values to between 0 and 255
        values = (values - values.min())*255 / (values.max() - values.min())
        indices = np.abs(gray[np.newaxis, ...] - values.reshape(-1, 1, 1)).argmin(axis=0)
        return np.uint8(indices)

    def quantize2index_use_all(self, gray):
        hs, ws = self.small_images[0].shape
        # calc avg of tiles in big image (resizing)
        h, w = gray.shape
        # new height and width are the sizes if every small image is 1 pixel. i.e 
        # if the big image is 10 small images wide, we resize to 10*(small image width)
        nh, nw = round(h * self.upscale / hs), round(w * self.upscale/ ws) 
        gray = cv2.resize(gray, (nw * ws, nh * hs))
        gray = gray.reshape(nh, hs, nw, ws).swapaxes(1, 2) # now (nh, nw, hs, ws)
        
        # calc avg of small images
        values = np.array([np.average(img) for img in self.small_images])
        # scale original image colors for values
        gray = gray * (values.max() - values.min()) / 255 + values.min()

        # calculating difference from every small image for every tile
        diff = (gray[np.newaxis, ...] - np.array(self.small_images).reshape(-1, 1, 1, hs, ws)).sum(axis=(3,4))
        indices = np.abs(diff).argmin(axis=0)
        return np.uint8(indices)