import numpy as np
import cv2

class ChooseTile:

    def __init__(self, big_image: np.array, small_images: list[np.array], tilingOption: str, gray, upscale):
        self.big_image = big_image
        self.small_images = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if gray else img for img in small_images]
        self.tilingOption = tilingOption
        self.gray = gray
        self.upscale = upscale
        assert(tilingOption == "average" or tilingOption == "use all")
    

    def tile(self):
        if self.gray:
            gray = cv2.cvtColor(self.big_image, cv2.COLOR_BGR2GRAY)
            # convert to numpy
            gray = np.float32(gray)
            if self.tilingOption == "average":
                indices = self.quantize2index_avg(gray)
            elif self.tilingOption == "use all":
                indices = self.quantize2index_use_all(gray)
        else: 
            indices = self.quantize2index_use_all(self.big_image)
        # converting image of indices to final result
        a, b = self.small_images[0].shape[:2]
        h, w = indices.shape[:2]
        return np.array(self.small_images)[indices.flatten()].reshape(h,w,a,b,-1).swapaxes(1,2).reshape(h*a, w*b, -1)


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

    def quantize2index_use_all(self, src):
        hs, ws, cs = self.small_images[0].shape
        # calc avg of tiles in big image (resizing)
        h, w, c = src.shape
        # new height and width are the sizes if every small image is 1 pixel. i.e 
        # if the big image is 10 small images wide, we resize to 10*(small image width)
        nh, nw = round(h * self.upscale / hs), round(w * self.upscale/ ws) 
        src = cv2.resize(src, (nw * ws, nh * hs))
        src = src.reshape(nh, hs, nw, ws, c).swapaxes(1, 2) # now (nh, nw, hs, ws, c)
        
        # calc avg of small images
        values = np.array([np.average(img, axis=(0,1)) for img in self.small_images])
        # scale original image colors for values
        # src = src * (values.max(axis=0) - values.min(axis=0)) / 255 + values.min(axis=0)

        # calculating difference from every small image for every tile
        diff = (src[np.newaxis, ...] - np.array(self.small_images).reshape(len(self.small_images), 1, 1, hs, ws, cs)).sum(axis=(3,4,5))
        indices = np.abs(diff).argmin(axis=0)
        return np.uint8(indices)