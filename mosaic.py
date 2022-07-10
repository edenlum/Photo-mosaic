import cv2
import numpy as np
from choose_tile import *

def quantize(src, color_res, values=None):
  gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
  # convert to numpy
  gray = np.float32(gray)
  if values is None:
    delta = 0.5 + 1/256
    # make the image have 6 levels of gray
    quantized = (np.round(gray*color_res/256+delta) - delta)*256/color_res
  else:
    values = np.array(values)
    # scale values to between 0 and 255
    values = (values - values.min())*255 / (values.max() - values.min())
    quantized = values[np.abs(gray[np.newaxis, ...] - values.reshape(-1, 1, 1)).argmin(axis=0)]
  return np.uint8(quantized)

def quantize2keys(big_image, values):
  gray = cv2.cvtColor(big_image, cv2.COLOR_BGR2GRAY)
  # convert to numpy
  gray = np.float32(gray)
  values = np.array(values)
  # scale values to between 0 and 255
  values = (values - values.min())*255 / (values.max() - values.min())
  quantized = np.abs(gray[np.newaxis, ...] - values.reshape(-1, 1, 1)).argmin(axis=0)
  return np.uint8(quantized)

def load_images(images_names):
  images = []
  for name in images_names:
    images.append(cv2.imread(f"{name}"))
  return images

def change_images_res(images, new_res):
  new_images = []
  for img in images:
    new_images.append(cv2.resize(img, new_res))
  return new_images

def write_images(names, images):
  for name, img in zip(names, images):
    cv2.imwrite(f"{name}.png", img)

def to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def create_image_from_small_images(image_name, images_to_tile_names, res_name, small_res = (20, 20), factor = 4, gray = True):
  to_gray = lambda x: [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in x]
  new_names = [f"{i}_low_res" for i in images_to_tile_names]
  a,b = small_res
  
  # load, change res, and convert to grayscale
  dice_images = load_images(images_to_tile_names)
  low_res_gray = to_gray(change_images_res(dice_images, small_res))
  
  # load main image and resize + quantize color (gray scale)
  image = cv2.imread(image_name)
  h,w,c = image.shape
  resized_image = cv2.resize(image, (w//factor, h//factor))

  # creating object that will choose how to tile the image
  tiler = ChooseTile(resized_image, low_res_gray, "average", gray=True)
  # returning an "image" of indices 
  quantized_indices = tiler.quantize2index()
  h,w = quantized_indices.shape
  # converting image of indices to final result
  res = np.array(low_res_gray)[quantized_indices.flatten()].reshape(h,w,a,b).swapaxes(1,2).reshape(h*a, w*b)

  if not cv2.imwrite(res_name, res):
    print("Writing result failed. Does the path exist?")