import cv2
import numpy as np

def quantize(src, color_res, values=None):
  gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
  # convert to numpy
  gray = np.float32(gray)
  if values is None:
    delta = 0.5 + 1/256
    # make the image have 6 levels of gray
    quantized = (np.round(gray*color_res/256+delta) - delta)*256/color_res
    # convert back to image
  else:
    values = np.array(values)
    # scale values to between 0 and 255
    values = (values - values.min())*255 / (values.max() - values.min())
    quantized = values[np.abs(gray[np.newaxis, ...] - values.reshape(-1, 1, 1)).argmin(axis=0)]
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

def create_image_from_small_images(image_name, images_to_tile_names, res_name, small_res = (20, 20), factor = 4):
  to_gray = lambda x: [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in x]
  new_names = [f"{i}_low_res" for i in images_to_tile_names]
  # load, change res, and convert to grayscale
  dice_images = load_images(images_to_tile_names)
  low_res_gray = to_gray(change_images_res(dice_images, small_res))
  # sort images by average color
  low_res_gray_sorted = sorted(low_res_gray, key=lambda x: np.average(x))
  values = [np.average(img) for img in low_res_gray_sorted]
  # write back results
  write_images(new_names, low_res_gray)
  # load main image and resize + quantize color (gray scale)
  image = cv2.imread(image_name)
  image = cv2.resize(image, (image.shape[1]*small_res[0]//factor, image.shape[0]*small_res[1]//factor))
  h,w,c = image.shape
  resized_image = cv2.resize(image, (w//small_res[1], h//small_res[0]))
  quantized = quantize(resized_image, len(low_res_gray), values)
  # write to file
  cv2.imwrite('quantized.png', quantized)
  map = {k:low_res_gray_sorted[i] for i,k in enumerate(np.sort(np.unique(quantized)))}
  h,w = quantized.shape
  res = np.array([map[k] for k in quantized.flatten()]).reshape(h,w,small_res[0],small_res[1]).swapaxes(1,2)
  res = res.reshape((h*small_res[0], w*small_res[1]))
  cv2.imwrite(res_name, res)