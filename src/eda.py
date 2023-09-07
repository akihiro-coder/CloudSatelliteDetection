import cv2
import numpy as np


# for num in range(1000):
#     file_path = f'../data/train_true_color/train_true_color_{num}.tif'
#     img = cv2.imread(file_path) # BGR, shape = (h, w)
#     if img.shape != (1000, 1000, 3):
#         print('no')


file_path = '../data/train_true_color/train_true_color_0.tif'
# file_path = '../data/train_mask/train_mask_0.tif'
img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)  # BGR, shape = (h, w)
img = img[:, :, 1]
print(img.dtype)
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
