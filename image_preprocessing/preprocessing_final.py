#-*-coding:utf-8-*-
import cv2
import numpy as np
from os import listdir
from PIL import ImageOps
from PIL import Image

#directory = 'C:\\Users\\rosy0\\OneDrive\\바탕 화면\\oss_data\\train\\10'
directory = 'C:\\Users\\rosy0\\GITHUB\\Oss\\image'

COUNT = 67308

for file in listdir(directory):
    img = directory + '\\' + file
    cloth = cv2.imread(img)
    print(file)
    back = cv2.imread('gray_background.png')

    if cloth.shape != (433, 300, 3):
        continue

    # remove image backgrounds
    background_removed_img = cloth.copy()

    # background_removed_img[np.where((cloth == back).all(axis=2))] = [0,0,0]

    difference = cv2.subtract(back, cloth)
    background_removed_img[np.where(([0,0,0] <= difference).all(axis=2) & (difference < [3,3,3]).all(axis=2))] = [0,0,0]

    # grayscale
    gray_img = cv2.cvtColor(background_removed_img, cv2.COLOR_BGR2GRAY)

    gray_img = Image.fromarray(gray_img)
    bbox = gray_img.getbbox()
    crop = gray_img.crop(bbox)
    #crop.show()
    print(crop.size)

    crop = np.array(crop)
    gaussian_img = cv2.GaussianBlur(crop, (3, 3), 0)
    mask = np.array([[-1, -1, -1], [-1, 10, -1], [-1, -1, -1]])  # 마스크 배열의 항목들의 합이 1이 되도록
    crop = cv2.filter2D(gaussian_img, -1, mask)
    crop = Image.fromarray((crop))

    if crop.size[1] > crop.size[0]:
        # print('aa')
        ratio = 28 / crop.size[1]
        dim = (int(crop.size[0] * ratio), 28)
    else:
        # print('bb')
        ratio = 28 / crop.size[0]
        dim = (28, int(crop.size[1] * ratio))

    crop = crop.resize(dim, Image.LANCZOS)
    # print(crop.size)

    new_size = 28
    delta_w = new_size - crop.size[0]
    delta_h = new_size - crop.size[1]
    padding = (delta_w//2, delta_h//2, delta_w-(delta_w//2), delta_h-(delta_h//2))
    new_im = ImageOps.expand(crop, padding)
    #new_im.show()
    # print(new_im.size)

    #new_im.save('gaus.png', 'png')
    # new_im.save('gaus_no.png', 'png')


    # crop
    # ret, thresh_gray = cv2.threshold(gray_img, thresh=1, maxval=255, type=cv2.THRESH_BINARY)
    # #thresh_gray = cv2.cvtColor(thresh_gray, cv2.COLOR_BGR2GRAY)
    # points = np.argwhere(thresh_gray == 255)
    # points = np.fliplr(points)
    # x, y, w, h = cv2.boundingRect(points)
    # x, y, w, h = x - 10, y - 10, w + 20, h + 10
    # croped_img = gray_img[y:y+h, x:x+w]
    # print(croped_img.shape)
    #
    # if(croped_img.shape[0] < 10 or croped_img.shape[1] < 10):
    #     continue
    #
    #
    # # sharpening
    # gaussian_img = cv2.GaussianBlur(croped_img, (3, 3), 0)
    # mask = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])  # 마스크 배열의 항목들의 합이 1이 되도록
    # LoG_img = cv2.filter2D(gaussian_img, -1, mask)
    #
    #
    # # resizing -> 224
    # if LoG_img.shape[1] > LoG_img.shape[0]:
    #     ratio = 224 / LoG_img.shape[1]
    #     dim = (224, int(LoG_img.shape[0] * ratio))
    # else:
    #     ratio = 224 / LoG_img.shape[0]
    #     dim = (int(LoG_img.shape[1] * ratio), 224)
    #
    # LoG_img = cv2.resize(LoG_img, dim, interpolation=cv2.INTER_AREA)
    #
    # # sharpening
    # bgr = cv2.cvtColor(LoG_img, cv2.COLOR_GRAY2BGR)
    # lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    # l, a, b = cv2.split(lab)
    # clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    #
    # cl = clahe.apply(l)
    # limg = cv2.merge((cl, a, b))
    #
    # final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    #
    # # resizing -> 112
    # if final.shape[1] > final.shape[0]:
    #     ratio = 112 / final.shape[1]
    #     dim = (112, int(final.shape[0] * ratio))
    # else:
    #     ratio = 112 / final.shape[0]
    #     dim = (int(final.shape[1] * ratio), 112)
    #
    # LoG_img = cv2.resize(final, dim, interpolation=cv2.INTER_AREA)
    #
    # # sharpening
    # lab = cv2.cvtColor(LoG_img, cv2.COLOR_BGR2LAB)
    # l, a, b = cv2.split(lab)
    # clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    #
    # cl = clahe.apply(l)
    # limg = cv2.merge((cl, a, b))
    #
    # final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    #
    # # resizing -> 28
    # if final.shape[1] > final.shape[0]:
    #     ratio = 28 / final.shape[1]
    #     dim = (28, int(final.shape[0] * ratio))
    # else:
    #     ratio = 28 / final.shape[0]
    #     dim = (int(final.shape[1] * ratio), 28)
    #
    # LoG_img = cv2.resize(final, dim, interpolation=cv2.INTER_AREA)
    #
    # # image extending(28x28)
    # y, x, h, w = (0, 0, LoG_img.shape[0], LoG_img.shape[1])
    # w_x = (28 - (w - x)) / 2
    # h_y = (28 - (h - y)) / 2
    #
    # if (w_x < 0):
    #     w_x = 0
    # elif (h_y < 0):
    #     h_y = 0
    #
    # M = np.float32([[1, 0, w_x], [0, 1, h_y]])
    # extended_img = cv2.warpAffine(LoG_img, M, (28, 28))

    # storing
    new_im = np.array(new_im)
    #new_im.save('C:\\Users\\rosy0\\GITHUB\\Oss\\after_processing_image\\'+ str(COUNT) + ".png", 'png')
    cv2.imwrite('C:\\Users\\rosy0\\GITHUB\\Oss\\after_processing_image\\'+ str(COUNT) + ".png", new_im)
    COUNT += 1
