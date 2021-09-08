#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
from PIL import Image
from PIL import ImageOps
import numpy as np

def preprocess_convnet(cloth, back):
    """
    :param cloth(string): 옷이 걸린 사진의 경로
    :param back(string):  옷이 걸리지 않은 사진(배경)의 경로
    :return rotate_im(PIL 이미지): ConvNet의 입력 형식에 맞게 전처리된 옷 이미지(28x28, grayscale)
    """

    # 카메라로부터 찍힌 옷 이미지의 확장자를 jpg에서 png로 바꾸어 저장한다.
    cloth = Image.open(cloth).convert('RGB')
    cloth.save('cloth.png', 'png')

    # 카메라로부터 찍힌 배경 이미지의 확장자를 jpg에서 png로 바꾸어 저장한다.
    back = Image.open(back).convert('RGB')
    back.save('back.png', 'png')

    # 앞서 저장한 두 png 이미지를 불러온다.
    cloth = cv2.imread("cloth.png")
    back = cv2.imread("back.png")

    # 1. remove image backgrounds
    # 앞서 찍은 두 이미지의 차이가 특정 임계값보다 작은 픽셀의 값을 [0, 0, 0]으로 바꾸어 배경제거를 해준다.
    background_removed_img = cloth.copy()

    difference = cv2.subtract(back, cloth)
    background_removed_img[np.where((difference < [100, 100, 100]).all(axis=2))] = [0, 0, 0]

    # 2. gray scale
    # BGR 세 채널을 가지던 이미지를 단일 채널을 가지는 흑백 이미지로 바꾸어준다.
    gray_img = cv2.cvtColor(background_removed_img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('gray_background_removed.png', gray_img)

    # 3. trimmimg(crop)
    # 이미지의 가장자리를 트리밍 처리한다.
    # PIL의 getbbox()를 이용하여 bounding box를 리턴받아 이를 기준으로 crop 하였다.
    gray_img = Image.fromarray(gray_img)
    bbox = gray_img.getbbox()
    croped_img = gray_img.crop(bbox)

    # 4. sharpening
    # 가우시안 연산자를 이용하여 이미지의 윤곽선을 선명하게 보이도록 한다.
    croped_img = np.array(croped_img)
    gaussian_img = cv2.GaussianBlur(croped_img, (3, 3), 0)
    mask = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])  # 마스크 배열의 항목들의 합이 1이 되도록
    sharpened_im = cv2.filter2D(gaussian_img, -1, mask)
    sharpened_im = Image.fromarray((sharpened_im))

    # 5. resizing
    # 이미지의 긴 변의 길이가 28이 되도록 resize 한다.
    if sharpened_im.size[1] > sharpened_im.size[0]:
        ratio = 28 / sharpened_im.size[1]
        dim = (int(sharpened_im.size[0] * ratio), 28)
    else:
        ratio = 28 / sharpened_im.size[0]
        dim = (28, int(sharpened_im.size[1] * ratio))
    resized_im = sharpened_im.resize(dim, Image.LANCZOS)

    # 6. extending
    # 이미지의 짧은 변의 길이가 28이 되도록 이미지를 extending 한다.
    # PIL의 ImageOps.expand() 함수를 이용하였다.
    new_size = 28
    delta_w = new_size - resized_im.size[0]
    delta_h = new_size - resized_im.size[1]
    padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
    extend_im = ImageOps.expand(resized_im, padding)

    # 7. rotate
    # 옷 전체를 잘리지 않게 찍기 위해 웹캠을 90도 회전시켰다. 따라서 다시 반대로 90도 회전시켜 줘야 한다.
    rotate_im = extend_im.rotate(90)

    # 전처리가 잘 되었는지 확인하기 위해 저장한다.
    rotate_im.save('preprocessed.png', 'png')

    return rotate_im