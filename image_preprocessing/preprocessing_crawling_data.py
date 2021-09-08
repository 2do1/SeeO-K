from PIL import Image
import cv2
import numpy as np
from os import listdir
from os.path import splitext

#directory = 'C:\\Users\\rosy0\\OneDrive\\바탕 화면\\oss_data\\train\\10'
directory = 'C:\\Users\\rosy0\\GITHUB\\Oss\\image'

'''
from PIL import Image
from os import listdir
from os.path import splitext

target_directory = '.'
target = '.png'

for file in listdir(target_directory):
    filename, extension = splitext(file)
    try:
        if extension not in ['.py', target]:
            im = Image.open(filename + extension)
            im.save(filename + target)
    except OSError:
        print('Cannot convert %s' % file)
'''

COUNT = 67112

for file in listdir(directory):
    img = directory + '\\' + file
    cloth = cv2.imread(img)
    #print(img)
    back = cv2.imread('gray_background.png')
    #print(back)
    print(COUNT, cloth.shape)
    print(COUNT, back.shape)

    if cloth.shape != (433, 300, 3):
        continue

    # 2. remove image backgrounds
    background_removed_img = cloth.copy()

    difference = cv2.subtract(back, cloth)
    background_removed_img[np.where((difference < [10, 10, 10]).all(axis=2))] = [0,0,0]

    gray_img = cv2.cvtColor(background_removed_img, cv2.COLOR_BGR2GRAY)

    gaussian_img = cv2.GaussianBlur(gray_img, (3, 3), 0)
    mask = np.array([[-1, -1, -1], [-1, 10, -1], [-1, -1, -1]])  # 마스크 배열의 항목들의 합이 1이 되도록
    LoG_img = cv2.filter2D(gaussian_img, -1, mask)


    # 4. pixel subsampling and resizing
    if LoG_img.shape[1] > LoG_img.shape[0]:
        ratio = 224 / LoG_img.shape[1]
        dim = (224, int(LoG_img.shape[0] * ratio))
    else:
        ratio = 224 / LoG_img.shape[0]
        dim = (int(LoG_img.shape[1] * ratio), 224)

    LoG_img = cv2.resize(LoG_img, dim, interpolation=cv2.INTER_AREA)

    bgr = cv2.cvtColor(LoG_img, cv2.COLOR_GRAY2BGR)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))

    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    if final.shape[1] > final.shape[0]:
        ratio = 112 / final.shape[1]
        dim = (112, int(final.shape[0] * ratio))
    else:
        ratio = 112 / final.shape[0]
        dim = (int(final.shape[1] * ratio), 112)

    LoG_img = cv2.resize(final, dim, interpolation=cv2.INTER_AREA)

    lab = cv2.cvtColor(LoG_img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))

    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    if final.shape[1] > final.shape[0]:
        ratio = 28 / final.shape[1]
        dim = (28, int(final.shape[0] * ratio))
    else:
        ratio = 28 / final.shape[0]
        dim = (int(final.shape[1] * ratio), 28)

    LoG_img = cv2.resize(final, dim, interpolation=cv2.INTER_AREA)

    # 6. image extending(28x28)
    y, x, h, w = (0, 0, LoG_img.shape[0], LoG_img.shape[1])
    w_x = (28 - (w - x)) / 2
    h_y = (28 - (h - y)) / 2

    if (w_x < 0):
        w_x = 0
    elif (h_y < 0):
        h_y = 0

    M = np.float32([[1, 0, w_x], [0, 1, h_y]])
    extended_img = cv2.warpAffine(LoG_img, M, (28, 28))

    cv2.imwrite('C:\\Users\\rosy0\\GITHUB\\Oss\\after_processing_image\\'+ str(COUNT) + ".png", extended_img)
    COUNT += 1
    # 7. image negating (complementing color)
    # negated_img = extended_img.copy()
    # height, width = (negated_img.shape[0], negated_img.shape[1])
    #
    # for i in range(0, width):
    #     for j in range(0, height):
    #         if ((negated_img[i, j] != [0, 0, 0]).all()):
    #             color = negated_img[i, j]
    #             negated_img[i, j] = (255 - color[0], 255 - color[1], 255 - color[2])

    # 8. grayscale
    # gray_img = cv2.cvtColor(negated_img, cv2.COLOR_BGR2GRAY)

    #  9. rotation - need only for real cloth data
    # rotate_img = cv2.rotate(extended_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    # cv2.imwrite('rotate.png', rotate_img)



