import torch
from torchvision import models
import cv2
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
import numpy as np


PATH = './'
num_classes = 15

labels = ['긴 겉옷', '긴팔 셔츠', '긴팔 원피스', '긴팔 티셔츠', '긴팔 후드티', '나시', '나시 원피스', '반바지',
          '반팔 셔츠', '반팔 원피스', '반팔 티셔츠', '슬림핏바지', '일자핏바지', '짧은 겉옷', '치마']

model = models.resnet18(pretrained=True)

num_ftrs = model.fc.in_features
model.fx = nn.Linear(num_ftrs, num_classes)

model.load_state_dict(torch.load(PATH + 'ResNet18_class_modified.pt'))

model.eval()


def preprocess(cl, ba):
    # JPG to PNG
    cloth = Image.open(cl).convert('RGB')
    cloth.save('cloth.png', 'png')

    back = Image.open(ba).convert('RGB')
    back.save('back.png', 'png')

    cloth = cv2.imread("cloth.png")
    back = cv2.imread("back.png")

    # remove image backgrounds
    background_removed_img = cloth.copy()

    difference = cv2.subtract(back, cloth)
    background_removed_img[np.where((difference < [10, 10, 10]).all(axis=2))] = [0, 0, 0]

    #### for testing with original png image
    ## background_removed_img = cv2.imread('./testing_cloth/sh1.png')

    # grayscale
    gray_img = cv2.cvtColor(background_removed_img, cv2.COLOR_BGR2GRAY)

    # crop image
    ret, thresh_gray = cv2.threshold(gray_img, thresh=5, maxval=255, type=cv2.THRESH_BINARY)
    points = np.argwhere(thresh_gray == 255)
    points = np.fliplr(points)
    x, y, w, h = cv2.boundingRect(points)
    x, y, w, h = x , y , w + 10, h + 10

    croped_img = gray_img[y:y+h, x:x+w]

    print(croped_img.shape)
    cv2.imwrite('croped.png', croped_img)

    # image sharpening
    gaussian_img = cv2.GaussianBlur(croped_img, (3, 3), 0)
    mask = np.array([[-1, -1, -1], [-1, 10, -1], [-1, -1, -1]])  # 마스크 배열의 항목들의 합이 1이 되도록
    LoG_img = cv2.filter2D(gaussian_img, -1, mask)

    # resize -> 224
    if LoG_img.shape[1] > LoG_img.shape[0]:
        ratio = 224 / LoG_img.shape[1]
        dim = (224, int(LoG_img.shape[0] * ratio))
    else:
        ratio = 224 / LoG_img.shape[0]
        dim = (int(LoG_img.shape[1] * ratio), 224)

    resized_img = cv2.resize(LoG_img, dim, interpolation=cv2.INTER_AREA)

    # image sharpening
    bgr = cv2.cvtColor(resized_img, cv2.COLOR_GRAY2BGR)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))

    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    # resize -> 112
    if final.shape[1] > final.shape[0]:
        ratio = 112 / final.shape[1]
        dim = (112, int(final.shape[0] * ratio))
    else:
        ratio = 112 / final.shape[0]
        dim = (int(final.shape[1] * ratio), 112)

    resized_img = cv2.resize(final, dim, interpolation=cv2.INTER_AREA)

    # image sharpeing
    lab = cv2.cvtColor(resized_img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))

    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    # resize -> 28
    if final.shape[1] > final.shape[0]:
        ratio = 28 / final.shape[1]
        dim = (28, int(final.shape[0] * ratio))
    else:
        ratio = 28 / final.shape[0]
        dim = (int(final.shape[1] * ratio), 28)

    resized_img = cv2.resize(final, dim, interpolation=cv2.INTER_AREA)

    # image extending(28x28)
    y, x, h, w = (0, 0, resized_img.shape[0], resized_img.shape[1])
    w_x = (28 - (w - x)) / 2
    h_y = (28 - (h - y)) / 2

    if (w_x < 0):
        w_x = 0
    elif (h_y < 0):
        h_y = 0

    M = np.float32([[1, 0, w_x], [0, 1, h_y]])
    extended_img = cv2.warpAffine(resized_img, M, (28, 28))

    # rotate(only for image from logitech)
    rotate_img = cv2.rotate(extended_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    cv2.imwrite('7.rotate_LoG.png', rotate_img)

    return rotate_img


def test(img, labels):
    # load images
    #path = 'C:\\Users\\rosy0\\GITHUB\\Oss\\CNN\\testing_cloth\\'
    #img = cv2.imread(path + "skirt2.png")

    transform = transforms.Compose([
        transforms.Grayscale(1),
        transforms.ToTensor()
    ])
    image = Image.fromarray(img)
    image = transform(image)

    image = image.unsqueeze(0)
    image2 = torch.cat([image, image], dim=1)
    image3 = torch.cat([image2, image], dim=1)

    output = model(image3)
    output_list = output.tolist()[0]

    predicted_idx = output_list.index(max(output_list))
    predicted_label = labels[predicted_idx]

    print('predicted output:', predicted_label)

preprocessed_img = preprocess('./testing_cloth/33.jpg', './testing_cloth/44.jpg')
#img = cv2.imread('./testing_cloth/longcoat2.png')
test(preprocessed_img, labels)
