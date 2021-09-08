import cv2
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np

PATH = './'

class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32,
                      kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc1 = nn.Linear(in_features=64 * 6 * 6, out_features=600)
        self.dropout = nn.Dropout2d(0.25)
        self.fc2 = nn.Linear(in_features=600, out_features=19)
        self.fc3 = nn.Linear(in_features=120, out_features=19)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        # x = self.fc3(x)
        # x = self.softmax(x)

        return x


model = torch.load(PATH + 'model_chochanggi.pt')

labels = ['가방', '긴 겉옷', '긴팔 셔츠', '긴팔 원피스', '긴팔 티셔츠', '긴팔 후드티', '나시', '나시 원피스', '반바지',
          '반팔 셔츠', '반팔 원피스', '반팔 티셔츠', '샌들', '스니커즈', '슬림핏바지', '앵클부츠', '일자핏바지', '짧은 겉옷', '치마_test']


def preprocess(cl, ba):
    cloth = Image.open(cl).convert('RGB')
    cloth.save('cloth.png', 'png')

    back = Image.open(ba).convert('RGB')
    back.save('back.png', 'png')

    cloth = cv2.imread("cloth.png")
    back = cv2.imread("back.png")

    # 2. remove image backgrounds
    background_removed_img = cloth.copy()

    difference = cv2.subtract(back, cloth)
    background_removed_img[np.where((difference < [100, 100, 100]).all(axis=2))] = [0, 0, 0]
    #background_removed_img = cv2.imread('./testing_cloth/tshirt1.png')

    gray_img = cv2.cvtColor(background_removed_img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('gray_background_removed.png', gray_img)

    ret, thresh_gray = cv2.threshold(gray_img, thresh=5, maxval=255, type=cv2.THRESH_BINARY)
    points = np.argwhere(thresh_gray == 255)
    points = np.fliplr(points)
    x, y, w, h = cv2.boundingRect(points)
    x, y, w, h = x , y , w + 10, h + 10
    croped_img = gray_img[y:y+h, x:x+w]
    cv2.imwrite('croped.png', croped_img)
    print(croped_img.shape)

    gaussian_img = cv2.GaussianBlur(croped_img, (3, 3), 0)
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

    rotate_img = cv2.rotate(extended_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    cv2.imwrite('aaaaaa.png', rotate_img)

    return rotate_img


def test(img, labels):
    # load images
    #path = 'C:\\Users\\rosy0\\GITHUB\\Oss\\CNN\\testing_cloth\\'

    #img = cv2.imread(path + "skirt2.png")
    cv2.imshow('img',img)
    cv2.waitKey(0)
    transform = transforms.Compose([
        transforms.Grayscale(1),
        transforms.ToTensor()
    ])
    image = Image.fromarray(img)
    image = transform(image)
    image = image.unsqueeze(0)
    print(image.shape)
    output = model(image)
    output_list = output.tolist()[0]
    predicted_idx = output_list.index(max(output_list))
    predicted_label = labels[predicted_idx]
    print('predicted output:', predicted_label)

preprocessed_img = preprocess('./testing_cloth/cloth.jpg', './testing_cloth/back.jpg')
#img = cv2.imread('./testing_cloth/slimfitpants1.png')
test(preprocessed_img, labels)

# def test(test_list):
#     # load images
#     path = 'C:\\Users\\rosy0\\GITHUB\\Oss\\CNN\\testing_cloth\\'
#
#     img = cv2.imread(path + "straightfitpants2.png")
#     transfrom = transforms.Compose([
#         transforms.Grayscale(1),
#         transforms.ToTensor()
#     ])
#     image = Image.fromarray(img)
#     image = transfrom(image)
#     image = image.unsqueeze(0)
#     output = model(image)
#     output_list = output.tolist()[0]
#     predicted_idx = output_list.index(max(output_list))
#     predicted_label = test_list[predicted_idx]
#     print('predicted output:', predicted_label)
#
#
# test(labels)
