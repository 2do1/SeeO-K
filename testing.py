import cv2
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms

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


model = torch.load(PATH + 'model.pt')
labels = ['가방', '긴 겉옷', '긴팔 셔츠', '긴팔 원피스', '긴팔 티셔츠', '긴팔 후드티', '나시', '나시 원피스', '반바지',
          '반팔 셔츠', '반팔 원피스', '반팔 티셔츠', '샌들', '스니커즈', '슬림핏바지', '앵클부츠', '일자핏바지', '짧은 겉옷', '치마']


def test(test_list):
    # load images
    print("wow")
    img = cv2.imread("60013.png")
    transfrom = transforms.Compose([
        transforms.Grayscale(1),
        transforms.ToTensor()
    ])
    # print(image.size)
    image = Image.fromarray(img)
    image = transfrom(image)
    # print(image.size())
    # plt.title("input image")
    # plt.imshow(image.squeeze(0), cmap='gray')
    # plt.show()
    image = image.unsqueeze(0)
    # print(image.size())
    output = model(image)
    output_list = output.tolist()[0]
    # print(output)
    # print(output_list[0])
    predicted_idx = output_list.index(max(output_list))
    # print(predicted_idx)
    predicted_label = test_list[predicted_idx]
    print('predicted output:', predicted_label)


test(labels)
