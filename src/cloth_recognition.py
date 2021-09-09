#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torchvision.transforms as transforms

""" 
* 옷의 종류를 판별하는 모듈이다.
* @author 이한정
* @version 1.0.0
"""

# training에 사용한 Convolutional Neural Network
class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
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
        self.fc2 = nn.Linear(in_features=600, out_features=10)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x

# 학습된 모델 불러오기
model = torch.load('../model/model_chochanggi_data4_10.pt')

# 분류 가능한 클래스들의 라벨 리스트
labels = ['가방', '긴 겉옷', '긴팔 티셔츠', '나시 원피스', '반바지',
          '반팔 원피스', '반팔 티셔츠', '일자핏바지', '짧은 겉옷', '치마']

def predict_cloth(img):
    """
    :param img(PIL 이미지): 알맞게 전처리 된(28x28, grayscale) 옷 이미지
    :return predicted_label(string): 문자열로 나타낸 옷의 종류
    """

    # training 시킬 때와 같이 전처리를 한다.
    transform = transforms.Compose([
        transforms.Grayscale(1),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081))
    ])

    img.save('../image_data/preprocessed_img.png','png')

    image = transform(img)
    image = image.unsqueeze(0) # 차원 추가

    output = model(image)

    output_list = output.tolist()[0]
    print('predicted kinds list:', output_list)
    predicted_idx = output_list.index(max(output_list)) # 예측값의 인덱스
    predicted_label = labels[predicted_idx] # 예측값

    print('predicted output:', predicted_label)

    return predicted_label
