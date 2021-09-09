import torch.nn as nn
import cv2

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

import sound_interface_gtts
import test

cap = cv2.VideoCapture(0, cv2.CAP_V4L)  # 노트북 웹캠을 카메라로 사용

ret, frame = cap.read()  # 사진 촬영
cv2.imwrite("../image_data/back.jpg", frame)  # 사진 저장

cap.release()

while(1): # signal == 1
    try:
        call_seeot = test.voice_recognition(2)

        if(call_seeot == "시옷"):
           sound_interface_gtts.main()

    except:
        continue