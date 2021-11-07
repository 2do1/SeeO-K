Apache License
Version 2.0, January 2004
http://www.apache.org/licenses/
Copyright (c) 2021 KOBOTEN kobot1010@gmail.com.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

import torch.nn as nn
import cv2
# convolution neural network
class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16,
                      kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc1 = nn.Linear(in_features=122112, out_features=600)
        self.dropout = nn.Dropout2d(0.2)
        self.fc2 = nn.Linear(in_features=600, out_features=16)
        self.fc3 = nn.Linear(in_features=120, out_features=16)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.reshape(x.size(0), -1)
        print(x.shape)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        # x = self.fc3(x)
        # x = self.softmax(x)

        return x


import sound_interface_gtts
import test
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db

cred = credentials.Certificate("see-ot-a14fe-firebase-adminsdk-wraln-1e78f4e053.json")
firebase_admin.initialize_app(cred, {'databaseURL': 'https://see-ot-a14fe-default-rtdb.firebaseio.com/'})
ref = db.reference()

cap = cv2.VideoCapture(0, cv2.CAP_V4L)  # 노트북 웹캠을 카메라로 사용

ret, frame = cap.read()  # 사진 촬영
cv2.imwrite("../image_data/back.jpg", frame)  # 사진 저장

cap.release()

while(1): # signal == 1
    try:
        call_seeot = test.voice_recognition(2)

        if(call_seeot == "시옷"):
           sound_interface_gtts.main(ref)

    except Exception as e:
        print(str(e))
        print(type(e))
        continue

call_seeot = test.voice_recognition(2)

# if(call_seeot == "시옷"):
#     sound_interface_gtts.main(ref)
