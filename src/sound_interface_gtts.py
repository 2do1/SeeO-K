# -*- coding: utf-8 -*-

import extract_clothes
import torch.nn as nn

'''
사용자의 음성 입력을 대기하고 적절한 명령어가 입력되면 해당하는 기능을 실행하는 기능을 가진 파일

@__author__ = 송수인
@version 1.0
'''

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



import voice_command_input
import test
import playsound

# 사용자에게 알맞은 명령어가 입력되었는지를 판단하기 위한 명령어 리스트
command_list = ["무슨 옷이야", "날씨 알려 줘", "저장해 줘", "양말 구별해 줘"]

# 사용자에게 음성 안내를 제공하고 음성 입력을 대기
def main(ref):
    while(True):
        try:
            playsound.playsound("../sound_data/intro.wav")
            playsound.playsound("../sound_data/dingdong.mp3")

            user_command = test.voice_recognition(4)

        except:
            playsound.playsound("../sound_data/idks.wav")
            continue

        if(user_command in command_list):
            playsound.playsound("../sound_data/dingdong.mp3")
            playsound.playsound("../sound_data/wait.wav")
            break

        else:
            playsound.playsound("../sound_data/idks.wav")
            continue

    voice_command = voice_command_input.Command()
    voice_command.command_response(user_command, ref)
