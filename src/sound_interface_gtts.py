# -*- coding: utf-8 -*-

import extract_clothes
import torch.nn as nn
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
        self.fc2 = nn.Linear(in_features=600, out_features=10)


    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x

import voice_command_input
import test
import playsound

command_list = ["무슨 옷이야", "날씨 알려 줘", "저장해 줘", "양말 구별해 줘"]

def main():
    while(True):
        try:
            playsound.playsound("../sound_data/intro.wav")
            playsound.playsound("../sound_data/dingdong.wav")

            user_command = test.voice_recognition(4)

        except:
            playsound.playsound("../sound_data/idks.wav")
            continue

        if(user_command in command_list):
            playsound.playsound("../sound_data/dingdong.wav")
            playsound.playsound("../sound_data/wait.wav")
            break

        else:
            playsound.playsound("../sound_data/idks.wav")
            continue

    voice_command = voice_command_input.Command()
    voice_command.command_response(user_command)