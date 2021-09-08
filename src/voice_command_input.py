from numpy.lib.function_base import insert
import pattern_recognition_v2
import extract_clothes
import cloth_recognition
import preprocess_cloth

import insert_clothes
import torch.nn as nn
import torchvision.transforms as transforms
import cv2
import playsound
import time
import gtts
import test
import weather
import insert_clothes
import cloth_recommendation
import socks
from PIL import Image


class Command:
    def __init__(self):
        self.color = ""
        self.pattern = ""
        self.kinds = ""
        self.fit = []
        self.picture = ""
        self.back = ""
        self.result_list = []
        self.yes = ["추천해 줘", "응"]
        self.no = ["아니", "괜찮아"]

    def pred_color(self, fw, bg):
        result_list = []
        color = extract_clothes.find_color_name()
        result_list.append(color)
        result_list.append(extract_clothes.find_matching_color_name(color))
        print(result_list)

        return result_list

    def pred_pattern(self, fw, bg):
        # pattern
        pattern = pattern_recognition_v2.pred_pattern(fw, bg)
        print("pattern = " + pattern)

        return pattern

    def pred_kinds(self, fw, bg):
        # kinds
        image = preprocess_cloth.preprocess_convnet(fw, bg)
        kinds = cloth_recognition.predict_cloth(image)
        print("kinds = " + kinds)

        return kinds

    def command_response(self, comm):

        if (comm == "무슨 옷이야"):
            cap = cv2.VideoCapture(0, cv2.CAP_V4L)  # 노트북 웹캠을 카메라로 사용

            ret, frame = cap.read()  # 사진 촬영
            cv2.imwrite("../image_data/cloth.jpg", frame)  # 사진 저장

            cap.release()

            self.picture = "../image_data/cloth.jpg"
            self.back = "../image_data/back.jpg"

            result_color = self.pred_color(self.picture, self.back)

            result_pattern = self.pred_pattern(self.picture, self.back)

            result_kinds = self.pred_kinds(self.picture, self.back)

            self.result_list.append(result_color)
            self.result_list.append(result_pattern)
            self.result_list.append(result_kinds)

            voice_command_color = self.result_list[0]
            voice_command_pattern = self.result_list[1]
            voice_command_kinds = self.result_list[2]

            if (voice_command_pattern == "줄 무늬"):
                result_string = "이 옷은" + voice_command_color[0][0] + " ." + voice_command_color[0][
                    1] + " ." + voice_command_pattern + " ." + voice_command_kinds + "입니다."

                tts = gtts.gTTS(text=result_string, lang="ko")
                tts.save("../sound_data/result.wav")

            else:
                result_string = "이 옷은" + voice_command_color[0][
                    0] + " ." + voice_command_pattern + " ." + voice_command_kinds + "입니다."

                tts = gtts.gTTS(text=result_string, lang="ko")
                tts.save("../sound_data/result.wav")

            playsound.playsound("../sound_data/result.wav")
            time.sleep(1)

            insert_clothes.insert_clothes(voice_command_kinds, voice_command_color[0][0])
            print("save!!")

            while (True):
                try:
                    playsound.playsound("../sound_data/recommandation.wav")
                    playsound.playsound("../sound_data/dingdong.wav")

                    user_command = test.voice_recognition(4)

                except:
                    playsound.playsound("../sound_data/idks.wav")
                    continue

                if (user_command in self.yes):
                    ###
                    wt = int(weather.weather())
                    c_list = cloth_recommendation.recommend_with_clothes(wt, voice_command_kinds)
                    
                    for i in voice_command_color[1]:
                        tts = gtts.gTTS(text=i, lang="ko")
                        tts.save("../sound_data/%s.wav" % i)

                        playsound.playsound("../sound_data/%s.wav" % i)
                        
                    for i in c_list:
                        tts = gtts.gTTS(text=i, lang="ko")
                        tts.save("../sound_data/%s.wav" % i)

                        playsound.playsound("../sound_data/%s.wav" % i)

                    playsound.playsound("../sound_data/result_r.wav")
                    time.sleep(1)
                    a = cloth_recommendation.DB(voice_command_color[1], c_list)
                    print(a)
                    if (a != 0):
                        for i in a:
                            tts = gtts.gTTS(text=i, lang="ko")
                            tts.save("../sound_data/%s.wav" % i)

                            playsound.playsound("../sound_data/%s.wav" % i)

                        playsound.playsound("../sound_data/result_s.wav")

                        break
                    else:
                        playsound.playsound("../sound_data/result_s.wav")


                elif (user_command in self.no):
                    break

        elif (comm == "날씨 알려 줘"):
            now_temperature = int(weather.weather())
            playsound.playsound("../sound_data/weather.wav")
            playsound.playsound("../sound_data/umbrella.wav")
            a, b, c = cloth_recommendation.recommend_without_clothes(now_temperature)

            d = a + b + c

            while (True):
                try:
                    playsound.playsound("../sound_data/recommandation_wt.wav")
                    playsound.playsound("../sound_data/dingdong.wav")

                    user_command = test.voice_recognition(4)

                except:
                    playsound.playsound("../sound_data/idks.wav")
                    continue

                if (user_command in self.yes):
                    for i in d:
                        tts = gtts.gTTS(text=i, lang="ko")
                        tts.save("../sound_data/%s.wav" % i)

                        playsound.playsound("../sound_data/%s.wav" % i)

                    playsound.playsound("../sound_data/result_r.wav")
                    break
                elif (user_command in self.no):
                    break

        elif (comm == "양말 구별해 줘"):
            cap = cv2.VideoCapture(0, cv2.CAP_V4L)  # 노트북 웹캠을 카메라로 사용

            ret, frame = cap.read()  # 사진 촬영
            cv2.imwrite("../image_data/socks.jpg", frame)  # 사진 저장

            cap.release()

            self.picture = cv2.imread("../image_data/socks.jpg")
            self.back = cv2.imread("../image_data/back.jpg")

            color1, color2 = socks.find_color_name(self.picture, self.back)

            if color1 != color2:
                tts = gtts.gTTS(text="두 양말의 색이 달라요. 왼쪽은" + color1 + ".오른쪽은" + color2 + "입니다.", lang="ko")
                tts.save("../sound_data/result_s.wav")
                playsound.playsound("../sound_data/result_s.wav")
            else:
                tts = gtts.gTTS(text="두 양말의 색이" + color1 +"으로 같아요.", lang="ko")
                tts.save("../sound_data/result_s.wav")
                playsound.playsound("../sound_data/result_s.wav")