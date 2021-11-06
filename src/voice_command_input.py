'''
* 사용자로부터 음성을 입력받아 해당하는 기능을 실행하는 코드
* 
* @author 송수인
* @version 1.0
'''

from numpy.lib.function_base import insert
import pattern_recognition_v2
import extract_clothes
import cloth_recognition
import preprocess_cloth

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
# import personal_color_starbucks.src.main as pcs

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

    # 입력된 옷의 색을 예측
    def pred_color(self, fw, bg, ref):
        result_list = []
        color = extract_clothes.find_color_name()
        result_list.append(color)
        result_list.append(extract_clothes.find_matching_color_name(color, ref))
        print(result_list)

        # 추정되는 색과 어울리는 색 리스트가 리턴
        return result_list

    # 입력된 옷의 패턴을 예측
    def pred_pattern(self, fw, bg):
        # pattern
        pattern = pattern_recognition_v2.pred_pattern(fw, bg)
        print("pattern = " + pattern)

        # 추정되는 패턴이 리턴
        return pattern

    # 입력된 옷의 종류를 리턴
    def pred_kinds(self, fw, bg):
        # kinds
        image = preprocess_cloth.preprocess_convnet(fw, bg)
        kinds = cloth_recognition.predict_cloth(image)
        print("kinds = " + kinds)

        # 추정되는 종류가 리턴
        return kinds

    # 입력되는 사용자의 명령어에 따라 해당하는 기능이 실행
    def command_response(self, comm, ref):

        # 옷 인식 요청 명령어
        if (comm == "무슨 옷이야"):
            cap = cv2.VideoCapture(0, cv2.CAP_V4L)  # 노트북 웹캠을 카메라로 사용

            ret, frame = cap.read()  # 사진 촬영
            cv2.imwrite("../image_data/cloth.jpg", frame)  # 사진 저장

            cap.release()

            self.picture = "../image_data/cloth.jpg"
            self.back = "../image_data/back.jpg"

            # 색 인식 기능 실행
            result_color = self.pred_color(self.picture, self.back, ref)

            # 패턴 인식 기능 실행
            result_pattern = self.pred_pattern(self.picture, self.back)

            # 종류 인식 기능 실행
            result_kinds = self.pred_kinds(self.picture, self.back)

            self.result_list.append(result_color)
            self.result_list.append(result_pattern)
            self.result_list.append(result_kinds)

            voice_command_color = self.result_list[0]
            voice_command_pattern = self.result_list[1]
            voice_command_kinds = self.result_list[2]

            # 인식 결과를 음성으로 변환하여 사용자에게 출력
            if (voice_command_pattern == "줄 무늬" or voice_command_pattern == "체크 무늬"):
                if (voice_command_color[0][0] != voice_command_color[0][1]):
                    result_string = "이 옷은" + voice_command_color[0][0] + " ." + voice_command_color[0][1] + " ." + voice_command_pattern + " ." + voice_command_kinds + "입니다."
                else:
                    result_string = "이 옷은" + voice_command_color[0][0] + " ." + voice_command_pattern + " ." + voice_command_kinds + "입니다."
                
                tts = gtts.gTTS(text=result_string, lang="ko")
                tts.save("../sound_data/result.wav")
  
            else:
                result_string = "이 옷은" + voice_command_color[0][0] + " ." + voice_command_pattern + " ." + voice_command_kinds + "입니다."

                tts = gtts.gTTS(text=result_string, lang="ko")
                tts.save("../sound_data/result.wav")

            playsound.playsound("../sound_data/result.wav")
            time.sleep(1)

            # 옷장 데이터베이스에 해당 옷 저장
            print('color',voice_command_color)
            print('kinds',voice_command_kinds)
            insert_clothes.insert_clothes(voice_command_kinds, voice_command_color[0][0], ref)
            print("save!!")

            # 추천 여부 사용자에게 질의 후, 입력 대기
            while (True):
                try:
                    playsound.playsound("../sound_data/recommandation.wav")
                    playsound.playsound("../sound_data/dingdong.mp3")

                    user_command = test.voice_recognition(4)

                except Exception as e:
                    playsound.playsound("../sound_data/idks.wav")
                    print(str(e))
                    print(type(e))
                    continue

                # 사용자의 대답이 긍정일 경우,
                if (user_command in self.yes):
                    ###
                    wt = int(weather.weather())
                    c_list = cloth_recommendation.recommend_with_clothes(wt, voice_command_kinds)
                    print("voice_command_color : ", voice_command_color)
                    print("voice_command_kinds : ", voice_command_kinds)
                    print("c_list : ", c_list, len(c_list))
                    if(len(c_list) != 0):
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
                        a = cloth_recommendation.DB(voice_command_color[1], c_list, ref)
                        print(a)
                        result_list = []
                        print("a: ", a)
                        if a !=0 :
                            for j in range(len(a)): 
                                if(j%2==1 and a[j] in c_list): 
                                    if(type(a[j-1]) is str):
                                        result_list.append(a[j-1])
                                        result_list.append(a[j])
                                    else:
                                        result_list.append(a[j-1][0])
                                        result_list.append(a[j])
                            print("result_list : ", result_list) 
                        if (len(result_list) != 0):
                            for i in result_list:
                                tts = gtts.gTTS(text=i, lang="ko")
                                tts.save("../sound_data/%s.wav" % i)

                                playsound.playsound("../sound_data/%s.wav" % i)

                            tts = gtts.gTTS(text="가 옷장에 있습니다.", lang="ko")
                            tts.save("../sound_data/result_exist.wav")
                            playsound.playsound("../sound_data/result_exist.wav")

                            break
                        else:
                            tts = gtts.gTTS(text="어울리는 옷이 옷장에 없습니다.", lang="ko")
                            tts.save("../sound_data/result_notexist.wav")
                            playsound.playsound("../sound_data/result_notexist.wav")
                            break
                    else:
                        tts = gtts.gTTS(text = "이 옷은 지금 날씨에 맞지 않아요.", lang="ko")
                        tts.save("../sound_data/result_nope.wav")
                        playsound.playsound("../sound_data/result_nope.wav")
                        break

                # 입력된 사용자의 대답이 부정일 경우,
                elif (user_command in self.no):
                    break

        # 날씨 정보 요청 명령어
        elif (comm == "날씨 알려 줘"):
            print(1)
            now_temperature = int(weather.weather())
            playsound.playsound("../sound_data/weather.wav")
            playsound.playsound("../sound_data/umbrella.wav")
            playsound.playsound("../sound_data/temperature_gap.wav")
            a, b, c = cloth_recommendation.recommend_without_clothes(now_temperature)

            d = a + b + c
            print(15)
            # 날씨에 따른 옷 추천 여부 질의 후, 사용자의 입력 대기
            while (True):
                try:
                    playsound.playsound("../sound_data/recommandation_wt.wav")
                    playsound.playsound("../sound_data/dingdong.mp3")

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

        # 두 양말의 색을 구별하는 기능 실행
        elif (comm == "양말 구별해 줘"):
            cap = cv2.VideoCapture(0, cv2.CAP_V4L)  # 노트북 웹캠을 카메라로 사용

            ret, frame = cap.read()  # 사진 촬영
            cv2.imwrite("../image_data/socks.jpg", frame)  # 사진 저장

            cap.release()

            self.picture = cv2.imread("../image_data/socks.jpg")
            self.back = cv2.imread("../image_data/back.jpg")

            color1, color2 = socks.find_color_name(self.picture, self.back)

            # 결과 값에 따른 결과를 음성으로 출력
            if color1 != color2:
                tts = gtts.gTTS(text="두 양말의 색이 달라요. 왼쪽은" + color1 + ".오른쪽은" + color2 + "입니다.", lang="ko")
                tts.save("../sound_data/result_socks.wav")
                playsound.playsound("../sound_data/result_socks.wav")
            else:
                tts = gtts.gTTS(text="두 양말의 색이" + color1 +"으로 같아요.", lang="ko")
                tts.save("../sound_data/result_socks.wav")
                playsound.playsound("../sound_data/result_socks.wav")

        # elif (comm == "퍼스널 컬러 알려줘"): #chuga
        #     cap = cv2.VideoCapture(0, cv2.CAP_V4L)

        #     ret, frame = cap.read()
        #     cv2.imwrite("../image_data/face.jpg", frame)

        #     self.picture = cv2.imread("../image_data/face.jpg")
        #     pcs.main(self.picture)
