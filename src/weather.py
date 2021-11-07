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

import requests
import gtts
from bs4 import BeautifulSoup
import re

"""
네이버 날씨 크롤링을 통해, 현재 위치의 날씨와 오전, 오후 강수확률을 갖고와 TTS를 생성해준다.
그리고 강수확률이 50% 이상인지 아닌지를 판단하여, 우산을 챙겨야하는지 아닌지 여부를 판단하고 TTS를 생성해준다.
@author : 이도원
@version 1.0.0
"""

def weather():
    """
    :return now_temperature : 현재 위치의 기온, 날씨에 따른 옷 추천을 하기 위해 return 해준다.
    """

    # 네이버 날씨 크롤링
    html = requests.get('https://weather.naver.com/')

    # parsing 작업
    soup = BeautifulSoup(html.text, 'html.parser')

    # 현재 위치
    now_address = soup.find('strong', {'class': 'location_name'}).text

    weather_box = soup.find('div', {'class': 'weather_area'})

    # 현재 온도
    now_temperature = re.findall('\d+', weather_box.find('strong', {'class': 'current'}).text)[0]

    # 현재 날씨
    today_weather = weather_box.find('span', {'class': 'weather before_slash'}).text


    weekly_box = soup.find('ul', {'class': 'week_list'})

    today_info = weekly_box.find('li', {'class': 'week_item today'})

    rain_list = today_info.findAll('span', {'class': "rainfall"})
    # 오전 강수 확률
    morning_rain_rate = re.findall("\d+", rain_list[0].text)[0]

    # 오후 강수 확률
    afternoon_rain_rate = re.findall("\d+", rain_list[1].text)[0]

    # 최저 기온
    lowest_temperature = re.findall("\d+", today_info.find('span', {'class': 'lowest'}).text)[0]

    # 최고 기온
    highest_temperature = re.findall("\d+", today_info.find('span', {'class': 'highest'}).text)[0]

    temperature_gap = int(highest_temperature) - int(lowest_temperature)

    # 현재 위치의 날씨와 오전, 오후 강수확률 TTS 생성
    tts = gtts.gTTS(
        text=now_address + "의 현재 온도는" + now_temperature + "도." + "현재 날씨는" + today_weather + "." + "오전 강수확률은 "
             + morning_rain_rate + "% 이고," + "오후 강수확률은" + afternoon_rain_rate + "% 입니다.", lang="ko")
    tts.save("../sound_data/weather.wav")

    # 오전 강수 확률 또는 오후 강수 확률이 50% 이상일 경우
    if (int(morning_rain_rate) >= 50 or int(afternoon_rain_rate) >= 50):
        tts = gtts.gTTS(text="비 올 확률이 50% 이상이기 때문에, 우산을 챙기세요", lang="ko")
        tts.save("../sound_data/umbrella.wav")
    # 오전 강수 확률 또는 오후 강수 확률이 50% 미만일 경우
    else:
        tts = gtts.gTTS(text="비 올 확률이 50% 미만이기 때문에, 우산을 안챙기셔도 됩니다", lang="ko")
        tts.save("../sound_data/umbrella.wav")

    # 일교차가 큰 경우(최고 기온, 최저 기온 차이가 10도 이상 일 경우)
    if(temperature_gap >= 10):
        tts = gtts.gTTS(text="일교차가 크니, 겉옷을 챙겨주십시오.", lang="ko")
        tts.save("../sound_data/temperature_gap.wav")
    else:
        tts = gtts.gTTS(text="일교차가 크지 않으니, 겉옷을 안챙기셔도 됩니다", lang="ko")
        tts.save("../sound_data/temperature_gap.wav")

    return now_temperature
