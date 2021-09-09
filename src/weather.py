"""
네이버 날씨 크롤링을 통해, 현재 위치의 날씨와 오전, 오후 강수확률을 갖고와, TTS를 생성해준다.
그리고 강수확률이 50% 이상인지 아닌지를 판단하여, 우산을 챙겨야하는지 아닌지 여부를 판단하고 TTS를 생성해준다.

@author : 이도원
@version 1.0.0
"""
    
import requests
import gtts
from bs4 import BeautifulSoup

def weather():
    """
    :return now_temperature : 현재 위치의 기온, 날씨에 따른 옷 추천을 하기 위해 return 해준다.
    """

    # 네이버 날씨 크롤링
    html = requests.get('https://search.naver.com/search.naver?query=날씨')  
    
    # parsing 작업
    soup = BeautifulSoup(html.text, 'html.parser')  

    weather_box = soup.find('div', {'class': 'weather_box'})

    # 현재 위치
    now_address = weather_box.find('span', {'class': 'btn_select'}).text
    # 현재 온도
    now_temperature = weather_box.find('span', {'class': 'todaytemp'}).text
    # 현재 날씨
    today_weather = weather_box.find('p', {'class': 'cast_txt'}).text

    weekly_box = soup.find('div', {'class': 'table_info weekly _weeklyWeather'})

    today_rain = weekly_box.findAll('li')

    rain_list = today_rain[0].find_all('span', {'class': 'num'})
    # 오전 강수 확률
    morning_rain_rate = rain_list[0].text
    # 오후 강수 확률
    afternoon_rain_rate = rain_list[1].text
    
    # 현재 위치의 날씨와 오전, 오후 강수확률 TTS 생성
    tts = gtts.gTTS(text=now_address + "의 현재 온도는" + now_temperature + "도." + "현재 날씨는" + today_weather + "." + "오전 강수확률은 "
                         + morning_rain_rate + "% 이고," + "오후 강수확률은" + afternoon_rain_rate + "% 입니다.", lang="ko")
    tts.save("../sound_data/weather.wav")

    # 오전 강수 확률 또는 오후 강수 확률이 50% 이상일 경우
    if(int(morning_rain_rate) >= 50 or int(afternoon_rain_rate) >= 50):
        tts = gtts.gTTS(text = "비 올 확률이 50% 이상이기 때문에, 우산을 챙기세요", lang="ko")
        tts.save("../sound_data/umbrella.wav")
    # 오전 강수 확률 또는 오후 강수 확률이 50% 미만일 경우
    else:
        tts = gtts.gTTS(text = "비 올 확률이 50% 미만이기 때문에, 우산을 안챙기셔도 됩니다", lang="ko")
        tts.save("../sound_data/umbrella.wav")
        
    return now_temperature 
