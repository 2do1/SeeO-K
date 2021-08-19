import requests # 웹사이트의 내용 가져오기
from bs4 import BeautifulSoup # html로 Parsing 작업

def Weather():
    html = requests.get('https://search.naver.com/search.naver?query=날씨')  # 네이버 날씨 이용

    soup = BeautifulSoup(html.text, 'html.parser')  # parsing 작업

    weather_box = soup.find('div', {'class': 'weather_box'})

    now_address = weather_box.find('span', {'class': 'btn_select'}).text
    # print('현재 위치: ' + now_address)

    now_temperature = weather_box.find('span', {'class': 'todaytemp'}).text
    # print('현재 온도: ' + now_temperature + '℃')

    today_weather = weather_box.find('p', {'class': 'cast_txt'}).text
    # print('현재 날씨: ' + today_weather)

    weekly_box = soup.find('div', {'class': 'table_info weekly _weeklyWeather'})
    today_rain_rate = weekly_box.findAll('li')
    morning_rain_rate = today_rain_rate[0].find('span', {'class': 'point_time morning'}).text.strip()
    afternoon_rain_rate = today_rain_rate[0].find('span', {'class': 'point_time afternoon'}).text.strip()
    # print("오늘 오전 ", morning_rain_rate)
    # print("오늘 오후 ", afternoon_rain_rate)

    return now_temperature, today_weather, morning_rain_rate, afternoon_rain_rate
