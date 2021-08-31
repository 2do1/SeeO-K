"""
Fashion MNIST의 부족한 이미지 데이터셋 보충을 위해 이미지 크롤링을 하는 코드이다.
Fashion MNIST와 마찬가지로 Zalando에서 이미지를 얻었다.
0~59999.png는 기존 Fashion MNIST의 train dataset이고 60000.png부터는 직접 Zalando에서 크롤링한 custom dataset이다.
"""

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
import urllib.request

SCROLL_PAUSE_TIME = 1
PAGE_NUM = 1 # 크롤링 할 페이지 조절 변수(시작 페이지)
COUNT = 64298 # 이미지 저장할 이름을 나타내는 변수(시작 이름)
driver = webdriver.Chrome

while PAGE_NUM <= 54:
    # 크롤링 할 사이트. q= 뒤에 원하는 검색어를 입력.
    driver.get("https://www.zalando.co.uk/women/?q=shorts&p=" + str(PAGE_NUM))

    # 페이지 스크롤 끝까지 내리는 코드
    last_height = driver.execute_script("return document.body.scrollHeight")

    while True:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

        time.sleep(SCROLL_PAUSE_TIME)

        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height

    # class 이용하여 원하는 이미지들을 선택
    images = driver.find_elements_by_css_selector("._6uf91T.z-oVg8.u-6V88.ka2E9k.uMhVZi.FxZV-M._2Pvyxl.JT3_zV.EKabf7.mo6ZnF._1RurXL.mo6ZnF.PZ5eVw")

    # 각 이미지들의 url을 얻은 후 저장
    for image in images:
        try:
            imgUrl = image.get_attribute("src")
            urllib.request.urlretrieve(imgUrl, 'C:\\Users\\rosy0\\GITHUB\\Oss\\Crawling\\crawling_image\\' + str(COUNT) + ".png")
            COUNT += 1
        except:
            pass

    PAGE_NUM += 1
