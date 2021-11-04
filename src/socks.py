from sklearn.cluster import KMeans
import numpy as np
import cv2
from matplotlib import pyplot as plt
import color_data
from PIL import Image
import white

"""
양말 색상 판별 및 짝이 맞는지 확인
@author 노성환
@version 1.0.0
"""
# 배경 제거
def extraction(image1, image2):
    """
    :param cloth(numpy array) : 옷 이미지
    :param back(numpy array) : 배경 이미지
    :return background_removed_img(numpy array) : 배경 제거된 옷 이미지
    """
    background_removed_img = image1.copy()

    difference = cv2.subtract(image2, image1)
    background_removed_img[np.where((difference < [70, 70, 70]).all(axis=2))] = [0, 0, 0]
    return background_removed_img


# 히스토그램 평균화
def centroid_histogram(clt):
    """
    :param clt(array) : 클러스터 픽셀
    :return hist(numpy array) :  히스토그램의 값들이 담긴 리스트
    """
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins=numLabels)
    hist = hist.astype("float")
    hist /= hist.sum()
    return hist


# 추출할 색의 범위 지정 및 각 색의 RGB 값과 그 점유율을 리스트로 반환, 히스토그램으로 표현
def plot_colors(hist, centroids):
    """
    :param hist() : 옷 종류
    :param centroids(string) : 옷 색상
    :return bar(numpy array) : 각 색의 분포도를 보여주기 위한 표
    :return percent_list(list) : 각 색의 분포도가 담긴 리스트
    :return color_list(list)  : 각 색의 rgb 값이 담긴 리스트
    """
    percent_list = []
    color_list = []

    # 배경은 퍼센트 부분에 포함x
    for (percent, color) in zip(hist, centroids):
        if int(color.astype("uint8").tolist()[0]) == int(color.astype("uint8").tolist()[1]) == int(
                color.astype("uint8").tolist()[2]) == 0:
            pass
        else:
            percent_list.append(percent)
    bar = np.zeros((50, int(300 * (sum(percent_list))), 3), dtype="uint8")
    startX = 0

    # histogram에서 배경색 제외
    for (percent, color) in zip(hist, centroids):
        endX = startX + (percent * 300)
        if int(color.astype("uint8").tolist()[0]) == int(color.astype("uint8").tolist()[1]) == int(
                color.astype("uint8").tolist()[2]) == 0:
            pass
        else:
            cv2.rectangle(bar, (int(startX), 0), (int(endX), 50), color.astype("uint8").tolist(), -1)
            startX = endX
            color_list.append(color)
    return bar, percent_list, color_list


# 옷의 색을 k개의 색으로 추려 표현
def image_color_cluster(image, k=5):
    """
    :param image(numpy array): 추출된 옷
    :param k(int) : 군집화할 개수
    :return c_list(list) : 각 색의 rgb 값이 담긴 리스트
    :return p_list(list) : 각 색의 분포도가 담긴 리스트
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.reshape((image.shape[0] * image.shape[1], 3))
    clt = KMeans(n_clusters=k)
    clt.fit(image)
    hist = centroid_histogram(clt)
    bar, p_list, c_list = plot_colors(hist, clt.cluster_centers_)
    return c_list, p_list


def find_color_name(cloth, bg):
    """
    :param cloth(numpy array) : 양말이 걸린 사진
    :param bg(numpy array) : 시작할 때 배경 사진
    :return priority_color1(string) : 양말 왼쪽의 색
    :return priority_color1(string) : 양말 오른쪽의 색
    """
    img = extraction(cloth, bg)
    img = white.white_b(img)

    # 이미지 절반으로 자르기
    h, w, c = img.shape
    cropped_img1 = img[0:h // 2, 0:w]
    cropped_img2 = img[h//2:h, 0:w]
    cv2.imwrite("../image_data/socks_left.jpg", cropped_img1)  # 사진 저장
    cv2.imwrite("../image_data/socks_right.jpg", cropped_img2)  # 사진 저장

    rgb_list, percent_list = image_color_cluster(cropped_img1)
    color_name_list1 = color_data.extract_color(rgb_list)

    # 가장 비중이 큰 옷의 색 확인
    priority_color1 = color_name_list1[percent_list.index(max(percent_list))]
    rgb_list, percent_list = image_color_cluster(cropped_img2)
    color_name_list2 = color_data.extract_color(rgb_list)

    # 가장 비중이 큰 옷의 색 확인
    priority_color2 = color_name_list2[percent_list.index(max(percent_list))]
    print(priority_color1, priority_color2)
    return priority_color1, priority_color2
