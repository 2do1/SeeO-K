from sklearn.cluster import KMeans
import numpy as np
import cv2
import color_data
from PIL import Image
from matplotlib import pyplot as plt
"""
배경 제거 및 색상 판별
* @author 김하연 노성환
* @version 1.0.0
"""

# 배경 제거
def extraction(cloth, back):
    """
    :param cloth(numpy array) : 옷 이미지
    :param back(numpy array) : 배경 이미지
    :return background_removed_img(numpy array) : 배경 제거된 옷 이미지
    """

    # 2. remove image backgrounds
    background_removed_img = cloth.copy()

    difference = cv2.subtract(back, cloth) # 배경 이미지와 배경에 옷이 추가된 이미지의 다른 부분만 
    background_removed_img[np.where((difference < [50, 50, 50]).all(axis=2))] = [0, 0, 0]
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
    :param hist(numpy array) : 히스토그램의 값들이 담긴 리스트
    :param centroids(numpy array) : 군집의 중심점
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


def find_color_name():
    """
    :return priority_color_list(list) : 각 RGB값들 중 가장 분포도가 높은 색들의 이름이 담긴 리스트
    """
    cloth = Image.open('../image_data/cloth.jpg').convert('RGB')
    cloth.save('../image_data/cloth.png', 'png')

    back = Image.open('../image_data/back.jpg').convert('RGB')
    back.save('../image_data/back.png', 'png')

    picture = cv2.imread('../image_data/cloth.png')
    back = cv2.imread('../image_data/back.png')
    img = extraction(picture, back)
    cv2.imwrite("../image_data/before.jpg", img)
    
    rgb_list, percent_list = image_color_cluster(img)
    color_name_list = color_data.extract_color(rgb_list)

    # 가장 비중이 큰 옷의 색 확인
    priority_color_list = []
    priority_color_list.append(color_name_list[percent_list.index(max(percent_list))])
    percent_list[percent_list.index(max(percent_list))] = 0
    priority_color_list.append(color_name_list[percent_list.index(max(percent_list))])

    return priority_color_list


def find_matching_color_name(priority_color_list, ref):
    """
    :param priority_color_list(list) : 각 RGB값들 중 가장 분포도가 높은 색들의 이름이 담긴 리스트
    :return matching_list(list) : 옷의 색과 어울리는 색들이 담긴 리스트
    """
    priority_color = priority_color_list[0]
    # DB(MYSQL) 연동
    
    # 어울리는 옷 가져오기
    result = ref.child('matching_color/{}'.format(priority_color)).get()

    return result
