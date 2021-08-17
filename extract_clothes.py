import numpy as np
import cv2
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
import pymysql
import color_data


# 배경 제거
def extraction(image1, image2):
    difference = cv2.subtract(image2, image1)
    # result = cv2.bitwise_(image1,difference)
    Conv_hsv_Gray = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
    image1[Conv_hsv_Gray == 0] = [0, 0, 0]
    return image1


# 히스토그램 평균화
def centroid_histogram(clt):
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins=numLabels)
    hist = hist.astype("float")
    hist /= hist.sum()
    return hist


# 추출할 색의 범위 지정 및 각 색의 RGB 값과 그 점유율을 리스트로 반환, 히스토그램으로 표현
def plot_colors(hist, centroids):
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
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.reshape((image.shape[0] * image.shape[1], 3))
    clt = KMeans(n_clusters=k)
    clt.fit(image)
    hist = centroid_histogram(clt)
    bar, p_list, c_list = plot_colors(hist, clt.cluster_centers_)
    print(p_list)
    print(c_list)
    plt.figure()
    plt.axis("off")
    plt.imshow(bar)
    plt.show()
    return c_list, p_list


if __name__ == '__main__':
    picture = cv2.imread('data_T.png')
    back = cv2.imread('wall.png')
    img = extraction(picture, back)
    rgb_list, percent_list = image_color_cluster(img)
    color_name_list = color_data.extract_color(rgb_list)

    # 가장 비중이 큰 옷의 색 확인
    priority_color = color_name_list[percent_list.index(max(percent_list))]
    print("====", priority_color, "====")

    # 추출된 옷 및 색 분포도 확인
    plt.subplot(121), plt.imshow(cv2.cvtColor(cv2.imread('data_T.png'), cv2.COLOR_BGR2RGB))
    plt.title("before"), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("after"), plt.xticks([]), plt.yticks([])
    plt.show()

    # DB(MYSQL) 연동
    db = pymysql.connect(host='34.64.248.176', user='root', password='kobot10', db='see_ot', charset='utf8')
    cursor = db.cursor(pymysql.cursors.DictCursor)
    # 어울리는 색 검색
    sql = "SELECT matching FROM color WHERE color = '{}';".format(priority_color)
    cursor.execute(sql)
    result = cursor.fetchall()
    # 검색 결과를 리스트로 반영
    matching_list = result[0]['matching'].split(', ')
    for i in matching_list:
        print(i)
