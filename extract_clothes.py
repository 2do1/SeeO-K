import cv2
import matplotlib.pyplot as plt
def extraction(image1, image2):
    difference = cv2.subtract(image2, image1)
    # result = cv2.bitwise_(image1,difference)
    Conv_hsv_Gray = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
    image1[Conv_hsv_Gray == 0] = [0, 0, 0]
    return image1

if __name__ == '__main__':
    picture = cv2.imread('data_T.png')
    back = cv2.imread('wall.png')
    img = extraction(picture,back)
    plt.subplot(121), plt.imshow(cv2.cvtColor(cv2.imread('data_T.png'),cv2.COLOR_BGR2RGB))
    plt.title("before"), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    plt.title("after"), plt.xticks([]), plt.yticks([])
    plt.show()


