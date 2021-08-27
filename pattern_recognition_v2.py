# -*- coding: utf-8 -*-

import cv2
import numpy as np
import tensorflow as tf

# 패턴 인식
def pred_pattern(fw, bg):
    # 배경 제거
    image_fw = cv2.imread(fw)
    image_bg = cv2.imread(bg)
    # load images
    image_fw = cv2.resize(image_fw, (224, 224))
    # cv2_imshow(image1)

    image_bg = cv2.resize(image_bg, (224, 224))
    difference = cv2.subtract(image_bg, image_fw)
    image_fw[np.where((difference == [0,0,0]).all(axis=2))] = [0,0,0]
    # cv2_imshow(image_fw)

    # 배경 제거된 이미지를 텐서로 변환
    input_image = tf.convert_to_tensor(image_fw, tf.dtypes.float32)
    # print(input_image)
    input_image = tf.expand_dims(input_image, 0)
    # print(input_image)

    # 모델 불러오기
    interpreter = tf.lite.Interpreter(model_path = "./model/model_unquant.tflite")
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_shape = input_details[0]['shape']
    # print(input_shape)
    interpreter.set_tensor(input_details[0]['index'], input_image)

    # 추론하기
    interpreter.invoke()

    labels = ["animal", "checkered","floral", "dotted", "striped", "solid"]
    output_data = interpreter.get_tensor(output_details[0]['index'])
    print(output_data)
    print(output_data)
    result = labels[np.argmax(output_data)]
    # print(labels[np.argmax(output_data)])

    # 추론 결과
    return result

if __name__ == '__main__':
    picture = "./origin.png"
    back = "./back.png"
    picture_bn = cv2.imread(picture)
    back_bn = cv2.imread(back)

    result = pred_pattern(picture_bn, back_bn)
    print(result)
