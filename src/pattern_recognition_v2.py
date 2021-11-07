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

# -*- coding: utf-8 -*-

'''
* google teachable machine으로 학습한 tensorflow lite 모델을 이용하여 옷의 패턴을 추론하는 기능을 하는 코드
* 
* @author 송수인
* @version 1.0
'''

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

# 패턴 인식
def pred_pattern(fw, bg):
    # 배경 제거
    cloth = Image.open(fw).convert('RGB')
    cloth.save('../image_data/cloth.png', 'png')

    back = Image.open(bg).convert('RGB')
    back.save('../image_data/back.png', 'png')

    image_fw = cv2.imread('../image_data/cloth.png')
    image_bg = cv2.imread('../image_data/back.png')
    background_removed_img = image_fw.copy()

    # 그림자 제거
    difference = cv2.subtract(image_bg, image_fw)
    background_removed_img[np.where((difference < [60, 60, 60]).all(axis=2))] = [0, 0, 0]

    # 효과적인 자르기를 위하여, 바운딩 박스를 활용
    cropped_img = Image.fromarray(background_removed_img)
    bbox = cropped_img.getbbox()
    cropped_img = cropped_img.crop(bbox)

    # 모델에 이미지를 알맞게 올리기 위하여 224, 224로 변환하고 노멀라이제이션 및 플롯형 변환 실행
    cropped_img = cropped_img.rotate(90)
    cropped_img = ImageOps.fit(cropped_img, (224, 224), Image.ANTIALIAS)
    cropped_img = np.asarray(cropped_img)
    #cropped_img = (cropped_img.astype(np.float32) / 127.0) - 1
    #cv2.imwrite('../image_data/d.png', cropped_img)

    # 배경 제거된 이미지를 텐서로 변환
    input_image = tf.convert_to_tensor(cropped_img, tf.dtypes.float32)
    input_image = tf.expand_dims(input_image, 0)

    # 모델 불러오기
    interpreter = tf.lite.Interpreter(model_path = "../model/model_unquant.tflite")
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_shape = input_details[0]['shape']
    interpreter.set_tensor(input_details[0]['index'], input_image)

    # 추론하기
    interpreter.invoke()

    labels = ["체크 무늬", "꽃 무늬", "줄 무늬", "민 무늬"]
    output_data = interpreter.get_tensor(output_details[0]['index'])
    print("pattern predicted list", output_data)
    result = labels[np.argmax(output_data)]

    return result

if __name__ == "__main__":
    rt = pred_pattern("cloth.jpg", "back.jpg")
    print(rt)
