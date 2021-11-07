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

"""
날씨와 어울리는 색을 이용한 추천 알고리즘
@author 김하연
@version 1.0.0
"""

# 옷 추천 알고리즘

def recommend_without_clothes(temp):
    """
    :param temp(int): 기온
    :return c_list_top, c_list_bottom, c_list_etc (list): 기온별 옷 리스트 상의, 하의, 그외
    """
    c_list_top = []  # 상의 리스트
    c_list_bottom = []  # 하의 리스트
    c_list_etc = []  # 원피스 리스트

    # 기온에 따라 맞는 옷차림 리스트에 추가
    if temp > 25:
        c_list_top.extend(["민소매 티셔츠", "반팔 티셔츠"])
        c_list_etc.extend(["반팔 원피스", "민소매 원피스"])
        c_list_bottom.extend(["반바지", "치마"])

    elif temp > 20:
        c_list_top.extend(["긴팔 티셔츠", "긴팔 후드티", "긴팔 셔츠"])
        c_list_etc.extend(["긴팔 원피스"])
        c_list_bottom.extend(["스키니", "일자 긴바지"])

    elif temp > 10:
        c_list_top.extend(["가디건","자켓", "긴팔 티셔츠", "긴팔 후드티", "긴팔 셔츠"])
        c_list_bottom.extend(["일자 긴바지", "스키니", "치마"])
        c_list_etc.extend(["긴팔 원피스"])

    elif temp > -5:
        c_list_top.extend(["코트","패딩", "긴팔 티셔츠", "긴팔 후드티", "긴팔 셔츠"])
        c_list_bottom.extend(["일자 긴바지", "스키니"])
        c_list_etc.extend(["긴팔 원피스"])

    return c_list_top, c_list_bottom, c_list_etc


# 옷 추천 기본
def recommend_with_clothes(temp, feature):
    """
    :param temp(int): 기온
    :param feature(string): 옷 종류
    :return c_list(list): 상하의 구별한 기온에 맞는 옷 리스트
    """
    c_list = []
    c_list_top, c_list_bottom, c_list_etc = recommend_without_clothes(temp)

    # 상의인이 하의인지 판별
    if feature in c_list_etc:
        c_list = c_list_etc
    elif feature in c_list_top:
        c_list = c_list_bottom
    elif feature in c_list_bottom:
        c_list = c_list_top

    return c_list


# 옷장에 해당하는 옷 있는지 찾아보기
def DB(matching_list, c_list, ref):
    """
    :param matching_list(list): 어울리는 색상 리스트
    :param c_list(list): 구별한 옷이 상의면 하의 리스트 하의면 상의 리스트
    :return sum(list): 어울리는 옷 중 옷장에 있는 옷 리스트
    """

    # cred = credentials.Certificate("see-ot-a14fe-firebase-adminsdk-wraln-1e78f4e053.json")

    # firebase_admin.initialize_app(cred, {'databaseURL': 'https://see-ot-a14fe-default-rtdb.firebaseio.com/'})

    # 어울리는 옷 가져오기
    sum = []
    recommend_list = ref.child('my_closet').get()
    print(recommend_list)
    if (matching_list[0] == '모든색'):
        for i,j in recommend_list.items():
            if (i in c_list):
                sum.append(j)
                sum.append(i)
    else:
        for i,j in recommend_list.items():
            for k in j:
                if (k in matching_list):
                    sum.append(k)
                    sum.append(i)

    print(sum)
    if len(sum) != 0:
        return sum
    else:
        return 0
