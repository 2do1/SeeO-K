import pymysql
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
    c_list_top = [] #상의 리스트
    c_list_bottom = [] #하의 리스트
    c_list_etc = [] #원피스 리스트

    # 기온에 따라 맞는 옷차림 리스트에 추가
    if temp > 25:
        c_list_top.extend(["나시", "반팔 티셔츠", "반팔 셔츠"])
        c_list_etc.extend(["반팔 원피스", "나시 원피스"])
        c_list_bottom.extend(["반바지", "치마"])

    elif temp > 20:
        c_list_top.extend(["긴팔 티셔츠", "긴팔 후드티", "긴팔 셔츠"])
        c_list_etc.extend(["긴팔 원피스"])
        c_list_bottom.extend(["슬림핏 바지", "일자핏 바지"])
        
    elif temp > 10:
        c_list_top.extend(["짧은 겉옷", "긴팔 티셔츠", "긴팔 후드티", "긴팔 셔츠"])
        c_list_bottom.extend(["일자핏 바지"])
        c_list_etc.extend(["긴팔 원피스"])
        
    elif temp > -5:
        c_list_top.extend(["긴 겉옷", "긴팔 티셔츠", "긴팔 후드티", "긴팔 셔츠"])
        c_list_bottom.extend(["일자핏 바지"])
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
        return 0
    elif feature in c_list_top:
        c_list = c_list_bottom
    elif feature in c_list_bottom:
        c_list = c_list_top
    
    return c_list


# 옷장에 해당하는 옷 있는지 찾아보기
def DB(matching_list, c_list):
    """
    :param matching_list(list): 어울리는 색상 리스트
    :param c_list(list): 구별한 옷이 상의면 하의 리스트 하의면 상의 리스트
    :return sum(list): 어울리는 옷 중 옷장에 있는 옷 리스트
    """
   
    # DB(MYSQL) 연동
    recommend_list = []
    db = pymysql.connect(host='35.232.131.79', user='root', password='kobot10', db='see_ot', charset='utf8')
    cursor = db.cursor(pymysql.cursors.DictCursor)
    # 어울리는 옷 가져오기
    if (matching_list == ["모든색"]): # 어울리는 색이 모든색인 경우 옷의 종류만 확인하여 불러옴
        for j in c_list:
            sql = "SELECT * FROM closet WHERE feature ='{}';".format(j)
            cursor.execute(sql)
            recommend_list.append(cursor.fetchall())
        
    else: # 어울리는 색과 맞는 옷을 기준으로 옷장에 있는지 탐색
        for i in matching_list:
            for j in c_list:
                sql = "SELECT * FROM closet WHERE color = '{}' AND feature ='{}';".format(i, j)
                cursor.execute(sql)
                recommend_list.append(cursor.fetchall())
    print(recommend_list)
    sum = []
    for i in range(len(recommend_list)):
        try:
            sum.append(recommend_list[i][0]['color'])
            sum.append(recommend_list[i][0]['feature'])
        except IndexError:
            pass
    print(sum)
    if len(sum)!=0 :
        return sum
    else:
        return 0
