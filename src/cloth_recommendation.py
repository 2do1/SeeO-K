import pymysql


# 옷 추천 알고리즘

def recommend_without_clothes(temp):
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