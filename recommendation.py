import pymysql
# 옷 추천 알고리즘

# 걸려있는 옷이 없는 경우
def recommend_without_clothes(temp):
    c_list_top = []
    c_list_bottom = []
    c_list_etc = []
    if temp > 27:
        c_list_top.extend(["나시"])
        c_list_etc.extend(["반팔 원피스", "나시 원피스"])
        c_list_bottom.extend(["반바지", "치마"])
    elif temp > 23:
        c_list_top.extend(["반팔 티셔츠", "반팔 셔츠"])
        c_list_bottom.extend(["반바지", "치마"])
        c_list_etc(["반팔 원피스"])
    elif temp > 20:
        c_list_top.extend(["긴팔 티셔츠", "긴팔 후드티", "긴팔 셔츠"])
        c_list_etc.extend(["긴팔 원피스"])
        c_list_bottom(["슬림핏 바지"])
    elif temp > 17:
        c_list_top.extend(["긴팔 티셔츠", "긴팔 후드티", "긴팔 셔츠"])
        c_list_etc(["긴팔 원피스"])
        c_list_bottom.extend(["일자핏 바지"])
    elif temp > 10:
        c_list_top.extend(["짧은 겉옷", "긴팔 티셔츠", "긴팔 후드티", "긴팔 셔츠"])
        c_list_bottom(["일자핏 바지"])
    elif temp > -5:
        c_list_top.extend(["긴 겉옷", "긴팔 티셔츠", "긴팔 후드티", "긴팔 셔츠"])
        c_list_bottom(["일자핏 바지"])

    return c_list_top, c_list_bottom, c_list_etc

# 걸려있는 옷이 있는 경우
def recommend_with_clothes(temp, matching_list, feature):
    # print(temp, matching_list, feature)
    c_list = []
    recommend_list = []
    c_list_top, c_list_bottom, c_list_etc = recommend_without_clothes(temp)
    # print(c_list_top, c_list_bottom, c_list_etc)
    # 상의인이 하의인지 판별
    if feature in c_list_etc:
        return 0
    elif feature in c_list_top:
        c_list = c_list_bottom
    elif feature in c_list_bottom:
        c_list = c_list_top

    # print(c_list)
    # DB(MYSQL) 연동
    db = pymysql.connect(host='34.64.248.176', user='root', password='kobot10', db='see_ot', charset='utf8')
    cursor = db.cursor(pymysql.cursors.DictCursor)
    # 어울리는 옷 가져오기
    for i in matching_list:
        for j in c_list:
            sql = "SELECT * FROM closet WHERE color = '{}' AND feature ='{}';".format(i,j)
            # print(sql)
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
    return sum

    # # 검색 결과를 리스트로 반영
    # matching_list = result[0]['matching'].split(', ')
