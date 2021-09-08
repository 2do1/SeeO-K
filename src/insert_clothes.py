import pymysql

def insert_clothes(feature, color):
    db = pymysql.connect(host='35.232.131.79', user='root', password='kobot10', db='see_ot', charset='utf8')
    cursor = db.cursor(pymysql.cursors.DictCursor)
    # 어울리는 색 검색
    
    sql = "INSERT INTO closet(color, feature) SELECT '{}', '{}' FROM DUAL WHERE NOT EXISTS (SELECT color, feature FROM closet WHERE color='{}' AND feature='{}');".format(color, feature, color, feature)
    cursor.execute(sql)
    db.commit()
    db.close()  # 연결 닫기