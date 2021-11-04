"""
closet table에 구별한 옷을 저장
* @author 김하연 노성환
* @version 1.0.0
"""
def insert_clothes(feature, color, ref):
    """
    :param feature(string) : 옷 종류
    :param color(string) : 옷 색상
    """
    # cred = credentials.Certificate("see-ot-a14fe-firebase-adminsdk-wraln-1e78f4e053.json")
    # firebase_admin.initialize_app(cred, {'databaseURL': 'https://see-ot-a14fe-default-rtdb.firebaseio.com/'})

    my_closet_ref = ref.child("my_closet")
   

    data = {feature : [color]}

    kind = list(data.keys())[0]

    kind_list = list(my_closet_ref.get().keys())

    color_list = my_closet_ref.child(kind).get()

    for i in range(len(kind_list)):
        if(kind == kind_list[i]):
            color_list.append(data[kind][0])

            color_list = set(color_list)
            color_list = list(color_list)
            data[kind][0] = color_list
            data = {kind : data[kind][0]}

            my_closet_ref.update(data)
        else:
            my_closet_ref.update(data)

