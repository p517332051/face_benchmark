
import pandas as pd
import requests
import os
def mkdir(IMG_SAVE_PATH):
    if os.path.exists(IMG_SAVE_PATH):
        return None
    else:
        os.mkdir(IMG_SAVE_PATH)
        print("creat ",IMG_SAVE_PATH)

def PULL_IMG_from_CSV(IMG_CSV,IMG_SAVE_PATH):
    IMG_PATH = pd.read_excel(IMG_CSV)
    face_pictures = IMG_PATH['face_picture']
    face_ids = IMG_PATH['face_id']
    devices  = IMG_PATH['devices']
    mkdir(IMG_SAVE_PATH)
    for indxe,(face_picture, face_id,device) in enumerate(zip(face_pictures, face_ids,devices) ):
        if device!='二号设备':
            continue
        print(face_picture)
        response = requests.get(face_picture)
        img = response.content
        img_path = os.path.join(IMG_SAVE_PATH, face_id) +"_"+str(indxe)+ ".jpg"
        with open(img_path, 'wb') as f:
            f.write(img)

IMG_PATH = "/Users/hongwei/魔镜人脸抓拍.xlsx"
IMG_SAVE_PATH = "/Users/hongwei/reseach/魔镜人脸抓拍_二号设备_all"
PULL_IMG_from_CSV(IMG_PATH,IMG_SAVE_PATH)

