import torch
import torchvision.transforms as transforms
import numpy as np
import os
from PIL import Image
import time

import cv2


def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output


def cosin_metric(x1, x2):
    return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))


def de_preprocess(tensor):
    return tensor * 0.5 + 0.5


hflip = transforms.Compose([
    de_preprocess,
    transforms.ToPILImage(),
    transforms.functional.hflip,
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])


def hflip_batch(imgs_tensor):
    hfliped_imgs = torch.empty_like(imgs_tensor)
    for i, img_ten in enumerate(imgs_tensor):
        hfliped_imgs[i] = hflip(img_ten)

    return hfliped_imgs


def get_transform(cfg):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg.INPUT.RGB_MEAN, std=cfg.INPUT.RGB_STD),
    ])
    return transform


def get_tensor(img_list, transform, input_size):
    images_tensor = []
    images_tensor112 = []
    for img in img_list:
        img = Image.open(img).convert('RGB')
        img_112 = cv2.resize( np.array(img),(112,112) )
        image_tensor = transform(img_112)

        images_tensor.append(image_tensor)
        image_tensor112 = transform(img_112)

        if image_tensor.shape[-1]==300:
            continue
        images_tensor112.append(image_tensor112)
    images_tensor = torch.cat(images_tensor, dim=0)
    images_tensor = images_tensor.view((len(img_list), 3, input_size[0], input_size[1]))
    images_tensor112 = torch.cat(images_tensor112, dim=0)
    images_tensor112 = images_tensor112.view((len(img_list), 3, 112, 112))

    return images_tensor,images_tensor112

def extract_feature(model, image_list, DEVICE, transform, INPUT_SIZE, EMBEDDING_SIZE, batch_size=64, tta=True):
    with torch.no_grad():
        idx = 0
        features = np.zeros([len(image_list), EMBEDDING_SIZE])
        while idx + batch_size <= len(image_list):
            images_tensor,images_tensor112 = get_tensor(image_list[idx: idx + batch_size], transform, INPUT_SIZE)
            if tta:
                fliped = hflip_batch(images_tensor)
                fliped112 = hflip_batch(images_tensor112)
                emb_batch = model(images_tensor.to(DEVICE),images_tensor112.to(DEVICE)) + model(fliped.to(DEVICE),fliped112.to(DEVICE))
                features[idx: idx + batch_size] = l2_norm(emb_batch).detach().cpu().numpy()
            else:
                features[idx: idx + batch_size] = l2_norm(model(images_tensor.to(DEVICE))).cpu().detach().numpy()
            idx += batch_size

        if idx < len(image_list):
            images_tensor,images_tensor112 = get_tensor(image_list[idx:], transform, INPUT_SIZE)
            if tta:
                fliped = hflip_batch(images_tensor)
                fliped112 = hflip_batch(images_tensor112)
                emb_batch = model(images_tensor.to(DEVICE),images_tensor112.to(DEVICE)) + model(fliped.to(DEVICE),fliped112.to(DEVICE))
                features[idx:] = l2_norm(emb_batch).detach().cpu().numpy()
            else:
                features[idx:] = l2_norm(model(images_tensor.to(DEVICE))).cpu().detach().numpy()

    return features

def rank_1(model, data_root, MULTI_GPU, DEVICE, INPUT_SIZE, EMBEDDING_SIZE, transform, batch_size=64, phase='val'):
    if MULTI_GPU and phase == 'val':
        model = model.module  # unpackage model from DataParallel
        model = model.to(DEVICE)
    else:
        model = model.to(DEVICE)
    model.eval()  # switch to evaluation mode

    start = time.time()
    with open(os.path.join(data_root, 'query.txt')) as qf:
        df_lines = qf.readlines()
    with open(os.path.join(data_root, 'gallery.txt')) as gf:
        gf_lines = gf.readlines()

    query_id_list = [line.strip('\n').split(' ')[1] for line in df_lines]
    query_img_list = [line.strip('\n').split(' ')[0] for line in df_lines]
    gallery_id_list = [line.strip('\n').split(' ')[1] for line in gf_lines]
    gallery_img_list = [line.strip('\n').split(' ')[0] for line in gf_lines]

    query_features = extract_feature(model, query_img_list, DEVICE, transform, INPUT_SIZE, EMBEDDING_SIZE, batch_size)
    gallery_features = extract_feature(model, gallery_img_list, DEVICE, transform, INPUT_SIZE, EMBEDDING_SIZE, batch_size)
    distance_matrix = np.matmul(query_features, gallery_features.T)

    q_id = np.array(query_id_list)
    g_id = np.array(gallery_id_list)
    result = g_id[np.argmax(distance_matrix, axis=1)]
    acc = sum(q_id == result) * 1.0 / len(query_id_list)
    # index = np.arange(0, len(query_id_list))
    # print(set(q_id) - set(q_id[index[q_id == result]]))
    end = time.time()
    total_time = end - start
    return total_time, acc




def extract_feature_dict(model, image_list, DEVICE, transform, INPUT_SIZE, EMBEDDING_SIZE, batch_size=64, tta=True):
    with torch.no_grad():
        idx = 0
        features = np.zeros([len(image_list), EMBEDDING_SIZE])
        while idx + batch_size <= len(image_list):
            images_tensor,images_tensor112 = get_tensor(image_list[idx: idx + batch_size], transform, INPUT_SIZE)
            if tta:
                fliped = hflip_batch(images_tensor)
                fliped112 = hflip_batch(images_tensor112)
                emb_batch = model(images_tensor.to(DEVICE),images_tensor112.to(DEVICE)) + model(fliped.to(DEVICE),fliped112.to(DEVICE))
                features[idx: idx + batch_size] = l2_norm(emb_batch).detach().cpu().numpy()
            else:
                features[idx: idx + batch_size] = l2_norm(model(images_tensor.to(DEVICE))).cpu().detach().numpy()
            idx += batch_size

        if idx < len(image_list):
            images_tensor,images_tensor112 = get_tensor(image_list[idx:], transform, INPUT_SIZE)
            if tta:
                fliped = hflip_batch(images_tensor)
                fliped112 = hflip_batch(images_tensor112)
                emb_batch = model(images_tensor.to(DEVICE),images_tensor112.to(DEVICE)) + model(fliped.to(DEVICE),fliped112.to(DEVICE))
                features[idx:] = l2_norm(emb_batch).detach().cpu().numpy()
            else:
                features[idx:] = l2_norm(model(images_tensor.to(DEVICE))).cpu().detach().numpy()

    return features

import json
import shutil

def mkdir(path):
    if os.path.exists(path):
        return
    os.mkdir(path)


def creat_face_json(model, query_data_root,gallery_data_root, MULTI_GPU, DEVICE, INPUT_SIZE, EMBEDDING_SIZE, transform,pic_save_path, batch_size=64, phase='val'):
    if MULTI_GPU and phase == 'val':
        model = model.module  # unpackage model from DataParallel
        model = model.to(DEVICE)
    else:
        model = model.to(DEVICE)
    model.eval()  # switch to evaluation mode

    Query_IMG_LIST = os.listdir(query_data_root)
    Query_img_list = [os.path.join(query_data_root,IMG_) for IMG_ in Query_IMG_LIST]
    query_features = extract_feature_dict(model, Query_img_list, DEVICE, transform, INPUT_SIZE, EMBEDDING_SIZE, batch_size)


    Gallery_IMG_LIST = os.listdir(gallery_data_root)
    Gallery_img_list = [os.path.join(gallery_data_root,IMG_) for IMG_ in Gallery_IMG_LIST]
    gallery_features = extract_feature_dict(model, Gallery_img_list, DEVICE, transform, INPUT_SIZE, EMBEDDING_SIZE, batch_size)
    distance_matrix = np.matmul(query_features, gallery_features.T)



    mkdir(pic_save_path)
    distance_matrix = (0.5-0.5*distance_matrix).T
    for distance_matrix_,img_path in zip(distance_matrix,Gallery_img_list):
        MinSort = np.argsort(distance_matrix_)
        min_distance = distance_matrix_[MinSort[0]]
        if min_distance>0.5:
            continue
        else:
            Min_distance = distance_matrix_[MinSort[:5]].tolist()
            MinSort = MinSort[:5].tolist()
            Query_img_list_Sub = [Query_img_list[MinSort_] for MinSort_ in MinSort]
            # Query_img_list_Sub.append(img_path)

            pic_save_path_ = os.path.join(pic_save_path, os.path.basename(img_path)[:-4])
            mkdir(pic_save_path_)
            shutil.copy(img_path, os.path.join(pic_save_path_,os.path.basename(img_path)))
            for Query_img_list_Sub_,Min_distance_ in zip(Query_img_list_Sub,Min_distance):
                if Min_distance_>0.5:
                    continue
                shutil.copy(Query_img_list_Sub_,os.path.join(pic_save_path_,str(Min_distance_))+'.jpg')



    JSON_DICT = {}
    for  feature,IMG_NAME in zip(list(query_features),Query_img_list_Sub):
        IMG_NAME = os.path.basename(IMG_NAME)
        JSON_DICT[IMG_NAME] = feature.tolist()

    JSON = json.dumps(JSON_DICT, indent=4)
    with open(r'face_json.json','w') as f:
        f.write(JSON)


    JSON_DICT = {}




