# -*- encoding:utf-8 -*-
'''
@Author:LiYue
@Name:data_process.py
@Data:7/22/19  14:00 PM
@
'''

import os
import shutil
from tqdm import tqdm
import glob

# get image file list with one data set, usually for train data
def get_img_file(img_dir, file_list, folder_depth=3):
    '''
    :param img_dir:
    :param file_list:
    :param folder_depth:
    :return:
    '''
    fl = open(file_list, 'w+')
    count = -1
    if folder_depth == 3:
        for folder in os.listdir(img_dir):
            subfolder = os.path.join(img_dir, folder)
            if os.path.isdir(subfolder):
                for id_folder in os.listdir(subfolder):
                    sub_idfolder = os.path.join(subfolder, id_folder)
                    if 'DS_Store' not in id_folder and len(os.listdir(sub_idfolder)) > 0:
                        count += 1
                        for img in os.listdir(sub_idfolder):
                            if 'DS_Store' not in img:
                                img_file = os.path.join(folder, id_folder, img)
                                id = str(count)
                                fl.write(img_file + ' ' + id + '\n')
    else:
        for id_folder in os.listdir(img_dir):
            sub_idfolder = os.path.join(img_dir, id_folder)
            if 'DS_Store' not in id_folder and len(os.listdir(sub_idfolder)) > 0:
                count += 1
                for img in os.listdir(sub_idfolder):
                    if 'DS_Store' not in img:
                        img_file = os.path.join(id_folder, img)
                        id = str(count)
                        fl.write(img_file + ' ' + id + '\n')

    fl.close()


def compare_img_file(img_folder_list, compare_file):
    fd = open(compare_file, 'w+')

    # for img_folder in img_folder_list:
    #     for id in os.listdir(img_folder):
    #         id_folder = os.path.join(img_folder, id)
    #         if 'DS_Store' not in id_folder and len(os.listdir(id_folder)) == 2:
    #             img_list = os.listdir(id_folder)
    #             if 'true' in img_folder:
    #                 img_file = os.path.join(id_folder, img_list[0]) + ' ' + os.path.join(id_folder, img_list[1]) + ' ' + '1'
    #             else:
    #                 img_file = os.path.join(id_folder, img_list[0]) + ' ' + os.path.join(id_folder, img_list[1]) + ' ' + '0'
    #             fd.write(img_file + '\n')
    img_test_list = None
    img_base_list = None
    folder_test = None
    folder_base = None
    for img_folder in img_folder_list:
        if 'test' in img_folder:
            folder_test = img_folder
            img_test_list = os.listdir(folder_test)
        else:
            folder_base = img_folder
            img_base_list = os.listdir(folder_base)

    for img_test in img_test_list:
        for img_base in img_base_list:
            id_test = img_test.split('_')[0]
            id_base = img_base.split('_')[0]
            img_test_file = os.path.join(folder_test, img_test)
            img_base_file = os.path.join(folder_base, img_base)
            if id_test == id_base:
                flag = str(1)
            else:
                flag = str(0)
            img_file = img_test_file + ' ' + img_base_file + ' ' + flag + '\n'
            fd.write(img_file)

    fd.close()


# get query and gallery file with one data set, usually for test set
def query_and_gallery(img_folder, query_file, gallery_file, gallary_folder=None,depth=2):
    '''
    :param img_folder:
    :param query_file:
    :param gallery_file:
    :return:
    '''
    query_file = os.path.join(img_folder, query_file)
    gallery_file = os.path.join(img_folder, gallery_file)
    qf = open(query_file, 'w+')
    gf = open(gallery_file, 'w+')

    count = -1
    if depth == 2:
        for id in os.listdir(img_folder):
            id_folder = os.path.join(img_folder, id)

            if '.DS_Store' not in id and os.path.isdir(id_folder) and len(os.listdir(id_folder)) > 1:
                count += 1
                for i, img in enumerate(os.listdir(id_folder)):
                    if not img.startswith('.'):
                        label = str(count)
                        line = img_folder + '/' + id + '/' + img + ' ' + label + '\n'
                        if i % 2 == 0:
                            qf.write(line)
                        else:
                            gf.write(line)

        if gallary_folder is not None:
            for img in os.listdir(gallary_folder):
                if not img.startswith('.'):
                    line = gallary_folder + '/' + img + ' ' + '-1' + '\n'
                    gf.write(line)

    else:
        img_folders = glob.glob(img_folder+'*/*')
        count = 0
        for folder in img_folders:
            for idx,img in enumerate(glob.glob(folder+"/*.jpg")):
                if idx%2:
                    qf.write(img+ ' '+str(count)+'\n')
                else:
                    gf.write(img+ ' '+str(count)+'\n')
                if idx==2:
                    break
            count+=1
    qf.close()
    gf.close()


# move data from data_source to data_dest
def data_merge(data_source, data_dest):
    '''
    :param data_source:
    :param data_dest:
    :return:os.path.join(data_dest, id_folder)
    '''

    assert os.path.exists(data_dest)

    for id_folder in tqdm(os.listdir(data_source)):
        sub_idfolder = os.path.join(data_source, id_folder)
        if 'DS_Store' not in id_folder and len(os.listdir(sub_idfolder)) > 0:
            if not os.path.exists(os.path.join(data_dest, id_folder)):
                os.mkdir(os.path.join(data_dest, id_folder))
            for img in os.listdir(sub_idfolder):
                if 'DS_Store' not in img :
                    source_img_file = os.path.join(sub_idfolder, img)
                    dest_img_file = os.path.join(data_dest, id_folder, img)
                    if not os.path.exists(dest_img_file):
                        shutil.copyfile(source_img_file, dest_img_file)
                    # else:
                        # print("{} exists".format(dest_img_file))


def select_img(data_source, data_dest=None):
    if data_dest and not os.path.exists(data_dest):
        os.mkdir(data_dest)

    # folder_list = os.listdir(data_source)
    for id_folder in tqdm(os.listdir(data_source)):
        sub_idfolder = os.path.join(data_source, id_folder)
        if 'DS_Store' not in id_folder and len(os.listdir(sub_idfolder)) > 0:
            for i, img in enumerate(os.listdir(sub_idfolder)):
                if 'DS_Store' not in img and i % 4 != 0:
                    source_img_file = os.path.join(sub_idfolder, img)
                    # dest_img_file = os.path.join(data_dest, img)
                    # shutil.copyfile(source_img_file, dest_img_file)
                    os.remove(source_img_file)


def merge_same_id(data_root, cluster_file):
    with open(cluster_file, 'r') as fp:
        lines = fp.readlines()
        for line in lines:
            id_list = line.strip().split(' ')[1:]
            assert len(id_list) > 1
            dest_id = id_list[0]
            dest_id_folder = os.path.join(data_root, dest_id)
            for id in id_list[1:]:
                id_folder = os.path.join(data_root, id)
                for img_file in os.listdir(id_folder):
                    source_img_file = os.path.join(id_folder, img_file)
                    dest_img_file = os.path.join(dest_id_folder, img_file)
                    if not os.path.exists(dest_img_file):
                        shutil.move(source_img_file, dest_img_file)

    for id in os.listdir(data_root):
        id_folder = os.path.join(data_root, id)
        if len(os.listdir(id_folder)) == 0:
            shutil.rmtree(id_folder)


def get_iamges_by_id(data_source, data_dest):
    id_img_dict = {}
    for img in os.listdir(data_source):
        id = img.split('_')[0]
        if id in id_img_dict.keys():
            id_img_dict[id].append(img)
        else:
            id_img_dict[id] = []
            id_img_dict[id].append(img)

    for id in id_img_dict.keys():
        if len(id_img_dict[id]) > 1:
            dest_id_folder = os.path.join(data_dest, id)
            if not os.path.exists(dest_id_folder):
                os.mkdir(dest_id_folder)
            for img in id_img_dict[id]:
                source_img_file = os.path.join(data_source, img)
                dest_img_file = os.path.join(dest_id_folder, img)
                shutil.copyfile(source_img_file, dest_img_file)


def get_img_list(data_source, data_dest):
    img_list = []
    for root, _, imgs in os.walk(data_source):
        for img in imgs:
            img_list.append(img)

    for img in tqdm(os.listdir(os.path.join(data_dest))):
        if img in img_list:
            # print(os.path.join(data_dest, img))
            os.remove(os.path.join(data_dest, img))

    # print(len(img_list))
    # print(len(set(img_list)))


if __name__ == '__main__':
    # data_root = '/data3/iqiyi/iqiyi_part1'
    # select_img(data_root)

    # data_source = '/data2/fr_train/iqiyi_part2_align'
    # data_dest = '/data3/iqiyi/iqiyi_align'
    # data_merge(data_source, data_dest)

    data_root = r'/data2/fr_train/msra_align_retina' #/data/liyue/data/msra_align_112*96
    # # data_dest = '/data2/futuretown/compare'
    file_list = r'/data2/fr_train/msra_face_112*112.txt'
    get_img_file(data_root, file_list, folder_depth=2)
    # get_img_list(data_source, data_dest)

    # img_folder = r'/data/liyue/data/DataFountain/training_retina_96*112/'
    # query_file = 'query.txt'
    # gallery_file = 'gallery.txt'
    # query_and_gallery(img_folder, query_file, gallery_file,depth=3)

    # img_folder_list = ['/data2/dingding/dingding_align_bak/native_base', '/data2/dingding/dingding_align_bak/native_test']
    # compare_file = '/data2/dingding/positive_and_negative_bak.txt'
    # compare_img_file(img_folder_list, compare_file)