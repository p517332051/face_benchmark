import os,sys
import shutil
from multiprocessing import Pool
#创建目录，如果目录存在则跳过
def mkdir_img_path(path):
    if os.path.exists(path):
        return
    else:
        mkdir_img_path(os.path.dirname(path))
        print("Create dir "+ path)
        os.mkdir(path)
## 把 list listTemp 拆分成int n个子list
def div(listTemp, n):
    divlist = []
    listTemp_nums = int(len(listTemp))
    fl = 0
    sub_num = int(listTemp_nums/n);
    for i in range(n-1):
        divlist.append(
            listTemp[i*sub_num:(i+1)*sub_num]
        )
        fl+=sub_num
    divlist.append(listTemp[fl:])
    assert listTemp_nums==sum([len(divlist_) for divlist_ in divlist])
    return divlist

#根据SampleTH,把每个ID的数据集分成大于SampleTH，和小于SampleTH的两个部分
def sample_nums_analyis(SampleListPath,SampleTH):
    SampleMatchNumsListLow = []
    SampleMatchNumsListUp  = []
    for SampleListPath_ in  SampleListPath:
        SampleNum = os.listdir(SampleListPath_)
        if len(SampleNum)<SampleTH:
            SampleMatchNumsListLow.append(SampleListPath_)
        else:
            SampleMatchNumsListUp.append(SampleListPath_)
    SampleMatchNums = len(SampleMatchNumsListLow)
    print("SampleNum low nums",SampleTH,": ",SampleMatchNums)
    return SampleMatchNumsListLow,SampleMatchNumsListUp
#将目录下的每个子数据集 比如 msra cele两个子数据集放入sample_nums_analyis去尽心拆分。
def ave_sample_per(data_root_list):
    data_root_list_SampleMatchNumsListLow = []
    data_root_list_SampleMatchNumsListUp = []
    for data_root in data_root_list:
        if not os.path.isdir(data_root):
            continue
        SampleList = os.listdir(data_root)
        SampleListPath = [os.path.join(data_root,Sample) for Sample in SampleList]
        print("dataset ID total ",len(SampleList))
        SampleMatchNumsListLow,SampleMatchNumsListUp = sample_nums_analyis(SampleListPath, 30)
        data_root_list_SampleMatchNumsListLow.append(SampleMatchNumsListLow)
        data_root_list_SampleMatchNumsListUp.append(SampleMatchNumsListUp)
    return data_root_list_SampleMatchNumsListLow,data_root_list_SampleMatchNumsListUp


##把data_root_list_SampleMatchNumsListUp下的每个路径 每个ID的图片从data_root,拷贝到new_data_root下
def CopyImg2NewPath(data_root,new_data_root,data_root_list_SampleMatchNumsListUp):
    for data_root_SampleMatchNumsListUp in data_root_list_SampleMatchNumsListUp:
        for data_root_SampleMatchNumsUp in data_root_SampleMatchNumsListUp:
            new_data_root_SampleMatchNumsUp = data_root_SampleMatchNumsUp.replace(data_root, new_data_root)
            mkdir_img_path(new_data_root_SampleMatchNumsUp)
            for SampleUp in os.listdir(data_root_SampleMatchNumsUp):
                NewSampleUp = os.path.join(new_data_root_SampleMatchNumsUp, SampleUp)
                SampleUp = os.path.join(data_root_SampleMatchNumsUp,SampleUp)
                shutil.copyfile(SampleUp, NewSampleUp)


if __name__ == "__main__":
    data_root = "/data2/fr_train"
    new_data_root = "/data2/M10"
    data_root_list = [os.path.join(data_root,data_root_) for data_root_ in os.listdir(data_root)]
    data_root_list_SampleMatchNumsListLow,data_root_list_SampleMatchNumsListUp = ave_sample_per(data_root_list)
    mkdir_img_path(new_data_root)
    divs = 30
    p = Pool(divs)
    data_root_list_SampleMatchNumsListUp = div(data_root_list_SampleMatchNumsListUp, divs)
    for data_root_list_SampleMatchNumsListUp_ in data_root_list_SampleMatchNumsListUp:
        p.apply_async(CopyImg2NewPath, args=(data_root,new_data_root,data_root_list_SampleMatchNumsListUp_))
    print('Waiting for all subprocesses done...')
    p.close()
    p.join()
    print('All subprocesses done.')


    # CopyImg2NewPath(data_root,new_data_root,data_root_list_SampleMatchNumsListUp)
    # for data_root_SampleMatchNumsListUp in data_root_list_SampleMatchNumsListUp:
    #     for data_root_SampleMatchNumsUp in data_root_SampleMatchNumsListUp:
    #         new_data_root_SampleMatchNumsUp = data_root_SampleMatchNumsUp.replace(data_root, new_data_root)
    #         mkdir_img_path(new_data_root_SampleMatchNumsUp)
    #         for SampleUp in os.listdir(data_root_SampleMatchNumsUp):
    #             NewSampleUp = os.path.join(new_data_root_SampleMatchNumsUp, SampleUp)
    #             SampleUp = os.path.join(data_root_SampleMatchNumsUp,SampleUp)
    #             shutil.copyfile(SampleUp, NewSampleUp)


