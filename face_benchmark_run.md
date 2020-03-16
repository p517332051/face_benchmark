# Face_benchmark





数据训练准备

举个例子，比如msra这个文件夹内有数据，存储方式是

your_path/data/msra/xiaoming/*.jpg

your_path/data/msra/xiaogang/*.jpg

……..

your_path/data/msra/laowang/*.jpg

然后使用your_path/face_benchmark/tools/data_process.py  将下面变量设置为

```
data_root=r'your_path/data/msra'
file_list = r'your_path/data/msra/msra_face_112*112.txt'
```

最后在your_path/data/msra这个目录下生成一个msra_face_112*112.txt的txt文件，用来记录msra每个ID的label。

然后在your_path/maskrcnn_benchmark/config/paths_catalog.py这个代码添加路径。数据路径添加和maskrcnn_benchmark一样。

```python
"msra_face": {
    "img_dir": r"you_path/data/msra",#msra数据集
    "ann_file": r"you_path/data/msra/msra_face_112*112.txt",#存储msra每个ID的label
    "im_info": [112, 112]  #你对齐后的图片大小
},
```

训练启动脚本

```shell
python tools/Muti_GPUS_Train.py --ngpus_per_node=8 --npgpu_per_proc=1 tools/train_face_netDivFC.py --skip-test --config-file configs/face_reg/face_net_msra_celeb.yaml DATALOADER.NUM_WORKERS 16 OUTPUT_DIR
```

--ngpus_per_node=8 表示的机器有8个GPU。

--npgpu_per_proc=1 表示你使用1个DDP，使用8个GPU拆分FC层进行训练。 

例子：

—ngpus_per_node=4 表示的机器有8个GPU。

—npgpu_per_proc=2 表示你使用2个DDP，8个GPU分成两份，分别给DDP在4个GPU下拆分FC。

目前训练速度使用的backbone是insightface的 IR_101

| 训练框架            | 训练数据集 | 训练速度     | backbone | 硬件      | 最大batch数 |
| ------------------- | ---------- | ------------ | -------- | --------- | ----------- |
| face_benchmark      | msra+cele  | 2天10小时    | IR_101   | 8卡1080ti | 512         |
| face.evolve.Pytorch | msra+cele  | 一周以上     | IR_101   | 8卡1080ti | 470         |
| Insight face        | msra+cele  | 你们去测试下 | IR_101   | 8卡1080ti | 480         |

