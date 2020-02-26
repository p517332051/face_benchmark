import os
import numpy as np
import torch
import torch.utils.data as data
import random
import tqdm
from PIL import Image
class FaceDataset(data.Dataset):
    def __init__(self, data_dir, ann_file, transforms=None, augmenter=None,im_info=[112,96]):
        assert transforms is not None

        self.root = data_dir
        self.file_list = ann_file
        self.augmenter = augmenter
        self.transform = transforms
        self.im_info = im_info
        image_list = []
        label_list = []
        with open(ann_file) as f:
            img_label_list = f.read().splitlines()

        self.image_label_list = []
        for info in img_label_list:
            image_path, label_name = info.split(' ')
            self.image_label_list.append([image_path, int(label_name)])
            image_list.append(image_path)
            label_list.append(int(label_name))

        self.image_list = image_list
        self.label_list = label_list
        self.class_nums = len(set(self.label_list))
        # self.class_nums = max(self.label_list)
        print("dataset size: ", len(self.image_list), '/', self.class_nums)

    def __getitem__(self, index):
        img_path = self.image_list[index]
        label = self.label_list[index]
        p = random.random()
        img = Image.open(os.path.join(self.root, img_path)).convert('RGB')
        if self.augmenter is not None and p<=0.2:
            img_array = np.asarray(img)
            img_array = self.augmenter.augment_image(img_array)
            img = Image.fromarray(img_array.astype('uint8')).convert('RGB')
        img = self.transform(img)

        return img, label, index

    def __len__(self):
        return len(self.image_list)

    def get_img_info(self, index):
        return {"height": self.im_info[0], "width": self.im_info[1]}
# def FaceDataset():
#     return FR_train_data



class TripletFaceDataset(data.Dataset):

    def __init__(self, data_dir, ann_file, n_triplets, transforms=None, augmenter=None,im_info=[112,96]):

        assert transforms is not None
        self.root = data_dir
        self.file_list = ann_file
        self.augmenter = augmenter
        self.transform = transforms
        self.im_info = im_info
        image_list = []
        label_list = []
        with open(self.file_list) as f:
            img_label_list = f.read().splitlines()
        self.image_label_list = []
        for info in img_label_list:
            image_path, label_name = info.split(' ')
            self.image_label_list.append([image_path, int(label_name)])
            image_list.append(image_path)
            label_list.append(int(label_name))

        self.image_list = image_list
        self.label_list = label_list
        self.class_nums = len(set(self.label_list))
        # self.class_nums = max(self.label_list)
        print("dataset size: ", len(self.image_list), '/', self.class_nums)

        self.n_triplets = n_triplets

        print('Generating {} triplets'.format(self.n_triplets))
        self.training_triplets = self.generate_triplets(self.image_list, self.label_list, self.n_triplets,self.class_nums)

    @staticmethod
    def generate_triplets(imgs, labels, num_triplets, n_classes):
        def create_indices(imgs, labels):
            inds = dict()
            for idx, img_path in enumerate(imgs):
                label = labels[idx]
                if label not in inds:
                    inds[label] = []
                inds[label].append(img_path)
            return inds

        triplets = []
        # Indices = array of labels and each label is an array of indices
        indices = create_indices(imgs, labels)

        for x in range(num_triplets):
            c1 = np.random.randint(0, n_classes-1)
            c2 = np.random.randint(0, n_classes-1)
            while len(indices[c1]) < 2:
                c1 = np.random.randint(0, n_classes-1)

            while c1 == c2:
                c2 = np.random.randint(0, n_classes-1)
            if len(indices[c1]) == 2:  # hack to speed up process
                n1, n2 = 0, 1
            else:
                n1 = np.random.randint(0, len(indices[c1]) - 1)
                n2 = np.random.randint(0, len(indices[c1]) - 1)
                while n1 == n2:
                    n2 = np.random.randint(0, len(indices[c1]) - 1)
            if len(indices[c2]) ==1:
                n3 = 0
            else:
                n3 = np.random.randint(0, len(indices[c2]) - 1)

            triplets.append([indices[c1][n1], indices[c1][n2], indices[c2][n3],c1,c2])
        return triplets
    def loader(self,img_path):
        p = random.random()
        img = Image.open(os.path.join(self.root, img_path)).convert('RGB')
        if self.augmenter is not None and p<=0.2:
            img_array = np.asarray(img)
            img_array = self.augmenter.augment_image(img_array)
            img = Image.fromarray(img_array.astype('uint8')).convert('RGB')
        return img
    def __getitem__(self, index):
        '''
        Args:
            index: Index of the triplet or the matches - not of a single image
        Returns:
        '''
        def transform(img_path):
            """Convert image into numpy array and apply transformation
               Doing this so that it is consistent with all other datasets
               to return a PIL Image.
            """

            img = self.loader(img_path)
            return self.transform(img)

        # Get the index of each image in the triplet
        a, p, n,c1,c2 = self.training_triplets[index]

        # transform images if required
        img_a, img_p, img_n = transform(a), transform(p), transform(n)
        return img_a, img_p, img_n,c1,c2

    def __len__(self):
        return len(self.training_triplets)
    def get_img_info(self, index):
        return {"height": self.im_info[0], "width": self.im_info[1]}