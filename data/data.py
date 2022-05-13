import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from .dataUtils import *
import random
from PIL import Image
from torchvision import transforms
from torchvision import datasets


class MyDataSet(Dataset):
    def __init__(self,
                 mode='train',
                 valid_category=None,
                 label2id_path='./data/dg_label_id_mapping.json',
                 test_image_path=None,
                 train_image_path='./public_dg_0416/train/',
                 transform_type = None,
                 ):
        '''
        :param mode:  train? valid? test?
        :param valid_category: if train or valid, you must pass this parameter
        :param label2id_path:
        :param test_image_path:
        :param train_image_path:  must end by '/'
        '''
        self.mode = mode
        self.transform_type = transform_type
        self.label2id = get_label2id(label2id_path)
        self.id2label = reverse_dic(self.label2id)
        self.train_image_path = train_image_path
        self.test_image_path = test_image_path

        self.transform = transforms.Compose([
            # add transforms here
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.RandomGrayscale(p=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        self.test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        # if train or valid, synthesize a dic contain num_images * dic, each subdic contain:
        # path、category_id、context_category

        if mode == 'test':
            self.images = get_test_set_images(test_image_path)

        if mode == 'train':
            self.total_dic = get_train_set_dic(train_image_path)
            if valid_category is not None:
                del self.total_dic[valid_category]
            self.synthesize_images()

        if mode == 'valid':
            self.total_dic = get_train_set_dic(train_image_path)
            self.total_dic = dict([(valid_category, self.total_dic[valid_category])])
            self.synthesize_images()

    def synthesize_images(self):
        self.images = {}

        count = 0

        for context_category, context_value in list(self.total_dic.items()):
            for category_name, image_list in list(context_value.items()):
                for img in image_list:
                    now_dic = {}
                    now_dic['path'] = self.train_image_path + context_category + '/' + category_name + '/' + img
                    now_dic['category_id'] = self.label2id[category_name]
                    now_dic['context_category'] = context_category
                    self.images[len(self.images)] = now_dic

                    # count +=1
                    # if count >= 10:
                    #     return 0

        # self.images = dict(list(self.images.items())[0:10])
        # print(self.images)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        if self.mode == 'test':
            img = Image.open(self.test_image_path + self.images[item])
            if self.transform_type == 'test' or None:
                img = self.test_transform(img)
            else:
                img = self.transform(img)
            return img, self.images[item]

        if self.mode == 'train' or self.mode == 'valid':
            img_dic = self.images[item]
            img = Image.open(img_dic['path'])
            img = self.transform(img)
            y = img_dic['category_id']

            # print(img_dic['path'], self.id2label[y])

            return img, y

    def get_id2label(self):
        return self.id2label


def get_loader(train_image_path, valid_image_path, label2id_path, batch_size=32, valid_category='autumn'):
    '''
    if you are familiar with me, you will know this function aims to get train loader and valid loader
    :return:
    '''
    context_category_list = ['autumn', 'dim', 'grass', 'outdoor', 'rock', 'water']
    if valid_category == 'rand':
        valid_category = context_category_list[random.randint(0, len(context_category_list) - 1)]

    print(f'we choose {valid_category} as valid, others as train')
    print('-' * 100)

    train_set = MyDataSet(mode='train', train_image_path=train_image_path,
                          label2id_path=label2id_path, valid_category=valid_category)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    if valid_category is None:
        return train_loader

    valid_set = MyDataSet(mode='valid', valid_category=valid_category,
                          train_image_path=valid_image_path, label2id_path=label2id_path)

    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True)

    print('managed to get loader!!!!!')
    print('-' * 100)

    return train_loader, valid_loader


def get_test_loader(batch_size=32,
                    test_image_path='./public_dg_0416/public_test_flat/',
                    label2id_path='./dg_label_id_mapping.json', transforms = None):
    '''
    No discriptions
    :return:
    '''
    test_set = MyDataSet(mode='test',
                         test_image_path=test_image_path,
                         label2id_path=label2id_path,
                         transform_type=transforms)
    loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)
    return loader, test_set.get_id2label()


if __name__ == '__main__':
    train_loader, valid_loader = get_loader(train_image_path='./public_dg_0416/train/',
                                            valid_image_path='./public_dg_0416/train/',
                                            label2id_path='dg_label_id_mapping.json')

    for x, y in valid_loader:
        print(x)
        print(y)
        assert False
