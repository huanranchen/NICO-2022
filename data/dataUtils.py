import json
import os


def reverse_dic(dic):
    return dict([(v, k) for (k, v) in dic.items()])


def get_label2id(file_name):
    # implement a function, which return a dic, mapping from label(str) to id(integer)
    # for example:
    # dic['cat']=1
    with open(file_name) as file:
        dict = json.load(file)

    return dict


def get_train_set_dic(path):
    '''
    :param path:  example './data/' last character must be '/'
    :return:
    '''
    # implement a function, which return a dic, whose keys are context of image and keys are dics mapping from image catagory to a list of image name.
    # for example:
    # dic['grass'] = dic1
    # dic1['airplane'] = [001321.jpg, 23129.jpg, 1ew0213.jpg]
    dict = {}
    list_file = os.listdir(path)
    for item in list_file:
        list_content = os.listdir(path + item)
        inside_dict = {}
        for key in list_content:
            inside_dict[key] = os.listdir(path + item + '/' + key)
        dict[item] = inside_dict

    return dict


def get_test_set_images(path):
    # implement a function, return a list of image names

    dir = os.listdir(path)
    print(dir)


if __name__ == '__main__':
    # get_test_set_images(path = './public_dg_0416/public_test_flat')
    total_dic = get_train_set_dic('./public_dg_0416/train/')
    dic_grass = total_dic['autumn']
    print(dic_grass['airplane'])
