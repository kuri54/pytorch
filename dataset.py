import os
import shutil

import cv2
import torch.utils.data as data

def split_dataset_into_3(path_to_dataset, train_ratio, valid_ratio):
    '''
    datasetを3つのサブセットに分割する(test,validation,train)
    
    Parameters
    ----------
    data_dir_path: str
        データセットが格納されているディレクトリのパス
    train_ratio: int
        train用に分割する割合
    valid_ratio: int
        validatioｎ用に分割する割合
    
    Returns
    -------
    path_to_datasetの直上にそれぞれのフォルダが作成される
    '''
    _, sub_dirs, _ = next(iter(os.walk(path_to_dataset)))  # retrieve name of subdirectories
    sub_dir_item_cnt = [0 for i in range(len(sub_dirs))]  # list for counting items in each sub directory(class)

    # directories where the splitted dataset will lie
    dir_train = os.path.join(os.path.dirname(path_to_dataset), 'train')
    dir_valid = os.path.join(os.path.dirname(path_to_dataset), 'validation')
    dir_test = os.path.join(os.path.dirname(path_to_dataset), 'test')

    for i, sub_dir in enumerate(sub_dirs):

        dir_train_dst = os.path.join(dir_train, sub_dir)  # directory for destination of train dataset
        dir_valid_dst = os.path.join(dir_valid, sub_dir)  # directory for destination of validation dataset
        dir_test_dst = os.path.join(dir_test, sub_dir)  # directory for destination of test dataset

        # variables to save the sub directory name(class name) and to count the images of each sub directory(class)
        class_name = sub_dir
        sub_dir = os.path.join(path_to_dataset, sub_dir)
        sub_dir_item_cnt[i] = len(os.listdir(sub_dir))

        items = os.listdir(sub_dir)

        # transfer data to trainset
        for item_idx in range(round(sub_dir_item_cnt[i] * train_ratio)):
            if not os.path.exists(dir_train_dst):
                os.makedirs(dir_train_dst)

            source_file = os.path.join(sub_dir, items[item_idx])
            dst_file = os.path.join(dir_train_dst, items[item_idx])
            shutil.copyfile(source_file, dst_file)

        # transfer data to validation
        for item_idx in range(round(sub_dir_item_cnt[i] * train_ratio) + 1,
                              round(sub_dir_item_cnt[i] * (train_ratio + valid_ratio))):
            if not os.path.exists(dir_valid_dst):
                os.makedirs(dir_valid_dst)

            source_file = os.path.join(sub_dir, items[item_idx])
            dst_file = os.path.join(dir_valid_dst, items[item_idx])
            shutil.copyfile(source_file, dst_file)

        # transfer data to testset
        for item_idx in range(round(sub_dir_item_cnt[i] * (train_ratio + valid_ratio)) + 1, sub_dir_item_cnt[i]):
            if not os.path.exists(dir_test_dst):
                os.makedirs(dir_test_dst)

            source_file = os.path.join(sub_dir, items[item_idx])
            dst_file = os.path.join(dir_test_dst, items[item_idx])
            shutil.copyfile(source_file, dst_file)

    return

# Dataset Classの定義
class MyDataset(data.Dataset):
    '''
    すでにサブセットに分割されているデータに対して使用できるDatasetクラス
    ex: train, valid, test
    
    Attrbutes
    ---------
    file_list: list
        画像のファイルパス
    classes_nemes: list
        ラベル名
    transform: object
        前処理
    '''
    def __init__(self, file_list, class_names, transform=None):
        self.file_list = file_list
        self.class_names = class_names
        self.transform = transform

    def __len__(self):
        '''
        画像の枚数を返す
        '''
        return len(self.file_list)
    
    def __getitem__(self, index):
        '''
        （前処理した）画像データとラベルを取得
        '''
        img_path = self.file_list[index]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 画像ラベルをファイル名から抜き出す
        label = self.file_list[index].split('/')[5]
        
        # ラベル名を数値に変換
        label = self.class_names.index(label)
        
        if self.transform is not None:
            image = self.transform(image=image)["image"]
        
        return image, label

def make_filepath_list(data_dir_path, phase):
    phase_data_dir_path = os.path.join(data_dir_path, phase)
    
    data_file_list = []
    
    for top_dir in os.listdir(phase_data_dir_path):
        file_dir = os.path.join(phase_data_dir_path, top_dir)
        file_list = os.listdir(file_dir)
    
        data_file_list += [os.path.join(file_dir, file) for file in file_list]
        
    print('{}: {}'.format(phase, len(data_file_list)))
    
    return data_file_list
