'''
Multi-Task

画像に対してパスや各ラベルなどが付いているcsvがあること前提
'''

'''csvから以下のようなlistを作成
・画像パスが入ったlist  X_trainなど
・マルチラベルのリスト  multi_label_trainなど
'''


class multi_task_dataset(Dataset):
    def __init__(self, my_list, transform=None):
        self.my_list = my_list
        self.transform = transform
        
    def __len__(self):
        return len(self.my_list[0])
    
    def __getitem__(self, index):
        img = Image.open('path_to_img')
        img = img.convert('RGB')
        
        multi_label1 = self.my_list[1][index]
        multi_label2 = self.my_list[2][index]
        ・
        ・
        ・
        
        if self.transform is not None:
            img = self.transform(img)

        list_of_labels = [torch.from_numpy(np.array(multi_label1)), 
                          torch.from_numpy(np.array(multi_label2)),
                          ・
                          ・
                          ・
                          ]
    return img1, list_of_labels[0], list_of_labels[1], ・・・


train_list = [X_train, multi_label1_train, multi_label2_train, ・・・・・・]
test_list = [X_train, multi_label1_test, multi_label2_test, ・・・・・・]

dataset
dataLoader

# model
# model_ft = models.resnet50(pretrained=True)
model_ft = timm.create_model('efficientnet_b0', pretrained=True)

# num_ftrs = model_ft.fc.in_features
# model_ft.fc = nn.Linear(num_ftrs, 512)

num_ftrs = model_ft.classifier.in_features
model_ft.classifier = nn.Linear(num_ftrs, 
                                512 # 要確認
                                )

class multi_output_model(torch.nn.Module):
    def __init__(self, model_core, dd):
        super(multi_output_model, self).__init__()
        
        self.base_model = model_core
        
        self.x1 = nn.Linear(512, 256)
        nn.init.xavier_nomal_(self.x1.weight) 
        
        self.bn1 = nn.BatchNorm1d(256, eps=2e-1)
        self.x2 =  nn.Linear(256, 256)
        nn.init.xavier_normal_(self.x2.weight)
        self.bn2 = nn.BatchNorm1d(256, eps=2e-1)
        
