#!/usr/bin/env python
# -*- coding: utf-8 -*- 

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import EfficientNetModel
import ISICDataset
import os
import gc

torch.manual_seed(0)

train_path = 'csvfiles\\train.csv'
class_path = 'csvfiles\\validation_class.csv'
eval_path = 'csvfiles\\validation.csv'

test_meta = 'csvfiles\\ISIC_2019_Test_Metadata.csv'

train_img_path = 'train2019\\'
test_img_path = 'test2019\\'

classes = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC', 'UNK']
images = pd.read_csv(test_meta)
df_class = pd.read_csv(class_path)

data_transforms = {
'train': transforms.Compose([transforms.Resize([299, 299]),
                            transforms.ToTensor(),
                            transforms.Normalize((0.6678, 0.5298, 0.5245), (0.1333, 0.1476, 0.1590))]),
'val': transforms.Compose([transforms.Resize([299, 299]),
                            transforms.ToTensor(),
                            transforms.Normalize((0.6678, 0.5298, 0.5245), (0.1333, 0.1476, 0.1590))]),
}

data_set_train = ISICDataset.ISICDataset(train_img_path, transform=data_transforms['train'], csv_path=train_path)
data_set_val = ISICDataset.ISICDataset(img_path=train_img_path, transform=data_transforms['val'], csv_path=eval_path)
data_set_test = ISICDataset.ISICDataset(test_img_path, transform=data_transforms['val'], csv_path=test_meta, test=True)

def DivideData():
    if os.path.exists('csvfiles\\validation_class.csv') and os.path.exists('csvfiles\\validation.csv') and os.path.exists('csvfiles\\train.csv'):
        return

    labels = pd.read_csv('csvfiles\\ISIC_2019_Training_GroundTruth.csv')
    labels = labels.sample(frac=1).reset_index(drop=True)
    train_set = labels.iloc[3334:, :]
    eval_set = labels.iloc[:3334, :]

    df_class = pd.DataFrame(data=eval_set.iloc[:, 0], columns=['image'])
    df_class['class'] = eval_set.iloc[:, 1:].idxmax(axis=1)
    df_class = df_class.reset_index()
    df_class.to_csv('csvfiles\\validation_class.csv', index=False)
    eval_set.to_csv('csvfiles\\validation.csv', index=False)
    train_set.to_csv('csvfiles\\train.csv', index=False)

if __name__ == "__main__":
    DivideData()    
    torch.cuda.empty_cache()
    gc.collect()

    torch.cuda.empty_cache()
    gc.collect()
    ef = EfficientNetModel.EfficientNetModel(classes, df_class, data_set_val, data_set_train)
    ef.Train(8, 0.01, 10, 0, 0, '3')
    accuracy = ef.validate()
    print (f'Got an accuracy of {accuracy}')
    del ef

    #modelFileName = ef.saveModel()
    #print (f'Model file {modelFileName}')
    #ef.loadFromFile("efficientnet\efficientnet3_lr_0.01_bs_16_ep_1_pretr_True_erase.mdl")    

    #ef.test(data_set_test, images)