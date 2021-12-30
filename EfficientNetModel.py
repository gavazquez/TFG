#!/usr/bin/env python
# -*- coding: utf-8 -*- 

import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pickle
from efficientnet_pytorch import EfficientNet
import os

class EfficientNetModel:
    
    def __init__(self, classes, dfClass, dataSet, dataSetTrain):
        self.classes = classes
        self.dataloader = DataLoader(dataSet, batch_size=64, shuffle=False, pin_memory=True)
        self.model = EfficientNet.from_pretrained('efficientnet-b3', num_classes=9)        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.softmax = nn.Softmax()
        self.dataSetTrain = dataSetTrain
        self.losses = []
        self.dfClass = dfClass

    def Train(self, batchSize, lr, eps, dec, mom, version, ensemble=False, pretr=True):    
        print(f'Will use EfficientNet training with device: {self.device}')
        print(f'Batch size: {batchSize}, Learning rate: {lr}, Decay: {dec}, Momentum {mom}, Epochs: {eps}')

        self.modelFileName = f'efficientnet\\efficientnet{version}_lr_{lr}_bs_{batchSize}_ep_{eps}_pretr_{pretr}_erase.mdl'
        self.lossesFileName = f'efficientnet\\efficientnet{version}_losses_lr_{lr}_bs_{batchSize}_ep_{eps}_pretr_{pretr}_erase'

        torch.cuda.empty_cache()
        model = self.model.to(self.device)
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=dec, momentum=mom)
        dataloader = DataLoader(self.dataSetTrain, batch_size=batchSize, shuffle=True, pin_memory=True)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(eps):
            model.train()
            for batch, sample in enumerate(dataloader):
                running_loss = 0.0
                optimizer.zero_grad()
                output = model(sample['image'].to(self.device))
                loss = criterion(output, torch.max(sample['label'], 2)[1].squeeze(-1).to(self.device))
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                #if batch % batchSize == 0:
                    #print(f'{epoch + 1}, {batch + 1}, {running_loss / batchSize}')
                self.losses.append(running_loss)
                del running_loss

        with open(self.lossesFileName, 'wb+') as myfile:
            pickle.dump(self.losses, myfile)
        return model

    def saveModel(self):
        if self.modelFileName is not None:
            torch.save(self.model, self.modelFileName)
            return self.modelFileName
        else:
            print('Train the model first')

    def loadFromFile(self, fileName):
        self.model = torch.load(fileName)

    def validate(self):
        df = pd.DataFrame(columns=self.classes)
        self.model.eval()
        with torch.no_grad():
            print('validating...')
            for idx, sample in enumerate(self.dataloader):
                outputs = self.model(sample['image'].to(self.device))
                outputs = self.softmax(outputs)
                outputs = outputs.cpu().numpy()
                df = df.append(pd.DataFrame(data=outputs, columns=self.classes))

            df['truth'] = df.idxmax(axis=1)
            df['accuracy'] = df['truth'].reset_index(drop=True) == self.dfClass['class'].reset_index(drop=True)
            accuracy = df['accuracy'].values.sum() / len(df['accuracy'])
            return accuracy

    def test(self, dateSetTest, images):
        dataloader_test = DataLoader(dateSetTest, batch_size=16, shuffle=False, pin_memory=True)
        if not os.path.exists("efficientnet\\results"):
            os.makedirs("efficientnet\\results")
        self.model.eval()
        self.model = self.model.to(self.device)
        df = pd.DataFrame(columns=self.classes)
        with torch.no_grad():
            for idx, sample in enumerate(dataloader_test):
                outputs = self.model(sample['image'].to(self.device))
                outputs = self.softmax(outputs)
                outputs = outputs.cpu().numpy()
                df = df.append(pd.DataFrame(data=outputs, columns=self.classes))
            df = df.reset_index()
            del df['index']
            df.insert(0, 'image', images['image'])
        df.to_csv(f'results\\test_results.csv', index=False)