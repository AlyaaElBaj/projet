# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 22:58:09 2020

@author: Alyaa
"""

import numpy as np
import matplotlib.pyplot as plt # plotting
import pandas as pd
from sklearn.impute import SimpleImputer
import numpy as np # linear algebra

from sklearn.preprocessing import StandardScaler

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

#Set diractory
import os;
path="C:/Users/Alyaa/Desktop/3AMines/ML"
os.chdir(path)
os.getcwd()

#test test gljfluglig
# Load and cleanup data from csv file.
df = pd.read_csv("Radar_Traffic_Counts.csv")
# cleanup leading space in names
df['location_name']=df.location_name.apply(lambda x: x.strip()) 
df.sample()
df.info()
nRow, nCol = df.shape
print(f'There are {nRow} rows and {nCol} columns')



# For more information about the Data Set:
#Aggregate data to get mean traffic in each direction, by the hour, at all sensor locations
#Agréger les données pour obtenir le trafic moyen dans chaque direction, par heure, à tous les emplacements des capteurs
hourly_vol = df.groupby(['location_name','location_latitude','location_longitude','Direction','Hour']).agg({'Volume':'mean'}).reset_index()
hourly_vol.sample(2)


# Creating a datetime index
#I dropped the minutes
df.index = pd.to_datetime(df["Year"] * 100000000 +df["Month"]*1000000+ df["Day"] * 10000 + df["Hour"]*100 , format="%Y%m%d%H%M")
df = df.drop(columns=[ "Year","Month", "Day","Day of Week","Hour","Minute","Time Bin"])
df.head()  #Our Data Set can be seen now as a Time Serie



# Dealing with missing values
df = df.replace(to_replace='None', value=np.nan).dropna()
# The Data Set contains a lot of rows, so dropping rows where there is None won't affect the size of our Data and we will still have a lot of Data to train the Model.
# In fact we could have let None, and predict the traffic volume even if as an input we don't really have an idea about the direction.

# I will drop the columns location_latitude and location_longitude because I will only use location name and direction as inputs to predict the output Volume
df = df.drop(columns=["location_latitude", "location_longitude"])

#sorting the data
df=df.sort_values(by='Year',ascending=True)

#plot
plt.rcParams['agg.path.chunksize'] = 10000
plt.plot(df.drop(columns=['location_name','Direction']))
plt.show()



#df.index[2745990]  
#Creating a train set and validation set
train_set = df[:'2019-08-13 02:00:00']
valid_set = df['2019-08-13 02:00:00':]
print('Proportion of train_set : {:.2f}%'.format(len(train_set)/len(df)))
#Proportion of train_set : 0.89%
print('Proportion of valid_set : {:.2f}%'.format(len(valid_set)/len(df)))
#Proportion of valid_set : 0.11%


# déterminons les inputs et outputs
train_x=train_set.drop(columns=["Volume"])
valid_x=valid_set.drop(columns=["Volume"])
train_y=train_set.drop(columns="location_name","Direction")
valid_y=valid_set.drop(columns="location_name","Direction")

class TrafficDataset(Dataset):
    def __init__(self,feature,target):
        self.feature = feature
        self.target = target
    
    def __len__(self):
        return len(self.feature)
    
    def __getitem__(self,idx):
        item = self.feature[idx]
        label = self.target[idx]
        
        return item,label
    
    
class CNN_ForecastNet(nn.Module):
    def __init__(self):
        super(CNN_ForecastNet,self).__init__()
        self.conv1d = nn.Conv1d(3,64,kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(64*2,50)
        self.fc2 = nn.Linear(50,1)
        
    def forward(self,x):
        x = self.conv1d(x)
        x = self.relu(x)
        x = x.view(-1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        
        return x
    
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = CNN_ForecastNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
criterion = nn.MSELoss()


train = TrafficDataset(train_x.reshape(train_x.shape[0],train_x.shape[1],1),train_y)
valid = TrafficDataset(valid_x.reshape(valid_x.shape[0],valid_x.shape[1],1),valid_y)
train_loader = torch.utils.data.DataLoader(train,batch_size=2,shuffle=False)
valid_loader = torch.utils.data.DataLoader(train,batch_size=2,shuffle=False)



train_losses = []
valid_losses = []
def Train():
    
    running_loss = .0
    
    model.train()
    
    for idx, (inputs,labels) in enumerate(train_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        preds = model(inputs.float())
        loss = criterion(preds,labels)
        loss.backward()
        optimizer.step()
        running_loss += loss
        
    train_loss = running_loss/len(train_loader)
    train_losses.append(train_loss.detach().numpy())
    
    print(f'train_loss {train_loss}')
    
def Valid():
    running_loss = .0
    
    model.eval()
    
    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(valid_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            preds = model(inputs.float())
            loss = criterion(preds,labels)
            running_loss += loss
            
        valid_loss = running_loss/len(valid_loader)
        valid_losses.append(valid_loss.detach().numpy())
        print(f'valid_loss {valid_loss}')
        

train_losses = []
valid_losses = []
def Train():
    
    running_loss = .0
    
    model.train()
    
    for idx, (inputs,labels) in enumerate(train_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        preds = model(inputs.float())
        loss = criterion(preds,labels)
        loss.backward()
        optimizer.step()
        running_loss += loss
        
    train_loss = running_loss/len(train_loader)
    train_losses.append(train_loss.detach().numpy())
    
    print(f'train_loss {train_loss}')
    
def Valid():
    running_loss = .0
    
    model.eval()
    
    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(valid_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            preds = model(inputs.float())
            loss = criterion(preds,labels)
            running_loss += loss
            
        valid_loss = running_loss/len(valid_loader)
        valid_losses.append(valid_loss.detach().numpy())
        print(f'valid_loss {valid_loss}')