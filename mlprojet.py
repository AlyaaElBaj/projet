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
import math

from sklearn.preprocessing import StandardScaler

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import random 



#Set diractory
import os;
path="C:/Users/Alyaa/Desktop/3AMines/ML"
os.chdir(path)
os.getcwd()


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
hourly_vol = df.groupby(['location_name','location_latitude','location_longitude','Direction','Hour']).agg({'Volume':'mean'}).reset_index()
hourly_vol.sample(2)


# Creating a datetime column
#I dropped the minutes
df[["Year"]]= pd.to_datetime(df["Year"] * 100000000 +df["Month"]*1000000+ df["Day"] * 10000 + df["Hour"]*100 , format="%Y%m%d%H%M")
df = df.drop(columns=[ "Month", "Day","Day of Week","Hour","Minute","Time Bin"])
df.head()  #Our Data Set can be seen now as a Time Serie

# I will drop the columns location_latitude and location_longitude because I will only use location name and direction as inputs to predict the output Volume
df = df.drop(columns=["location_latitude", "location_longitude"])

#sorting the data by 'Year'
df=df.sort_values(by='Year',ascending=True)

# Dealing with missing values
df = df.replace(to_replace='None', value=np.nan).dropna()
# The Data Set contains a lot of rows, so dropping rows where there is None won't affect the size of our Data and we will still have a lot of Data to train the Model.
# In fact we could have let None, and predict the traffic volume even if as an input we don't really have an idea about the direction.

#we want to count number of couples (location, direction)
#list of location_names
names=df['location_name'].unique().tolist()
#list of directions 
directions=['NB','SB','EB','WB']

#we sum the volume over minutes for each 'Year'
group= df.groupby(by = ['location_name','Year','Direction'], as_index=False)['Volume'].sum()
group.head()

#Rearrange data
d = {'location_name': group['location_name'], 'Direction': group['Direction'],'Year': group['Year'],'Volume':group['Volume']}
data2=pd.DataFrame(data=d)

date = data2.groupby(['location_name','Direction']).size()
data_couples = data2.groupby(['location_name','Direction']).size().reset_index().rename(columns={0:'sum'})
#32 rows=32 couples (location, direction)
data_couples.info()
#let's check if all the couples have sufficient data
min(data_couples['sum']) #11491
max(data_couples['sum']) # 17206
data_couples['sum'].mean() #14977.125
#We're all good
#for each location and direction we will have a time series



#This function takes a couple (location,direction) and returns its time series
def ts_for_couple(location,direction):
    #location and direction are strings
    extract=df.loc[df.location_name==location][df.Direction==direction]
    #dates=extract['Date']
    volume=extract['Volume'].to_numpy()
    return volume


#On va stocker notre data dans un dictionnaire qui prend comme clé le couple (location,direction) et lui attribue sa série 
#temporelle correspondante.
dict_df={}
couples=data_couples[['location_name','Direction']]
couples=[tuple(couples.iloc[i]) for i in range(couples.shape[0])]
volume=[]
for couple in couples:
    location,direction=couple
    volume.append(ts_for_couple(location,direction))
for i in range(len(volume)):
    key=couples[i] #(location,direction)
    dict_df[key]=volume[i] 


#on va scallé les volume (aka notre y) plus tard 
    
########################################################################
############################## CNN Model ###############################
########################################################################


    
    
class TimeCNN(nn.Module):
    def __init__(self):
        super(TimeCNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1), #64 entrée #80 sortie
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)  
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(8) #2
        )
        self.fc1 = nn.Linear(in_features=64*8, out_features=130)   #130 par 80
        self.drop = nn.Dropout2d(0.3)
        #self.fc2 = nn.Linear(in_features=128, out_features=32)
        self.fc2 = nn.Linear(in_features=130, out_features=24*7)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.drop(out)
        #out = self.fc2(out)
        out = self.fc2(out)
        return out
    
    
#seq=ts_for_couple(names[0], directions[0])
#building the sliding window
#n_steps=24*30*2 => 2 months
#horizon=24*7 => 1 week
#min sum in data is 11491 we will have a lot more than 7(mincount//n_steps) samples
#we only move the window by 24*7 (1week)
#This function helps do so: it takes in arguments a traffic volume times series for a couple (location,direction)
#and applies a slicing window (length=n_steps) and creates samples with length=n_steps and their labels (size=horizon)

def ts_sequence_building(seq,horizon=24*7,n_steps=24*30*2):
    #for the Min-Max normalization X-min(seq)/max(seq)-min(seq)
    max_seq=max(seq)
    min_seq=min(seq)
    seq_norm=max_seq-min_seq
    xlist, ylist = [], []
    for i in range(len(seq)//horizon):
        end= i*horizon + n_steps
        if end+horizon > len(seq)-1:
            break
        xx = (seq[i*horizon:end]-min_seq)/seq_norm
        xlist.append(torch.tensor(xx,dtype=torch.float32))
        yy = (seq[end:(end+horizon)]-min_seq)/seq_norm
        ylist.append(torch.tensor(yy,dtype=torch.float32))
    print("number of samples %d and sample size %d (%d months)" %(len(xlist),len(xlist[0]),n_steps/(24*30)))
    return(xlist,ylist)


#This function splits samples ans labels datasets xlist and ylist into a training and test set
def train_test_set(xlist,ylist):
    X_train, X_test, Y_train, Y_test =train_test_split(xlist,ylist,test_size=0.2,random_state=1)  # test set is #20% of the dataset
    return(X_train,Y_train,X_test,Y_test)

def model_traffic(mod,seq,num_ep=60,horizon=24*7,n_steps=24*30*2):
    #inputs are the model mod, the Time Series sequence and the number of epochs
    #building the model
    xlist,ylist = ts_sequence_building(seq,horizon,n_steps)
    X_train,Y_train,X_test,Y_test=train_test_set(xlist,ylist)
    idxtr = list(range(len(X_train)))
    #loss and optimizer
    loss = torch.nn.MSELoss()
    opt = torch.optim.Adam(mod.parameters(),lr=0.0005)
    for ep in range(num_ep):
        random.shuffle(idxtr)
        ep_loss=0.
        mod.train()
        for j in idxtr:
            opt.zero_grad()
            #forward pass
            haty = mod(X_train[j].view(1,1,-1))
            # print("pred %f" % (haty.item()*vnorm))
            lo = loss(haty,Y_train[j].view(1,-1))
            #backward pass
            lo.backward()
            #optimization
            opt.step()
            ep_loss += lo.item()
        #model evaluation
        mod.eval()
        test_loss=0
        for i in range(len(X_test)):    
            haty = mod(X_test[i].view(1,1,-1))
            test_loss+= loss(haty,Y_test[i].view(1,-1))
        if ep%50==0:
            print("epoch %d training loss %1.9f test loss %1.9f" % (ep, ep_loss, test_loss.item()))
    #selected model (last epoch)
    test_loss=0
    for i in range(len(X_test)):    
        haty = mod(X_test[i].view(1,1,-1))
        test_loss+= loss(haty,Y_test[i].view(1,-1))
    return ep_loss,test_loss

#Now let's train and evaluate the model for each (location,direction)

results = pd.DataFrame( columns = ["couple", "training_loss", "test_loss"])
num_ep=10000
horizon=24*7
n_steps=24*28*3
for l,d in dict_df.keys():
    seq=dict_df[(l,d)] #volume sequence for (l,d) location, direction
    xlist,ylist = ts_sequence_building(seq,horizon,n_steps)
    print("couple:",(l,d))
    print("number of samples in the dataset:", len(xlist))
    mod = TimeCNN()
    train_loss, test_loss =model_traffic(mod,seq,num_ep,horizon,n_steps)
    print("train_loss, test_loss =", train_loss, test_loss, "\n")
    results.loc[len(results)] = [couple, train_loss, test_loss]
    del(mod)

"""
#############################################################################################################
############################################################################################################# """

#################### Une Nouvelle approche #######################
                    
def split_sequence(sequence, n_steps):
    x, y = list(), list()
    for i in range(len(sequence)):
        
        end_ix = i + n_steps
        
        if end_ix > len(sequence)-1:
            break
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        x.append(seq_x)
        y.append(seq_y)
    return array(x), array(y)

raw_seq = [10,20,30,40,50,60,70,80,90]
n_steps = 3
train_x,train_y = split_sequence(train_set.Elec_kW.values,n_steps)
valid_x,valid_y = split_sequence(valid_set.Elec_kW.values,n_steps)



#scaler_y = MinMaxScaler()
#print(scaler_y.fit(y))
#yscale=scaler_y.transform(y)
#on ne peut pas scaller x car c'est des variables qualitatives mais j'ai scaller y (pas sure si j ai le droit de scaller que y)



######################## autre façon pour le val set et train set ######################################

#df.index[2745990]  
#Creating a train set and validation set
#train_set = df[:'2019-08-13 02:00:00']
#valid_set = df['2019-08-13 02:00:00':]
#print('Proportion of train_set : {:.2f}%'.format(len(train_set)/len(df)))
##Proportion of train_set : 0.89%
#print('Proportion of valid_set : {:.2f}%'.format(len(valid_set)/len(df)))
##Proportion of valid_set : 0.11%
###
#
## déterminons les inputs et outputs
#train_x=train_set.drop(columns=["Volume"])
#valid_x=valid_set.drop(columns=["Volume"])
#train_y=train_set.drop(columns="location_name","Direction")
#valid_y=valid_set.drop(columns="location_name","Direction")
#faut transformer le x et le y en vecteur
#je crois faut mettre le x et y avant les set train et validation train

#######################################################################################################""



#class TrafficDataset(Dataset):
#    def __init__(self,feature,target):
#        self.feature = feature
#        self.target = target
#    
#    def __len__(self):
#        return len(self.feature)
#    
#    def __getitem__(self,idx):
#        item = self.feature[idx]
#        label = self.target[idx]
#        
#        return item,label
#    
#    
#class CNN_ForecastNet(nn.Module):
#    def __init__(self):
#        super(CNN_ForecastNet,self).__init__()
#        self.conv1d = nn.Conv1d(3,64,kernel_size=1)
#        self.relu = nn.ReLU(inplace=True)
#        self.fc1 = nn.Linear(64*2,50)
#        self.fc2 = nn.Linear(50,1)
#        
#    def forward(self,x):
#        x = self.conv1d(x)
#        x = self.relu(x)
#        x = x.view(-1)
#        x = self.fc1(x)
#        x = self.relu(x)
#        x = self.fc2(x)
#        
#        return x
#    
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#model = CNN_ForecastNet().to(device)
#optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
#criterion = nn.MSELoss()
#
#
#train = TrafficDataset(train_x.reshape(train_x.shape[0],train_x.shape[1],1),train_y)
#valid = TrafficDataset(valid_x.reshape(valid_x.shape[0],valid_x.shape[1],1),valid_y)
#train_loader = torch.utils.data.DataLoader(train,batch_size=2,shuffle=False)
#valid_loader = torch.utils.data.DataLoader(train,batch_size=2,shuffle=False)
#
#
#
#train_losses = []
#valid_losses = []
#def Train():
#    
#    running_loss = .0
#    
#    model.train()
#    
#    for idx, (inputs,labels) in enumerate(train_loader):
#        inputs = inputs.to(device)
#        labels = labels.to(device)
#        optimizer.zero_grad()
#        preds = model(inputs.float())
#        loss = criterion(preds,labels)
#        loss.backward()
#        optimizer.step()
#        running_loss += loss
#        
#    train_loss = running_loss/len(train_loader)
#    train_losses.append(train_loss.detach().numpy())
#    
#    print(f'train_loss {train_loss}')
#    
#def Valid():
#    running_loss = .0
#    
#    model.eval()
#    
#    with torch.no_grad():
#        for idx, (inputs, labels) in enumerate(valid_loader):
#            inputs = inputs.to(device)
#            labels = labels.to(device)
#            optimizer.zero_grad()
#            preds = model(inputs.float())
#            loss = criterion(preds,labels)
#            running_loss += loss
#            
#        valid_loss = running_loss/len(valid_loader)
#        valid_losses.append(valid_loss.detach().numpy())
#        print(f'valid_loss {valid_loss}')
#        
#
#train_losses = []
#valid_losses = []
#def Train():
#    
#    running_loss = .0
#    
#    model.train()
#    
#    for idx, (inputs,labels) in enumerate(train_loader):
#        inputs = inputs.to(device)
#        labels = labels.to(device)
#        optimizer.zero_grad()
#        preds = model(inputs.float())
#        loss = criterion(preds,labels)
#        loss.backward()
#        optimizer.step()
#        running_loss += loss
#        
#    train_loss = running_loss/len(train_loader)
#    train_losses.append(train_loss.detach().numpy())
#    
#    print(f'train_loss {train_loss}')
#    
#def Valid():
#    running_loss = .0
#    
#    model.eval()
#    
#    with torch.no_grad():
#        for idx, (inputs, labels) in enumerate(valid_loader):
#            inputs = inputs.to(device)
#            labels = labels.to(device)
#            optimizer.zero_grad()
#            preds = model(inputs.float())
#            loss = criterion(preds,labels)
#            running_loss += loss
#            
#        valid_loss = running_loss/len(valid_loader)
#        valid_losses.append(valid_loss.detach().numpy())
#        print(f'valid_loss {valid_loss}')
#        
#epochs = 200
#for epoch in range(epochs):
#    print('epochs {}/{}'.format(epoch+1,epochs))
#    Train()
#    Valid()
#    gc.collect()