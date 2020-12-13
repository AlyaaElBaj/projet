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
df[["Year"]]= pd.to_datetime(df["Year"] * 100000000 +df["Month"]*1000000+ df["Day"] * 10000 + df["Hour"]*100 , format="%Y%m%d%H%M")
df = df.drop(columns=[ "Month", "Day","Day of Week","Hour","Minute","Time Bin"])
df.head()  #Our Data Set can be seen now as a Time Serie

# I will drop the columns location_latitude and location_longitude because I will only use location name and direction as inputs to predict the output Volume
df = df.drop(columns=["location_latitude", "location_longitude"])

#sorting the data
df=df.sort_values(by='Year',ascending=True)

# Dealing with missing values
df = df.replace(to_replace='None', value=np.nan).dropna()
# The Data Set contains a lot of rows, so dropping rows where there is None won't affect the size of our Data and we will still have a lot of Data to train the Model.
# In fact we could have let None, and predict the traffic volume even if as an input we don't really have an idea about the direction.

#count number of couples (location, direction)
#list of location_names
names=df['location_name'].unique().tolist()
#list of directions 
directions=['NB','SB','EB','WB']

#we sum the volume over minutes for each 'Year'
group= df.groupby(by = ['location_name','Year','Direction'], as_index=False)['Volume'].sum()
group.head()

#rearrange data
d = {'location_name': group['location_name'], 'Direction': group['Direction'],'Year': group['Year'],'Volume':group['Volume']}
data2=pd.DataFrame(data=d)

date = data2.groupby(['location_name','Direction']).size()
count_u = data2.groupby(['location_name','Direction']).size().reset_index().rename(columns={0:'count'})
#32 rows=32 couples (location, direction)
count_u.info()
#let's check if all the couples have sufficient data
min(count_u['count']) #11491
max(count_u['count']) # 17206
count_u['count'].mean() #14977.125
count_u=count_u.sort_values(['count'])
#we delete couples who have less than 100 counts
new_data = count_u[count_u['count'] > 4000] #we kept our 32 couples
min(new_data['count']) #11491
#for each location and direction we will have a time series
#where the volume is a function of "Date-Hour"





def ts_for_couple(location,direction):
    #location and direction are strings
    extract=df.loc[df.location_name==location][df.Direction==direction]
    #dates=extract['Date']
    volume=extract['Volume'].to_numpy()
    return volume


#On va stocker notre data dans un dictionnaire qui prend comme clé le couple (location,direction) et lui attribue sa série 
#temporelle correspondante.
dict_df={}
couples=new_data[['location_name','Direction']]
couples=[tuple(couples.iloc[i]) for i in range(couples.shape[0])]
volume=[]
for couple in couples:
    location,direction=couple
    volume.append(ts_for_couple(location,direction))
for i in range(len(volume)):
    key=couples[i] #(location,direction)
    dict_df[key]=volume[i] 


#on va scallé les volume (aka notre y) plus tard 
    
    
    
    
    
    
    
 """
#df.index[2745990]  
#Creating a train set and validation set
train_set = df[:'2019-08-13 02:00:00']   #oublie pas de changer l'indice au lieu de year
test_set = df['2019-08-13 02:00:00':]
print('Proportion of train_set : {:.2f}%'.format(len(train_set)/len(df)))
#Proportion of train_set : 0.89%
print('Proportion of test_set : {:.2f}%'.format(len(test_set)/len(df)))
#Proportion of test_set : 0.11%




#Let's focus on train set
train_set.describe()
print(train_set.shape)
train_set.head()


#Let's focus on validation set
test_set.describe()
print(train_set.shape)
train_set.head()

#Let's focus on train set
train_set.describe()
print(train_set.shape)
train_set.head()


#Let's focus on validation set
test_set.describe()
print(train_set.shape)
train_set.head()


print(train_set.index.min(), train_set.index.max())
print(test_set.index.min(), test_set.index.max())

def sin_transform(values):
    return np.sin(2*np.pi*values/len(set(values)))

def cos_transform(values):
    return np.cos(2*np.pi*values/len(set(values)))


df['dayofweek_sin'] = sin_transform(df['Day of Week'])
df['dayofweek_cos'] = cos_transform(df['Day of Week'])
df['month_sin'] = sin_transform(df['Month'])
df['month_cos'] = cos_transform(df['Month'])
df['day_sin'] = sin_transform(df['Day'])
df['day_cos'] = cos_transform(df['Day'])



plt.plot(sin_transform(np.arange(0,12)), label='month_sin')
plt.plot(cos_transform(np.arange(0,12)), label='month_cos')
plt.legend()
"""


#### Le réseau de neuronnes #####
    
    
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
        self.fc3 = nn.Linear(in_features=130, out_features=24*7)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.drop(out)
        #out = self.fc2(out)
        out = self.fc3(out)
        return out
#seq=Get_Time_Series(names[0], directions[0])
#building the sliding window
#n_steps=24*30*2 #2 months
#horizon=24*7 #1 week
#min count in data is 11491 we will have a lot more than 7(mincount//n_steps) samples-we only move the window by 24*7
def split_ts(seq,horizon=24*7,n_steps=24*30*2):
    """ this function take in arguments a traffic Time Series for the couple (l,d)
    and applies a sliding window of length n_steps to generates samples having this 
    length and their labels (to be predicted) whose size is horizon
    """
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

def train_test_set(xlist,ylist):
    """ this functions splits the samples and labels datasets xlist and ylist
    (given by the function split_ts) into a training set and a test set
    """
    """
    data_size=len(xlist)
    test_size=int(data_size*0.2) #20% of the dataset
    #training set
    X_train  = xlist[:data_size-test_size]
    Y_train = ylist[:data_size-test_size]
    #test set
    X_test = xlist[data_size-test_size:]
    Y_test = ylist[data_size-test_size:]
    """
    X_train, X_test, Y_train, Y_test =train_test_split(xlist,ylist,test_size=0.2,random_state=1)
    return(X_train,Y_train,X_test,Y_test)

def model_traffic(mod,seq,num_ep=60,horizon=24*7,n_steps=24*30*2):
    #inputs are the model mod, the Time Series sequence and the number of epochs
    #building the model
    xlist,ylist = split_ts(seq,horizon,n_steps)
    X_train,Y_train,X_test,Y_test=train_test_set(xlist,ylist)
    idxtr = list(range(len(X_train)))
    #loss and optimizer
    loss = torch.nn.MSELoss()
    opt = torch.optim.Adam(mod.parameters(),lr=0.0005)
    for ep in range(num_ep):
        shuffle(idxtr)
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


###################################################################
# TRAINING AND EVALUATION OF THE MODEL FOR EACH (LOCATION,DIRECTION)
###################################################################
results = pd.DataFrame( columns = ["couple", "training_loss", "test_loss"])
num_ep=10000
horizon=24*7
n_steps=24*28*3
for l,d in data_dict.keys():
    seq=data_dict[(l,d)] #volume sequence for (l,d) location, direction
    xlist,ylist = split_ts(seq,horizon,n_steps)
    print("couple:",(l,d))
    print("number of samples in the dataset:", len(xlist))
    mod = TimeCNN()
    train_loss, test_loss =model_traffic(mod,seq,num_ep,horizon,n_steps)
    print("train_loss, test_loss =", train_loss, test_loss, "\n")
    results.loc[len(results)] = [couple, train_loss, test_loss]
    del(mod)

 

#on va scallé les volume (aka notre y) plus tard """



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