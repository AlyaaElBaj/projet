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



#Set diractory
import os;
path="C:/Users/Alyaa/Desktop/FinalProjetML/Final_ProjectML"
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

######################################################################
# Distribution graphs (histogram/bar graph) of column data
def plotPerColumnDistribution(df, nGraphShown, nGraphPerRow):
    nunique = df.nunique()
    df = df[[col for col in df if nunique[col] > 1 and nunique[col] < 50]] # For displaying purposes, pick columns that have between 1 and 50 unique values
    nRow, nCol = df.shape
    columnNames = list(df)
    nGraphRow = (nCol + nGraphPerRow - 1) / nGraphPerRow
    plt.figure(num = None, figsize = (6 * nGraphPerRow, 8 * nGraphRow), dpi = 80, facecolor = 'w', edgecolor = 'k')
    for i in range(min(nCol, nGraphShown)):
        plt.subplot(nGraphRow, nGraphPerRow, i + 1)
        columnDf = df.iloc[:, i]
        if (not np.issubdtype(type(columnDf.iloc[0]), np.number)):
            valueCounts = columnDf.value_counts()
            valueCounts.plot.bar()
        else:
            columnDf.hist()
        plt.ylabel('counts')
        plt.xticks(rotation = 90)
        plt.title(f'{columnNames[i]} (column {i})')
    plt.tight_layout(pad = 1.0, w_pad = 1.0, h_pad = 1.0)
    plt.show()
    
############################################################################

#plotPerColumnDistribution(df.drop("Direction"), 10, 5)
    #

# For more information about the Data Set:
#Aggregate data to get mean traffic in each direction, by the hour, at all sensor locations
#Agréger les données pour obtenir le trafic moyen dans chaque direction, par heure, à tous les emplacements des capteurs
hourly_vol = df.groupby(['location_name','location_latitude','location_longitude','Direction','Hour']).agg({'Volume':'mean'}).reset_index()
hourly_vol.sample(2)


# Creating a datetime index
df.index = pd.to_datetime(df["Year"] * 100000000 +df["Month"]*1000000+ df["Day"] * 10000 + df["Hour"]*100 +df["Minute"], format="%Y%m%d%H%M")
df = df.drop(columns=["Year", "Month", "Day","Day of Week","Hour","Minute","Time Bin"])
df.head()  #Our Data Set can be seen now as a Time Serie
#May be je me trompe ici parce j'ai l'impression que ça ne va pas avec un ordre chronologique.


# Dealing with missing values
df = df.replace(to_replace='None', value=np.nan).dropna()
# The Data Set contains a lot of rows, so dropping rows where there is None won't affect the size of our Data and we will still have a lot of Data to train the Model.
#okay pas sure parce que là c'est une série tempo on a besoin d'un temps continu non ?
