# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 09:14:22 2020

@author: HP
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#importing the data
df0=pd.read_csv("C:\\Users\\HP\\Documents\\ML\\Projects\\apple products pricing\\phones.csv")
df=df0.copy()
#explroing the dataset

#shape of the data
rows,cols=df.shape

#unique columns of the dataset
df.columns.unique()

#datatypes of the columns 
df.dtypes

#checking for missing data
df.isnull().sum() #no missing data

#diff types of iphones
def uniq_models(df,x):
    return df[x].unique(),len(df[x].unique())

for i in df0.columns.unique():
    print(i)
    l,n= uniq_models(df0,i)
    print("names {}".format(l))
    print("unique values {}".format(n))
#years from release
    #have not missied a year of release from 2007
#looking at all the models
    #28 diff iphones have released
#looking at all the  back camera types
    #3 diff types of back cameras
#front cameras
    # either it has or it does not 
#looking at diff ram types
    #8 diff RAM models have been used
#storage options
    #8 diff storage options 
#screen size
    #9 diff screen sizes used
#processors
    #10 diff processor models used 
#cellular types
    #4 types as expected
#all the prices it is available in
    #available at 21 diff prices 

# releasd in their respective years
def release_yr(x):
    for i in  df.year.unique():
        l=df.loc[df["year"]==i,x].unique()
        for j in range (0,len(l)):
            print( " {} was released in the year {}".format(l[j],i))

release_yr("model")
release_yr("RAM")

#siri release year
df.loc[df["siri"]==1,'year'].values[0] #released in 2011

#heat map of correlations without processor and cellular encoding
sns.heatmap(df0.corr(),annot=True)

def enc_var(df,x):
    temp=pd.get_dummies(df[x],drop_first=True)
    df=df.drop(x,axis=1)
    df=pd.concat([df,temp],axis=1)
    return df
    
df=enc_var(df,"processor")
df=enc_var(df, "cellular")

#correlation with price
#strong relations
temp=df.drop(["model","price in usd"],axis=1).columns.unique()
for i in temp:
    cor=df["price in usd"].corr(df[i])
    if cor>0.3 or cor<-0.2:
        print(" {} is the corr between price and {} ".format(cor,i))

#graphs
#most used
def used_things(df,x):
    df[x].value_counts().plot.bar()
    plt.title(x)
    plt.xlabel("types of {}".format(x))
    plt.ylabel("times used")
    plt.show()

temp=df0.drop(["price in usd"],axis=1).columns.unique()
for i in temp:
    used_things(df0,i)

#OBSERVATIONS Acoordin gto data
#most types of models were released in the year 2020
#iphone se is the most used model
#64 gb model is the most amount of times
#a14 and a13 are most used processors
#4gb RAM variant is mostly used
#manby models still have one camera
#many model have the front camera
#4.7 inch screen is most used screen size
#siri is available in most of the phones
#4g is the most popular cellular type
#face unlock will increase in future

#to see any outrageous price exists
df["price in usd"].plot.box()
plt.ylabel("Price")
plt.title("Price box plot")
plt.show()
print("intrestingly iphones are relatively priced i.e, no outrageous prices")

#price plots
def price_plt_num(df,x):
            l,_=uniq_models(df,x)
            plt.bar(df[x],height=df["price in usd"])
            plt.xlabel(x)
            plt.ylabel("price")
            plt.title("Comparison between price and {}".format(x))
            plt.xlim(min(l)-1,max(l)+5)
            plt.show()

def price_plt_categorical(df,x):
        plt.bar(x=df[x],height=df["price in usd"])
        plt.xlabel(x)
        plt.ylabel("price")
        plt.title("Comparison between price and {}".format(x))
        plt.show()
   
#price plots
#as model increases prices also increase
price_plt_categorical(df,"model")
#as year passes price increase
price_plt_num(df,"year")
#as processor version increases price increase
price_plt_categorical(df0,"processor")
#storage increases price increases
price_plt_num(df,"storage")
#cellular version increases price increses
price_plt_categorical(df0, "cellular")
#screen size increses price increses
price_plt_num(df, "screen")
#camera increses price increases
price_plt_num(df,'f_camera')
price_plt_num(df,'b_camera')

#highly correlated data

#feature engineering and selection
# convert  usd prices to indian price
usd_ind=[41.35,43.51,48.41,45.73,46.67,53.44,56.57,62.33,62.97,66.46,67.79,70.09,70.39,73.60]
years=df.year.unique()
for i in range(0,len(years)):
    for j in df.loc[df["year"]==years[i],"price in usd"].index:
        df["indian_prices"][j]=df.loc[df["year"]==years[i],"price in usd"][j]*usd_ind[i]

#converting screen size from inches to cm
df["screen_size_cm"]=df['screen']*2.54

#converting memories to mb
df["ram_mb"]=df["RAM"]*1024
df["storage_mb"]=df["storage"]*1024

#dropping the columns no required
#dep and indep variables
x=df.drop(["model","storage","RAM","price in usd","indian_prices","screen"],axis=1) #indep variables 
#screen size and camera are highly correlated, but lets leave it there
y=df["indian_prices"] #dep variable

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1,random_state=101)

#scaling
from sklearn.preprocessing import StandardScaler
scl=StandardScaler()
x_train=scl.fit_transform(x_train)
x_test=scl.transform(x_test)


#multiple linear reg
from sklearn.linear_model import LinearRegression
ml_reg=LinearRegression()
ml_reg.fit(x_train,y_train)
y_pred_ml=ml_reg.predict(x_test)
ml_reg.score(x_test,y_test)

#random forest regression
from sklearn.ensemble import RandomForestRegressor
rf_reg=RandomForestRegressor(n_estimators=100,random_state=101)
rf_reg.fit(x_train,y_train)
y_pred_rf_reg=rf_reg.predict(x_test)
rf_reg.score(x_test,y_test)


#predicting with user information
def user_pred():
    data={}
    for k in x.columns.unique():
        data[k]=0
    xa=pd.DataFrame(data,columns=[i for i in x.columns.unique()],index=[0])
    for i in xa.columns:
        inp=int(input("enter value for {} enter 0 if not needed :".format(i)))
        if not inp==0:
            xa[i]=inp
    y_user=ml_reg.predict(scl.transform(xa))
    print("the price can be {} rupees or {} dollars".format(y_user[0],y_user[0]/74))
user_pred()




