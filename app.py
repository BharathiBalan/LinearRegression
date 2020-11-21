import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from statsmodels.stats.outliers_influence import variance_inflation_factor

#LOADING THE DATA
boston_data=load_boston()
boston=pd.DataFrame(boston_data.data,columns=boston_data.feature_names)
target=pd.DataFrame(boston_data.target)
# print("Started")
# # print(boston.head())
# print("Stopped")
# # print(target)
# print("Target over")
# print("Describe: ",boston.describe())
# print("Info    : ",boston.info())
# print("Shape   : ",boston.shape)
#
# print(boston_data.keys())
# print(boston_data.DESCR)
#
#
#Scaling the data
scalar=sklearn.preprocessing.StandardScaler()
scaled_boston=scalar.fit_transform(boston)
#print(type(scaled_boston))

#Checking for multi-coliearity
vif_1=pd.DataFrame()
vif_1['VIF']=[variance_inflation_factor(scaled_boston,i) for i in range(scaled_boston.shape[1])]
vif_1['features']=boston_data.feature_names
#print("VIF : \n",(vif_1))

#Model1
x_train_1, x_test_1, y_train_1, y_test_1= train_test_split(scaled_boston,target,test_size=0.2,random_state=22)

lin_reg_1=LinearRegression()
lin_reg_1.fit(x_train_1,y_train_1)

score_1=r2_score(lin_reg_1.predict(x_test_1),y_test_1)
print("score for 1st model is ", score_1)


#Model2
df2=pd.DataFrame(boston_data.data,columns=boston_data.feature_names)
df2.drop(columns='RAD',inplace=True)


scaled_df2 = scalar.fit_transform(df2)

vif_2=pd.DataFrame()
vif_2['VIF']=[variance_inflation_factor(scaled_df2,i) for i in range(scaled_df2.shape[1])]
#print("VIF : \n",(vif_2))

x_train_2, x_test_2, y_train_2, y_test_2= train_test_split(scaled_df2,target,test_size=0.2,random_state=22)

lin_reg_2=LinearRegression()
lin_reg_2.fit(x_train_2,y_train_2)

score_2=r2_score(lin_reg_2.predict(x_test_2),y_test_2)
print("score for 2nd model is ", score_2)