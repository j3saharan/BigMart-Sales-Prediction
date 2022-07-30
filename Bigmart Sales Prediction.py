#!/usr/bin/env python
# coding: utf-8

# ## Importing Libraries

# In[162]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Importing Bigmart Dataset

# In[163]:


bigmart_df = pd.read_csv('Bigmart_sales_Train.csv')


# In[164]:


bigmart_df.head()   #printing first 5 rows of bigmart data table


# In[165]:


bigmart_df.info()               # getting info of different columns


# In[166]:


bigmart_df.shape          #shape of the dataset


# In[167]:


bigmart_df.columns      # columns of dataset


# ## Predictors - All the column except ' Item_Outlet_Sales '
# ## Target - Item_Outlet_Sales

# In[168]:


bigmart_df.describe().T


# ## A). Checking out each column / Univariate Analysis
# ### 1. Categorical Variables
# - Item_Identifier
# - Item_Fat_Content
# - Item_Type
# - Outlet_Identifier 
# - Outlet_Size 
# - Outlet_Location_Type
# - Outlet_Type
# - Outlet_Establishment_Year

# In[169]:


# 1. Item_identifiers
bigmart_df['Item_Identifier'].unique()  #approx all the items have 


# In[170]:


item_ident = pd.Series(bigmart_df['Item_Identifier'].unique())
item_ident


# In[171]:


print(len(item_ident))    # number of unique items


# ###### There are 1559 products to sell !

# In[172]:


# 2. Item_Fat_Content
bigmart_df['Item_Fat_Content'].value_counts()


# ###### We have to merge 'LF', 'low fat' into 'Low Fat' and 'reg' into 'Regular'

# In[173]:


bigmart_df['Item_Fat_Content'].replace(to_replace='LF', value='Low Fat', inplace=True)     # replacing 'LF' by 'Low Fat'
bigmart_df['Item_Fat_Content'].replace(to_replace='low fat', value='Low Fat', inplace=True)# replacing 'low fat' by 'Low Fat'
bigmart_df['Item_Fat_Content'].replace(to_replace='reg', value='Regular', inplace=True)    # replacing 'reg' by 'Regular'


# In[174]:


bigmart_df['Item_Fat_Content'].value_counts()       


# In[175]:


sns.set()
plt.figure(figsize=(9,5))                       # increasing the size of the plot
sns.countplot(bigmart_df['Item_Fat_Content'])


# In[176]:


#3. Item_Type
bigmart_df['Item_Type'].value_counts()


# In[177]:


plt.figure(figsize=(15,7))             # increasing the size of the plot
plt.xticks(rotation=45)                #rotating the xlabels by 45 degree
sns.countplot(bigmart_df['Item_Type'])


# ###### Most sold items are 'Fruits and Vegetables'

# In[178]:


#4. Outlet_Identifier
bigmart_df['Outlet_Identifier'].value_counts()


# In[179]:


plt.figure(figsize=(12,7))
sns.countplot(bigmart_df['Outlet_Identifier'])


# ###### Most Active 'Outlet_Identifier' is 'OUT027' followed by 'OUT013'

# In[180]:


#5. Outlet_Size
bigmart_df['Outlet_Size'].value_counts()


# In[181]:


plt.figure(figsize=(12,7))
sns.countplot(bigmart_df['Outlet_Size'])


# ###### The number of 'Medium' size outlets are maximum. And number of 'High' size outlets are only 932

# In[182]:


#6. Outlet_Location_Type
bigmart_df['Outlet_Location_Type'].value_counts()


# In[183]:


plt.figure(figsize=(12,7))
sns.countplot(bigmart_df['Outlet_Location_Type'])


# ###### Maximum number of outlets are in 'Tier 3' cities

# In[184]:


#7. Outlet_Type
bigmart_df['Outlet_Type'].value_counts()


# In[185]:


plt.figure(figsize=(12,7))
sns.countplot(bigmart_df['Outlet_Type'])


# ###### The maximum number of 'Oulet_Types' are of 'Supermarket Type1'

# In[186]:


bigmart_df['Outlet_Establishment_Year'].value_counts()


# In[187]:


plt.figure(figsize=(12,7))
sns.countplot(bigmart_df['Outlet_Establishment_Year'])


# ###### Maximum outlets were established in year 1985 

# In[188]:


bigmart_df.info()


# ### 2. Continuous Variables
# - Item_Weight
# - Item_Visibility
# - Item_MRP
#     - Item_Outlet_Sales    ----->>> (target variable)

# In[189]:


#1. Item_Weight
plt.figure(figsize=(12,7))
sns.distplot(bigmart_df['Item_Weight'])


# ###### Maximum item_weight density Distribution can be seen in the range of 5 to 10

# In[190]:


#2. Item_Visibility
plt.figure(figsize=(12,7))
sns.distplot(bigmart_df['Item_Visibility'])


# ###### Maximum density for 'item_Visibility' can be seen in the range of 0.00 to 0.10 ( near 0.05 )

# In[191]:


#3. Item_MRP
plt.figure(figsize=(12,7))
sns.distplot(bigmart_df['Item_MRP'])


# ###### Maximum number of items have MRP near to 100, the in then range of 150-200 and also we have significant items near the MRP of 50

# In[192]:


#4. Item_Outlet_Sales          ----->>>> This is the target variable
plt.figure(figsize=(12,7))
sns.distplot(bigmart_df['Item_Outlet_Sales'] )


# ###### Most of the outlets have 'Item_Outlet_Sales' between 0 and 2000

# ## B). Bivariate Analysis 

# ###### We can find the relation of each Feature with other variable using correlation

# In[193]:


correlation = bigmart_df.corr()           #finding correlation
sns.heatmap(correlation, annot=True)


# In[194]:


sns.pairplot(bigmart_df)


# In[195]:


# categorical values
plt.figure(figsize=(12,7))
sns.barplot(y='Item_Fat_Content',x='Item_Weight', data=bigmart_df)


# ## Null Values Treatment

# In[196]:


bigmart_df.isnull().sum()    #getting the count of null values


# #### We can replace Null values of 'Item_Weight' by Mean

# In[197]:


mean_value = bigmart_df['Item_Weight'].mean()
mean_value


# In[198]:


bigmart_df['Item_Weight'].fillna(value = mean_value, inplace=True)


# In[199]:


bigmart_df.isnull().sum() 


# #### We can replace Null values of 'Outlet_Size' by its Mode

# In[200]:


bigmart_df['Outlet_Size'].mode()


# #### Replacing with mode 

# In[201]:


bigmart_df['Outlet_Size'].fillna(value='Medium', inplace=True)


# ## OR

# In[202]:


# filling the missing values in "Outlet_Size" column with Mode
# mode_of_Outlet_size = bigmart_df.pivot_table(values='Outlet_Size', columns='Outlet_Type', aggfunc=(lambda x: x.mode()[0]))


# In[203]:


# print(mode_of_Outlet_size)


# In[204]:


# miss_values = bigmart_df['Outlet_Size'].isnull()   


# In[205]:


# print(miss_values)


# In[206]:


# bigmart_df.loc[miss_values, 'Outlet_Size'] = bigmart_df.loc[miss_values,'Outlet_Type'].apply(lambda x: mode_of_Outlet_size[x])


# In[207]:


# checking for missing values
bigmart_df.isnull().sum()


# ## Label encoding

# In[208]:


from sklearn.preprocessing import LabelEncoder


# In[209]:


encoder = LabelEncoder()


# In[210]:


bigmart_df['Item_Identifier']= encoder.fit_transform(bigmart_df['Item_Identifier'])

bigmart_df['Item_Fat_Content']= encoder.fit_transform(bigmart_df['Item_Fat_Content'])

bigmart_df['Item_Type']= encoder.fit_transform(bigmart_df['Item_Type'])

bigmart_df['Outlet_Identifier']= encoder.fit_transform(bigmart_df['Outlet_Identifier'])

bigmart_df['Outlet_Size']= encoder.fit_transform(bigmart_df['Outlet_Size'])

bigmart_df['Outlet_Location_Type']= encoder.fit_transform(bigmart_df['Outlet_Location_Type'])

bigmart_df['Outlet_Type']= encoder.fit_transform(bigmart_df['Outlet_Type'])


# In[211]:


bigmart_df.head()


# ## Separating Predictors and Target variable

# In[212]:


X = bigmart_df.drop(columns='Item_Outlet_Sales')
y = bigmart_df['Item_Outlet_Sales']


# ## Train Test Split

# In[213]:


from sklearn.model_selection import train_test_split


# In[214]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)


# In[215]:


print(X_train.shape, X_test.shape)


# ## Imorting Model and Metrices

# # 1. XGBRegressor

# In[216]:


from xgboost import XGBRegressor
from sklearn.metrics import r2_score


# ## Fitting data into model

# In[217]:


reg = XGBRegressor()
reg.fit(X_train, y_train)


# #### making predictions for train and test data

# In[218]:


train_pred = reg.predict(X_train)
test_pred = reg.predict(X_test)


# #### Checking accuracy 

# In[219]:


train_score = r2_score(y_train, train_pred)
test_score = r2_score(y_test, test_pred)


# In[220]:


print(" Train data r2_score is:",train_score)
print(" Test data r2_score is:",test_score)


# # 2. Linear Regression

# In[221]:


from sklearn.linear_model import LinearRegression


# In[222]:


lin_reg = LinearRegression()


# In[223]:


# fitting the data into model
lin_reg.fit(X_train, y_train)


# In[224]:


## making predictions
lin_reg_predict_train = lin_reg.predict(X_train)
lin_reg_predict_test = lin_reg.predict(X_test)


# In[225]:


# accuracy of model
lin_reg_train_score = r2_score(y_train, lin_reg_predict_train)
lin_reg_test_score = r2_score(y_test, lin_reg_predict_test)


# In[226]:


print("The train Score is:",lin_reg_train_score)
print("The test score is:", lin_reg_test_score)


# # 3. Decision Tree

# In[227]:


from sklearn.tree import DecisionTreeRegressor


# In[228]:


DT_reg = DecisionTreeRegressor()


# In[229]:


# fitting the data into model
DT_reg.fit(X_train, y_train)


# In[230]:


# predicting the values from model
DT_train_predict = DT_reg.predict(X_train)
DT_test_predict = DT_reg.predict(X_test)


# In[231]:


# checking the accuracy score of the model
DT_train_score = r2_score(y_train, DT_train_predict)
DT_test_score = r2_score(y_test, DT_test_predict)


# In[232]:


print("The train Score is:",DT_train_score)
print("The test score is:", DT_test_score)


# #### Hence this is overfitting condition (predicted the training data well, but test data prediction is poor)

# # 4. Random Forest

# In[233]:


from sklearn.ensemble import RandomForestRegressor  


# In[234]:


RF_reg = RandomForestRegressor()


# In[235]:


# fitting the data to model
RF_reg.fit(X_train, y_train)


# In[236]:


# predicting the values by model
RF_train_predict = RF_reg.predict(X_train)
RF_test_predict = RF_reg.predict(X_test)


# In[237]:


# checking accuracy of the model
RF_train_score = r2_score(y_train, RF_train_predict)
RF_test_score = r2_score(y_test, RF_test_predict)


# In[238]:


print("The train Score is:",RF_train_score)
print("The test score is:", RF_test_score)

