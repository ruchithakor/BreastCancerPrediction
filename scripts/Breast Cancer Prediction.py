#!/usr/bin/env python
# coding: utf-8

# ## Breast Cnacer Prediction

# ### 1. Load libraries

# In[1]:


#import all the libraries used 
import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


# ### 2. Load Data 

# In[2]:


#load the dataset using read_csv from pandas library
df = pd.read_csv("breast_cancer_data.csv")

#first few rows of data to get basic idea of dataset
df.head()


# ### 3. Clean and prepare the data

# In[3]:


# from the df.head() output we can see that there is one "unnnaamed" column with NAN values. also the column "id" is not 
# useful for analysis. So, we will remove both the columns

df.drop( columns = ['id', 'Unnamed: 32'], inplace = True )


# In[4]:


#check the shape od dataframe
df.shape


# In[5]:


# we have 569 samples and 31 columns. 
# Out of 31 columns 30 are mean, standard error and extreme values of ten features calculated of each sample cells.
# one column - diagnosis is the variable that we will predict.
df.head()


# In[6]:


# Check for missing or null data points
df.isnull().sum()


# In[7]:


df.isna().sum()


# In[8]:


# From the results we can see that there are no missing values ot null values in the dataset


# #### Categorical data

# In[9]:


print(df.diagnosis.unique())
df['diagnosis'].value_counts().to_frame()
# We can see that out of 569 samples we have 212 samples with malignant breast cancer and 357 with benign


# In[10]:


# Check the basic summary of the numerical columns like mean, std, minimum, maximum and quartiles
df.describe()


# ### 4. EDA

# #### with different plots we will check the features with major impact and that can be used for classification

# ### for mean value of features

# In[11]:


f,a = plt.subplots( nrows = 5, ncols = 2, figsize = (10,15))
sns.boxplot(  y="diagnosis", x= "radius_mean", data=df,  orient='h' , palette = 'Blues', ax=a[0][0])
sns.boxplot(  y="diagnosis", x= "texture_mean", data=df,  orient='h' , palette = 'Greens', ax=a[0][1])
sns.boxplot(  y="diagnosis", x= "perimeter_mean", data=df,  orient='h' , palette = 'Reds', ax=a[1][0])
sns.boxplot(  y="diagnosis", x= "area_mean", data=df,  orient='h' , palette = 'binary', ax=a[1][1])
sns.boxplot(  y="diagnosis", x= "smoothness_mean", data=df,  orient='h' , palette = 'Purples', ax=a[2][0])
sns.boxplot(  y="diagnosis", x= "compactness_mean", data=df,  orient='h' , palette = 'Blues', ax=a[2][1])
sns.boxplot(  y="diagnosis", x= "concavity_mean", data=df,  orient='h' , palette = 'Greens', ax=a[3][0])
sns.boxplot(  y="diagnosis", x= "concave points_mean", data=df,  orient='h' , palette = 'Reds', ax=a[3][1])
sns.boxplot(  y="diagnosis", x= "symmetry_mean", data=df,  orient='h' , palette = 'binary', ax=a[4][0])
sns.boxplot(  y="diagnosis", x= "fractal_dimension_mean", data=df,  orient='h' , palette = 'Purples', ax=a[4][1])
plt.tight_layout()


# #### From the above graphs we can see that there are not any particular outilers in th data which we should remove or clean

# 
# Also, we can notice that mean value of radius, perimeter, area, compactness, concavity and concave points can be used in 
# classification of the cancer. As from the graph it can be observed that larger values of these features have correlation 
# with malignant tumors
# 
# Mean values of texture, smoothness and symmetry have some impact on classification of tumor malignant or benign
# 
# But, fractal dimension mean values are not showing any particular preference for one diagnosis over other
# 
# We can check these results of EDA with hypothesis testing anf later build the model on these features.
# 
# 

# ### 5. Hypothesis building and Testing

# #### from above EDA results we can build hypothesis that
# #### 1. Values mean of radius, perimeter, area, compactness, concavity, concave points, texture, smoothness and symmetry have impact on classification of tumor malignant or benign
# 
# #### 2. No difference in  fractal_dimension for benign or malignant tumor
# 
# #### we will use ANOVA test to check if these hypothesis are correct or not

# ### Radius_mean

# In[12]:


## ANOVA for mean_radius
## Null hypothesis: there is no difference in radius_mean for Malignant and Benign tumor
## Alternate hypothesis: there is difference in radius_mean for Malignant and Benign tumor
    
df_anova = df[['radius_mean','diagnosis']]
grps = pd.unique(df_anova.diagnosis.values)
d_data = {grp:df_anova['radius_mean'][df_anova.diagnosis == grp] for grp in grps}
 
F, p = stats.f_oneway(d_data['M'], d_data['B'])
print("p-value for significance is: ", p)
if p<0.05:
    print("reject null hypothesis")
    print("Alternate hypothesis: there is difference in radius_mean for Malignant and Benign tumor")
else:
    print("accept null hypothesis")
    print("Null hypothesis: there is no difference in radius_mean for Malignant and Benign tumor")


# ### Texture_mean

# In[13]:


## ANOVA for texture_mean
## Null hypothesis: there is no difference in texture_mean for Malignant and Benign tumor
## Alternate hypothesis: there is difference in texture_mean for Malignant and Benign tumor
    
df_anova = df[['texture_mean','diagnosis']]
grps = pd.unique(df_anova.diagnosis.values)
d_data = {grp:df_anova['texture_mean'][df_anova.diagnosis == grp] for grp in grps}
 
F, p = stats.f_oneway(d_data['M'], d_data['B'])
print("p-value for significance is: ", p)
if p<0.05:
    print("reject null hypothesis")
    print("Alternate hypothesis: there is difference in texture_mean for Malignant and Benign tumor")
else:
    print("accept null hypothesis")
    print("Null hypothesis: there is no difference in texture_mean for Malignant and Benign tumor")


# ### perimeter_mean

# In[14]:


## ANOVA for perimeter_mean
## Null hypothesis: there is no difference in perimeter_mean for Malignant and Benign tumor
## Alternate hypothesis: there is difference in perimeter_mean for Malignant and Benign tumor
    
df_anova = df[['perimeter_mean','diagnosis']]
grps = pd.unique(df_anova.diagnosis.values)
d_data = {grp:df_anova['perimeter_mean'][df_anova.diagnosis == grp] for grp in grps}
 
F, p = stats.f_oneway(d_data['M'], d_data['B'])
print("p-value for significance is: ", p)
if p<0.05:
    print("reject null hypothesis")
    print("Alternate hypothesis: there is difference in perimeter_mean for Malignant and Benign tumor")
else:
    print("accept null hypothesis")
    print("Null hypothesis: there is no difference in perimeter_mean for Malignant and Benign tumor")


# ### area_mean

# In[15]:


## ANOVA for area_mean
## Null hypothesis: there is no difference in area_mean for Malignant and Benign tumor
## Alternate hypothesis: there is difference in area_mean for Malignant and Benign tumor
    
df_anova = df[['area_mean','diagnosis']]
grps = pd.unique(df_anova.diagnosis.values)
d_data = {grp:df_anova['area_mean'][df_anova.diagnosis == grp] for grp in grps}
 
F, p = stats.f_oneway(d_data['M'], d_data['B'])
print("p-value for significance is: ", p)
if p<0.05:
    print("reject null hypothesis")
    print("Alternate hypothesis: there is difference in area_mean for Malignant and Benign tumor")
else:
    print("accept null hypothesis")
    print("Null hypothesis: there is no difference in area_mean for Malignant and Benign tumor")


# ### smoothness_mean

# In[16]:


## ANOVA for smoothness_mean
## Null hypothesis: there is no difference in smoothness_mean for Malignant and Benign tumor
## Alternate hypothesis: there is difference in smoothness_mean for Malignant and Benign tumor
    
df_anova = df[['smoothness_mean','diagnosis']]
grps = pd.unique(df_anova.diagnosis.values)
d_data = {grp:df_anova['smoothness_mean'][df_anova.diagnosis == grp] for grp in grps}
 
F, p = stats.f_oneway(d_data['M'], d_data['B'])
print("p-value for significance is: ", p)
if p<0.05:
    print("reject null hypothesis")
    print("Alternate hypothesis: there is difference in smoothness_mean for Malignant and Benign tumor")
else:
    print("accept null hypothesis")
    print("Null hypothesis: there is no difference in smoothness_mean for Malignant and Benign tumor")


# ### compactness_mean

# In[17]:


## ANOVA for compactness_mean
## Null hypothesis: there is no difference in compactness_mean for Malignant and Benign tumor
## Alternate hypothesis: there is difference in compactness_mean for Malignant and Benign tumor
    
df_anova = df[['compactness_mean','diagnosis']]
grps = pd.unique(df_anova.diagnosis.values)
d_data = {grp:df_anova['compactness_mean'][df_anova.diagnosis == grp] for grp in grps}
 
F, p = stats.f_oneway(d_data['M'], d_data['B'])
print("p-value for significance is: ", p)
if p<0.05:
    print("reject null hypothesis")
    print("Alternate hypothesis: there is difference in compactness_mean for Malignant and Benign tumor")
else:
    print("accept null hypothesis")
    print("Null hypothesis: there is no difference in compactness_mean for Malignant and Benign tumor")


# ### concavity_mean

# In[18]:


## ANOVA for concavity_mean
## Null hypothesis: there is no difference in concavity_mean for Malignant and Benign tumor
## Alternate hypothesis: there is difference in concavity_mean for Malignant and Benign tumor
    
df_anova = df[['concavity_mean','diagnosis']]
grps = pd.unique(df_anova.diagnosis.values)
d_data = {grp:df_anova['concavity_mean'][df_anova.diagnosis == grp] for grp in grps}
 
F, p = stats.f_oneway(d_data['M'], d_data['B'])
print("p-value for significance is: ", p)
if p<0.05:
    print("reject null hypothesis")
    print("Alternate hypothesis: there is difference in concavity_mean for Malignant and Benign tumor")
else:
    print("accept null hypothesis")
    print("Null hypothesis: there is no difference in concavity_mean for Malignant and Benign tumor")


# ### concave points_mean

# In[19]:


## ANOVA for concave points_mean
## Null hypothesis: there is no difference in concave points_mean for Malignant and Benign tumor
## Alternate hypothesis: there is difference in concave points_mean for Malignant and Benign tumor
    
df_anova = df[['concave points_mean','diagnosis']]
grps = pd.unique(df_anova.diagnosis.values)
d_data = {grp:df_anova['concave points_mean'][df_anova.diagnosis == grp] for grp in grps}
 
F, p = stats.f_oneway(d_data['M'], d_data['B'])
print("p-value for significance is: ", p)
if p<0.05:
    print("reject null hypothesis")
    print("Alternate hypothesis: there is difference in concave points_mean for Malignant and Benign tumor")
else:
    print("accept null hypothesis")
    print("Null hypothesis: there is no difference in concave points_mean for Malignant and Benign tumor")


# ### symmetry_mean

# In[20]:


## ANOVA for symmetry_mean
## Null hypothesis: there is no difference in symmetry_mean for Malignant and Benign tumor
## Alternate hypothesis: there is difference in symmetry_mean for Malignant and Benign tumor
    
df_anova = df[['symmetry_mean','diagnosis']]
grps = pd.unique(df_anova.diagnosis.values)
d_data = {grp:df_anova['symmetry_mean'][df_anova.diagnosis == grp] for grp in grps}
 
F, p = stats.f_oneway(d_data['M'], d_data['B'])
print("p-value for significance is: ", p)
if p<0.05:
    print("reject null hypothesis")
    print("Alternate hypothesis: there is difference in symmetry_mean for Malignant and Benign tumor")
else:
    print("accept null hypothesis")
    print("Null hypothesis: there is no difference in symmetry_mean for Malignant and Benign tumor")


# ### fractal_dimension_mean

# In[21]:


## ANOVA for fractal_dimension_mean
## Null hypothesis: there is no difference in fractal_dimension_mean for Malignant and Benign tumor
## Alternate hypothesis: there is difference in fractal_dimension_mean for Malignant and Benign tumor
    
df_anova = df[['fractal_dimension_mean','diagnosis']]
grps = pd.unique(df_anova.diagnosis.values)
d_data = {grp:df_anova['fractal_dimension_mean'][df_anova.diagnosis == grp] for grp in grps}
 
F, p = stats.f_oneway(d_data['M'], d_data['B'])
print("p-value for significance is: ", p)
if p<0.05:
    print("reject null hypothesis")
    print("Alternate hypothesis: there is difference in fractal_dimension_mean for Malignant and Benign tumor")
else:
    print("accept null hypothesis")
    print("Null hypothesis: there is no difference in fractal_dimension_mean for Malignant and Benign tumor")


# #### From the above Hypothesis test results we can see that out of 10 features of each sample, we can use 9 samples to build the clssification model. As, Fractal Dimension is not having anu particular preference we will not include this feature for our model

# column "diagnosis" in the dataset contains the catagorical data. 
# 
# it has two unique values - M for malignant and B for benign
# 
# we will convert them into numbers do that predictive models can understand them better
# 
# Map M -> 1 and B -> 0

# In[22]:


df['diagnosis'] = df['diagnosis'].map({ 'M' : 1, 'B' : 0})
print(df.diagnosis.unique())


# ### Train-Test splitting

# In[23]:


# will make tweo dataframes X and Y, X will contain all the features which will be used in classification
# Y will contain the target variable
X = df[["radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean", "compactness_mean", 
        "concavity_mean", "concave points_mean", "symmetry_mean"]]

Y = df[["diagnosis"]]


# In[24]:


X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size = 0.25, random_state = 0 )


# In[25]:


X_train.shape


# In[26]:


X_test.shape


# ### Feature scaling

# As all the features in dataset highly vary in magnitudes, units and range, they need to be brought to the same level of magnitude. So that machine learning model will consider all the features equally important.
# 
# We will use Scaling to do the same and transform all features in similar scale

# In[27]:


# scling all the features
sc = StandardScaler()

# Fit and Transfor Train data in a  way so that data has 0 mean and unit variance
# Fromula used: x_scale = (x - mean)/sd

X_train = sc.fit_transform(X_train)

# Transform test data with the same mean and sd
X_test = sc.transform(X_test)


# ### 6. Modeling

# Mainly Machine Learning algorithms can be classified into to groups:
# 1. Supervised learning
# 2. Unsupervised learning
# 
# In Supervised learning input and desired output data are provided and labeled. While, in unsupervised learning information is neither classified or labeled.
# 
# Supervised learning algorithms can further be grouped into two main algorithms - Regression and Classification
# 
# Regression problem is when output variable is continuous like 'Salary' or 'Price' and Classification problem is when output variable is category like 'Malignant' or 'Benign' in our problrm.
# 
# So, here in out problem we will be using different classification algorithms and try to classify if the tumor is malignant or benign and Choose the model with best accuracy.

# In[28]:


#funcction fo find accuracy from confusion matrix
def accuracy(matrix):
    # sum of True Negative and True Positive
    diagonal_sum = matrix.trace()
    
    # sum of total
    sum_total = matrix.sum()
    
    return (diagonal_sum/sum_total)*100


# #### 1. Logistic Regression

# In[29]:


"""
Logistic Regression predicts the probability of occurence of an event by fitting the data in logit function based on given
independent set of variables and estimates discrete values (Binary 0/1, yes/no, T/F)  
"""
# Fit the model
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit( X_train, Y_train )

#predict for the test data
Y_pred_lr = lr.predict(X_test)

#confusion matrix
cm_lr = confusion_matrix(Y_test, Y_pred_lr, labels = [1,0])
print(cm_lr)

#accuracy 
print("Accuracy of Logistic regresion model: " + str(accuracy(cm_lr)) + " %")


# #### 2. Decision Tree

# In[30]:


"""
Decision Tree algorithm splits the data into two or more homogeneous sets based on most significant attributes, making 
groups as distinct as possible
"""
# Fir the model
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit( X_train, Y_train )

# Predict for the test data
Y_pred_dt = dt.predict(X_test)

#Confusion Matrix
cm_dt = confusion_matrix(Y_test, Y_pred_dt, labels = [1,0])
print(cm_dt)

#accuracy 
print("Accuracy of Decision Tree model: " + str(accuracy(cm_dt)) + " %")


# #### 3. Naive-Bayes Classifier

# In[31]:


"""
Naive-Bayes classifier calculates the possibility of whether a data points belongs within a certain class or not. 
It is an extension of Bayes theorem wherein each feature assumes independece
"""
# Fit the model
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit( X_train, Y_train )

# Predict for the test data
Y_pred_nb = nb.predict(X_test)

#Confusion Matrix
cm_nb = confusion_matrix(Y_test, Y_pred_nb, labels = [1,0])
print(cm_nb)

#Accuracy
print("Accuracy of Naive-Bayes Classifier: " + str(accuracy(cm_nb)) + " %")


# #### 4. K-Nearest Neighbors

# In[32]:


"""
K-Nearest Neighbor is a pattern recognition algorithm which basically stores all available cases to classify the new
vase by a majority vote of its k neighbors
"""
# Fit the Model
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier( n_neighbors = 5, metric = 'minkowski', p = 2 )
knn.fit( X_train, Y_train )

# Predict for the test data
Y_pred_knn = knn.predict(X_test)

#Confusion Matrix
cm_knn = confusion_matrix(Y_test, Y_pred_knn, labels = [1,0])
print(cm_knn)

#accuracy
print("Accuracy of the K-Nearest Neighbor algorithm: " + str(accuracy(cm_knn)) + " %")


# #### 4. Support Vector Machine

# In[33]:


"""
SVM plots the each data item as a point in n-dimensional space (n is the number of features) with the value of each feature 
being the value of coordinate.

After that it finds the hyperplane between different classified groups of the data. Depending on where the test data lands,
we can claasify ther class of the data.
"""
# Fit the model
from sklearn.svm import SVC
svm = SVC( kernel = 'linear', random_state = 0 )
svm.fit( X_train, Y_train )

# Predict for the test data
Y_pred_svm = svm.predict( X_test )

# Confusion Matrix
cm_svm = confusion_matrix( Y_test, Y_pred_svm, labels = [1,0] )
print(cm_svm)

# accuracy
print("Acuuracy for the Support Vector Machine: " + str(accuracy(cm_svm)) + " %")


# #### 5. Random Forest 

# In[34]:


"""
Random Forest algorithm is an expansion of decision tree. 

First it constructs some decision trees with the training daa. Then it fit the new data within one of the trees as
"Random Forest"
"""
# Fit the model
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier( n_estimators=20, random_state = 0, max_depth = 4)
rf.fit( X_train, Y_train )

# predict for the test data
Y_pred_rf = rf.predict( X_test )

# Confusion Matrix
cm_rf = confusion_matrix( Y_test, Y_pred_rf, labels = [1,0] )
print(cm_rf)

# accuracy
print("Acuuracy for the Random Forest algorithm: " + str(accuracy(cm_rf)) + " %")


# Random Forest model also returns the feature important matrix which can be used to check importance of features.

# In[35]:


features = ["radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean", "compactness_mean", 
            "concavity_mean", "concave points_mean", "symmetry_mean"]
imp_feature = pd.Series( rf.feature_importances_, index = features ).sort_values( ascending = False )
print(imp_feature)


# From the result we can see that there are 5 important features, highly impacting the diagnosis - concave points_mean, concavity_mean, area_mean, radius_mean, perimeter_mean

# In[36]:


print(cm_rf)


# From the confusion matrix we can see that 52 samples were classified as malignant and were actually malignant in the test
# data and 87 were classified as benign which were actually benign. 
# So, 
# 1. True Positive : 52 
# 2. True Negative : 87
# 
# ALso, out of 143 samples 4 samples were wrongly classified. 
# there were 3 samples which were actually negative but classified as positive and 1 sample which was classified as negative but was positive.
# So,
# 3. False Positive : 3
# 4. False Negative : 1

# #### From the analysis we can see that Random Forest Model with the 9 predictors ("radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean", "compactness_mean",  "concavity_mean", "concave points_mean", "symmetry_mean") is the best model to predict Breast Cancer. It gives the accuracy of 97.20 % with the test data set. 

# In[ ]:




