
# coding: utf-8

# Information about all the attributes can be found here:
# 
# https://www.kaggle.com/c/allstate-claims-severity/data

# The aim of the challenge is to predict the 'loss' based on the variables in the dataset. Hence, this is a regression problem.

# In[28]:


# Read raw data from the file



import numpy as np
import pandas as pd
from pandas import Series,DataFrame

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

import seaborn as sns

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

import os,sys
from scipy import stats


# Read the train dataset

# In[34]:


allstate = pd.read_csv('train.csv',skipinitialspace=True)


# Display the first five rows of the data

# In[35]:


allstate.head()


# 
# Data statistics : Description

# In[39]:


allstate.describe()


#   Machine Learning Algorithms : Evaluation and Prediction
# 
#   Linear Regression 

# In[40]:



# Import for Linear Regression
import sklearn
from sklearn.linear_model import LinearRegression

# Create a LinearRegression Object
lreg = LinearRegression()



# Data Preparation: Categorical Variables

# In[41]:


# Binarizing Categorical Variables

import pandas as pd
features = allstate.columns
cats = [feat for feat in features if 'cat' in feat]
for feat in cats:
    allstate[feat] = pd.factorize(allstate[feat], sort=True)[0]


# Splitting the data into train and test

# In[42]:


# Preparing data for train and test split 
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error
import numpy 

X=allstate.drop(['id','loss'],1).fillna(value=0)
Y=allstate['loss']
X.head()


seed = 8
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=test_size, random_state=seed)


# In[43]:



# Implement Linear Regression

lreg.fit(X_train,y_train)       


# In[44]:


# Predictions on training and testing sets
pred_train = lreg.predict(X_train)
pred_test = lreg.predict(X_test)


# Accuracy of the model :Linear regression

# In[45]:



from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, pred_test))
print('MSE:', metrics.mean_squared_error(y_test, pred_test))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, pred_test)))


# 
#   Machine learning algorithm : Ridge Regression for Evaluation, prediction
# 

# In[46]:


# Implement Ridge Regression

from sklearn import linear_model
reg = linear_model.Ridge (alpha = 0.1)
reg.fit (X_train,y_train)


# In[47]:


# Predictions on training and testing sets
pred_train = reg.predict(X_train)
pred_test = reg.predict(X_test)


from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, pred_test))
print('MSE:', metrics.mean_squared_error(y_test, pred_test))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, pred_test)))


#  Machine learning algorithm : Lasso Regression for Evaluation, prediction
# 

# In[50]:


from sklearn import linear_model
reg = linear_model.Lasso(alpha = 0.1)
reg.fit (X_train,y_train)

pred_test = reg.predict(X_test)


from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, pred_test))
print('MSE:', metrics.mean_squared_error(y_test, pred_test))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, pred_test)))


#  Machine learning algorithm : KNN for Evaluation, prediction
# 

# In[16]:


# loading library
from sklearn.neighbors import KNeighborsRegressor

# instantiate learning model (k = 3)
knn = KNeighborsRegressor(n_neighbors=3)

# fitting the model
knn.fit(X_train, y_train)

# predict the response
pred = knn.predict(X_test)



from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, pred))
print('MSE:', metrics.mean_squared_error(y_test, pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, pred)))


#  Machine learning algorithm : Decisiontrees for Evaluation, prediction

# In[49]:


from sklearn import tree
model = tree.DecisionTreeRegressor()
model.fit(X_train, y_train)
# Predictions on training and testing sets
pred_train = model.predict(X_train)
pred_test = model.predict(X_test)

from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, pred_test))
print('MSE:', metrics.mean_squared_error(y_test, pred_test))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, pred_test)))


# Decisiontrees for Evaluation, prediction with cross validation

# In[18]:



from sklearn import tree
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score
from sklearn import tree

clf_gini = DecisionTreeRegressor(random_state = 100,
                               max_depth=5, min_samples_leaf=2)
clf_gini.fit(X_train, y_train)

# Predictions on training and testing sets
pred_train = clf_gini.predict(X_train)
pred_test = clf_gini.predict(X_test)

from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, pred_test))
print('MSE:', metrics.mean_squared_error(y_test, pred_test))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, pred_test)))


# Decisiontrees for Evaluation, prediction with cross validation and hyperparameter tuning

# In[19]:


from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics
from sklearn import grid_search
param_grid = {'max_depth': np.arange(3, 10)}

tree = grid_search.GridSearchCV(DecisionTreeRegressor(), param_grid)

tree.fit(X_train, y_train)
pred_test = tree.predict(X_test)
tree_performance = metrics.mean_absolute_error(y_test, pred_test)

print (format(tree_performance))


# Randomforest  for Evaluation, prediction

# In[20]:



#random forest
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=5)
model.fit(X_train, y_train)
# Predictions on training and testing sets
pred_train = model.predict(X_train)
pred_test = model.predict(X_test)

from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, pred_test))
print('MSE:', metrics.mean_squared_error(y_test, pred_test))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, pred_test)))


# In[22]:


#grid search cv for best params
#couldnt do it for computational reasons


#n_features = X.shape[1]

#from sklearn.grid_search import RandomizedSearchCV
#grid = RandomizedSearchCV(model, n_iter=20,
 #           param_distributions=dict(
  #                                        max_depth=np.arange(5,20+1),
   #                                       max_features=np.arange(1, n_features+1)
    #                                )
    #     )
#grid.fit(X, Y)
#print(grid.best_params_)
#
#model = RandomForestRegressor(max_features=grid.best_params_["max_features"],
 #                             max_depth=grid.best_params_["max_depth"])
#model.fit(train_X, train_y)
# Predictions on training and testing sets
#pred_train = model.predict(X_train)
#pred_test = model.predict(X_test)

#from sklearn import metrics
#print('MAE:', metrics.mean_absolute_error(y_test, pred_test))
#print('MSE:', metrics.mean_squared_error(y_test, pred_test))
#print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, pred_test)))


# Adaboost regression for Evaluation, prediction
# 

# In[11]:


# importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor


# In[16]:



regr = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4),
                          n_estimators=300, random_state = 100)

regr.fit(X_train, y_train)


# In[17]:


# Predictions on training and testing sets
pred_train = regr.predict(X_train)
pred_test = regr.predict(X_test)

from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, pred_test))
print('MSE:', metrics.mean_squared_error(y_test, pred_test))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, pred_test)))


# Gradient boosting trees regression for Evaluation, prediction
# 

# In[18]:


from sklearn.ensemble import GradientBoostingRegressor

import numpy as np
import matplotlib.pyplot as plt

from sklearn import ensemble
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error


# In[20]:


params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
          'learning_rate': 0.01, 'loss': 'ls'}
clf = ensemble.GradientBoostingRegressor(**params)

clf.fit(X_train, y_train)
mse = mean_squared_error(y_test, clf.predict(X_test))
print("MSE: %.4f" % mse)


# In[24]:


print('MAE:', metrics.mean_absolute_error(y_test, clf.predict(X_test)))

from xgboost import XGBRegressor


# Extreme Gradient boosting trees regression for Evaluation, prediction

# In[25]:


# fitting model on training data

model = XGBRegressor(max_depth=6, n_estimators=500, learning_rate=0.1, subsample=0.8, colsample_bytree=0.4,
                     min_child_weight = 3,  seed=7)
model.fit(X_train, y_train)


# In[26]:


# Predictions on training and testing sets
pred_train = model.predict(X_train)
pred_test = model.predict(X_test)

from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, pred_test))
print('MSE:', metrics.mean_squared_error(y_test, pred_test))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, pred_test)))


# In[16]:


#ensemble of models

from sklearn.datasets import load_boston
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from vecstack import stacking


X_train, X_test, y_train, y_test = train_test_split(X, Y, 
    test_size = 0.2, random_state = 0)


# Caution! All models and parameter values are just 
# demonstrational and shouldn't be considered as recommended.
# Initialize 1-st level models.

models = [
    ExtraTreesRegressor(random_state = 0, n_jobs = -1, 
        n_estimators = 100, max_depth = 3),
        
    RandomForestRegressor(random_state = 0, n_jobs = -1, 
        n_estimators = 100, max_depth = 3),
        
    XGBRegressor(seed = 0, nthread = -1, learning_rate = 0.1, 
        n_estimators = 100, max_depth = 3)]
    


# In[47]:


#S_train, S_test = stacking(models, X_train, y_train, X_test, 
   # regression = True, metric = mean_absolute_error, n_folds = 4, 
#random_state = 0, verbose = 2)


# Multilayer perceptron using keras and Tensorflow

# In[4]:



import numpy
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# importing the  training and test data

# In[ ]:


df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')
print('training: ', df_train.shape)
print('test: ', df_test.shape)


# Data preparation

# In[5]:


#Convert to Numpy arrays and separate features/targets
training_samples = df_train.as_matrix()
training_targets = training_samples[:,-1]
training_samples = training_samples[:,1:-1]

test_samples = df_test.as_matrix()
test_samples = test_samples[:,1:]


#Encode the Labels of the categorical data
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
# [0:116]
allLabels = np.concatenate( ( training_samples[:, 0:116].flat , test_samples[:, 0:116].flat ) )
le.fit( allLabels )
del allLabels

#Transform the labels to int values
for colIndex in range(116):
    training_samples[:, colIndex] = le.transform(training_samples[:, colIndex])
    test_samples[:, colIndex] = le.transform( test_samples[:, colIndex] )


# splitting the data into train and test

# In[6]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(training_samples, training_targets)


# Scaling the independent variables

# In[7]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# imporitng the dependencies

# In[19]:


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasRegressor


# Initiating the model:

# In[20]:


Regressor =Sequential()

Regressor.add(Dense(output_dim = 66, init = 'uniform', activation ='relu', input_dim =130))


# In[21]:


Regressor.add(Dense(output_dim = 66, init = 'uniform', activation ='relu'))


# In[22]:


#output layer

Regressor.add(Dense(output_dim = 1, init = 'uniform', activation ='relu', input_dim =131))
Regressor.add(Dropout(p=0.1))


# In[23]:


Regressor.compile(optimizer='adam',loss='mean_absolute_error')


# In[26]:


Regressor.fit(X_train, y_train, batch_size =20, nb_epoch=50)


# In[52]:


#Evaluating, Improving and Tuning the ANN

# Evaluating the ANN
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense
def build_Regressor():
    Regressor = Sequential()
    Regressor.add(Dense(output_dim = 66, init = 'uniform', activation ='relu', input_dim =130))
    Regressor.add(Dense(output_dim = 66, init = 'uniform', activation ='relu'))
    Regressor.add(Dense(output_dim = 1, init = 'uniform', activation ='relu', input_dim =131))
    Regressor.compile(optimizer='adam',loss='mean_absolute_error')
    return Regressor
Regressor = KerasRegressor(build_fn = build_Regressor, batch_size = 10, epochs = 100)
accuracies = cross_val_score(estimator = Regressor, X = X_train, y = y_train, cv = 10, n_jobs = -1)
mean = accuracies.mean()
variance = accuracies.std()


# In[15]:


from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
def build_Regressor(optimizer):
    Regressor = Sequential()
    Regressor.add(Dense(output_dim = 66, init = 'uniform', activation ='relu', input_dim =130))
    Regressor.add(Dense(output_dim = 66, init = 'uniform', activation ='relu'))
    Regressor.add(Dense(output_dim = 1, init = 'uniform', activation ='relu', input_dim =131))
    Regressor.compile(optimizer='adam',loss='mean_absolute_error')
    return Regressor
Regressor = KerasRegressor(build_fn = build_Regressor)
parameters = {'batch_size': [25, 32],
              'epochs': [10, 15],
              'optimizer': ['adam', 'rmsprop']}
grid_search = GridSearchCV(estimator = Regressor,
                           param_grid = parameters,
                           cv = 10)
grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_


# Model comparison: 
#     
# Linear Regression:  MAE: 1331.211
# RIDGE  Regression:  MAE: 1331.230
# lasso  Regression:  MAE: 1330.936
# knn              :  MAE: 1715.262
# 
# decision trees    : MAE: 1745.644
# decision trees 
# with cross 
# validation        : MAE: 1454.944
# cart with 
# parameter tuning  :MAE :1378.320
# random forest     :MAE: 1375.474
# adaboost          :MAE: 6840.072
# gbm               :MAE: 1271.082
# xgboost           :MAE: 1183.753
# Multi layer
# Perceptron         :MAE: 1192.468
# 
# 
#     

# Extreme gradient boosting algorithm is doing a better job in explaining the model.
# 
# The algorithms have been run without any fine tuning due to computational reasons.
# 
# The models use all the variables given in the data set and I'll be working to findout the best 
# predictors and run further analysis on the data.
# 
# The Multilayer perceptron and the xgboost algorithms can be fine tuned to better the MAE.
# 
# Will be experimenting on how to stack models to gain a better accuracy in the next versions of this kernel
