
# coding: utf-8

# In[41]:


import pandas as pd
from sklearn import neighbors, linear_model, tree
from sklearn.preprocessing import Imputer


data = pd.read_csv("E:\\train.csv")
cols = data.columns
new_data = data.dropna(axis='columns')._get_numeric_data().fillna(-1).astype('float32')
y = new_data['SalePrice']
X = new_data.drop(columns=['SalePrice'])


# In[42]:


dtr = tree.DecisionTreeRegressor()
dtr.fit(X, y)
knn = neighbors.KNeighborsClassifier()
knn.fit(X, y)
log = linear_model.LogisticRegression()
log.fit(X, y)


# In[38]:


test = pd.read_csv("E:\\test.csv")
test_arr = ['Id', 'MSSubClass', 'LotArea', 'OverallQual', 'OverallCond',
       'YearBuilt', 'YearRemodAdd', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF',
       'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea',
       'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr',
       'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageCars',
       'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch',
       'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold']
test_X = test[test_arr]._get_numeric_data().fillna(-1).astype('float32')


# In[39]:


DTR_y = dtr.predict(test_X)
knn_y = knn.predict(test_X)
log_y = log.predict(test_X)


# In[40]:


df_DTR = pd.DataFrame(DTR_y)
df_DTR.index +=1461
df_DTR.rename(columns={0: 'ID', 1: 'SalePrice'}, inplace=True)
df_DTR.to_csv("E:\\DTR.csv")

df_KNN = pd.DataFrame(knn_y)
df_KNN.index +=1461
df_KNN.rename(columns={0: 'ID', 1: 'SalePrice'}, inplace=True)
df_KNN.to_csv("E:\\KNN.csv")

df_LOG = pd.DataFrame(log_y)
df_LOG.index +=1461
df_LOG.rename(columns={0: 'ID', 1: 'SalePrice'}, inplace=True)
df_LOG.to_csv("E:\\LOG.csv")

