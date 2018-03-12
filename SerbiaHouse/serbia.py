 
##Import the required libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import DataFrame, Series
import ggplot as ggplot
from sklearn import model_selection, preprocessing
from sklearn.feature_selection import VarianceThreshold
import xgboost as xgb
from xgboost import XGBClassifier
## Read all the files and initial data analysis 

train_df=pd.read_csv("train.csv",parse_dates=["timestamp"])
test_df=pd.read_csv("test.csv",parse_dates=["timestamp"])
macros_df=pd.read_csv("macro.csv",parse_dates=["timestamp"])

##Exploratory Data Analysis
##Target Variable 
sns.distplot(train_df.price_doc,kde=False)
 
##Data Quality checks

train_df.info(verbose=True) 
train_dtypes=train_df.dtypes.reset_index() 
train_dtypes.columns=["column_name","type"] 
train_dtypes.groupby("type").count()

##Most of the columns are of interger type. Some are categorical

##Missing values

train_missing=train_df.isnull().sum().reset_index()
train_missing.columns=["Column_Name","Count"]
train_missing["Percentage"]=train_missing.Count/30471

##Columns having missing values

train_missing[train_missing.Count!=0]
## 132 columns have missing values in them. A lot of them have missing values in the 

train_missing.Column_Name[train_missing.Percentage>0.4].reset_index() 
categorical_train_df=train_df.select_dtypes(include=[np.object]) 
categorical_train_df.apply(lambda x: x.nunique())
 
##Zero Variance  
numeric_df=train_df.select_dtypes(include=[np.number])
 
#Unique values of objects 
unique_values_df= numeric_df.apply(lambda x: x.nunique()).reset_index()
unique_values_df.columns=["column_name","count"]

##Checking for categories within the numeric fields
unique_values_df[unique_values_df["count"]==2]
 
##None of the numerical columns have categories in them

##There are too many features. Run a random forest of XG boost to find out important features

###XGB model to get important features

for f in train_df.columns:
    if train_df[f].dtype=='object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(train_df[f].values)) 
        train_df[f] = lbl.transform(list(train_df[f].values))
        
train_y = train_df.price_doc.values
train_X = train_df.drop(["id", "timestamp", "price_doc"], axis=1)

xgb_params = {
    'eta': 0.05,
    'max_depth': 8,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'silent': 1
}
dtrain = xgb.DMatrix(train_X, train_y, feature_names=train_X.columns.values)
model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=100)

# plot the important features #
fig, ax = plt.subplots(figsize=(12,18))
xgb.plot_importance(model, max_num_features=50, height=0.8, ax=ax)
plt.show()

##Selecting the top 50 Paramaters
importance = model.get_fscore()
importance_df=pd.DataFrame(pd.Series(importance ) ).reset_index()
importance_df.columns=["Column_Name","Score"]

importance_df=importance_df.sort_values("Score",ascending=False)

###Analyzing important features

def describe(df,stats):
    d=df.describe() 
    return d.append(df.reindex_axis(d.columns,1).agg(stats))

 
    
summary_stats_df=describe(train_df, ['median','skew', 'mad', 'kurt']).transpose().reset_index()
summary_stats_df=summary_stats_df.rename(columns={"index":"column_name"})
summary_stats_df=summary_stats_df[summary_stats_df.column_name.isin(importance_df.Column_Name.values[:50])]


##Full_Sq

 train_df["price_doc"][train_df.full_sq==0]

sns.lmplot(y=np.log1p("full_sq"),x=np.log1p("price_doc"),data=train_df,fit_reg=False)

sns.lmplot(x="price_doc",y="life_sq",data=train_df,fit_reg=False)
