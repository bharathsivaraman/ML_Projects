 # Import the required libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import DataFrame, Series
import matplotlib.ticker as ticker
from sklearn import model_selection, preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
from sklearn.cross_validation import KFold
import xgboost as xgb
from xgboost import XGBClassifier
import copy
from sklearn.ensemble import RandomForestRegressor as RGR
from sklearn.ensemble import DecisionTreeRegressor  as dt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV


## Read all the files and initial data analysis

train_df = pd.read_csv("train.csv", parse_dates = ["timestamp"])
test_df = pd.read_csv("test.csv", parse_dates = ["timestamp"])
macros_df = pd.read_csv("macro.csv", parse_dates = ["timestamp"])
train_lat_df=pd.read_csv("train_lat_lon.csv")
test_lat_df=pd.read_csv("test_lat_lon.csv")
 

##Create two new columns Year_Sold and Month_Sold using timestamp
 
train_df["year_sold"]=train_df.timestamp.dt.year
train_df["month_sold"]=train_df.timestamp.dt.month


test_df["year_sold"]=test_df.timestamp.dt.year
test_df["month_sold"]=test_df.timestamp.dt.month

 ## Exploratory Data Analysis## Target Variable
sns.distplot(train_df.price_doc, kde = False)

 ## Data Quality checks

train_df.info(verbose = True)
train_dtypes = train_df.dtypes.reset_index()
train_dtypes.columns = ["column_name", "type"]
train_dtypes.groupby("type").count()


 ## Most of the columns are of interger type.Some are categorical

 ## There are too many features.Run a random forest of XG boost to find out important features




########## Baseline model on Imp features #####################

 ### XGB model to get important features
 
train_mod_df=copy.deepcopy(train_df)

for f in train_mod_df.columns:
   if train_mod_df[f].dtype == 'object':
       lbl = preprocessing.LabelEncoder()
       lbl.fit(list(train_mod_df[f].values))
       train_mod_df[f] = lbl.transform(list(train_mod_df[f].values))

train_y = train_mod_df.price_doc.values
train_X = train_mod_df.drop(["id", "timestamp", "price_doc"], axis = 1)

xgb_params = {
   'eta': 0.05,
   'max_depth': 8,
   'subsample': 0.7,
   'colsample_bytree': 0.7,
   'objective': 'reg:linear',
   'eval_metric': 'rmse',
   'silent': 1
 }
dtrain = xgb.DMatrix(train_X, train_y, feature_names = train_X.columns.values)
model = xgb.train(dict(xgb_params, silent = 0), dtrain, num_boost_round = 100)

 # plot the important features#
fig, ax = plt.subplots(figsize = (12, 18))
xgb.plot_importance(model, max_num_features = 50, height = 0.8, ax = ax)
plt.show()

del train_mod_df

#################### End of Baseline Model ################################

## Selecting the top 50 Paramaters

importance = model.get_fscore()
importance_df = pd.DataFrame(pd.Series(importance)).reset_index()
importance_df.columns = ["Column_Name", "Score"]

importance_df = importance_df.sort_values("Score", ascending = False)

 ### Analyzing important features

def describe(df, stats):
   d = df.describe()
   return d.append(df.reindex_axis(d.columns, 1).agg(stats))

summary_stats_df = describe(train_df, ['median', 'skew', 'mad', 'kurt']).transpose().reset_index()
summary_stats_df = summary_stats_df.rename(columns = {
   "index": "column_name"
 })
summary_stats_df = summary_stats_df[summary_stats_df.column_name.isin(importance_df.Column_Name.values[: 50])]

 ## Get Data Frame with important features only

train_imp_df = train_df.iloc[: , train_df.columns.isin(importance_df.Column_Name.values[: 50])]
train_imp_df["price_doc"] = train_df["price_doc"]

##combine test and train to cleanse data
 
test_imp_df=test_df.iloc[: , test_df.columns.isin(importance_df.Column_Name.values[: 50])] 
test_imp_df=test_imp_df.assign(price_doc=0) 
 
combine_df=train_imp_df.append(test_imp_df).reset_index()



########### Collinearity Analysis ###########


numeric_df = combine_df.select_dtypes(include = [np.number])

numeric_df=numeric_df.drop(numeric_df[["price_doc","index"]],axis=1)

def get_redundant_pairs(df):
    '''Get diagonal and lower triangular pairs of correlation matrix'''
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop

get_redundant_pairs(combine_df)


def get_top_abs_correlations(df, n=5):
    au_corr = df.corr().abs().unstack()
    labels_to_drop = get_redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr[0:n]


corr_df= get_top_abs_correlations(numeric_df, 40).reset_index()

corr_df.columns=["Column1","Column2","Corr"]

col_drop=combine_df.columns[combine_df.columns.isin(corr_df["Column2"].unique())]

#### Dropping columns from furtther analysis

combine_df=combine_df.drop(labels=col_drop,axis=1).reset_index()

################ End of Collinearity Analysis ##############


 ########## Missing values analysis ######

combine_df_missing = combine_df.isnull().sum().reset_index()
combine_df_missing.columns = ["Column_Name", "Count"]
combine_df_missing["Percentage"] = combine_df_missing.Count / 30471

 ## Columns having missing values

combine_df_missing[combine_df_missing.Count != 0]## 11 columns have missing values in them 
combine_df_missing.Column_Name[combine_df_missing.Percentage > 0.4].reset_index()


############ End of Missing Values ############


######### EDA of varaibles that are missing ##########
##### Check the demograhic data to see for data error, and missing values

 ## Keep build year and state.Might be important variables
combine_df.groupby("state")["state"].count()

 ## State = 33 is an error convert into 3. Make missing state as 0. State defines the condition of the houses
combine_df["state"] = np.where(combine_df["state"] == 33, 3, combine_df["state"])

 ## From the median price it looks like the state = 1 isthe worse condition and 4 being the bes
combine_df.groupby("state")["price_doc"].median()

 # filling missing values for state by imputing 0
combine_df["state"] = combine_df["state"].fillna(0)## Result of imputing seems accurate.All the houses that have missing values seem to have the lowest median price

tmp=DataFrame(np.where(combine_df["build_year"]
 .isnull(), "missing", combine_df["build_year"])).reset_index()

tmp.columns=["index","build_year"]

tmp.groupby("build_year")["build_year"].count()


del tmp
 

#cleaning up build_year a bit
#set 1691 to 1961
#set 20052009 to 2009
#set 4965 to 1965
#set 215 to 2015
#set 0 and 1 to missing 
#set 2 to 2014 and 3 to 2013, 20 2014

combine_df["build_year"]=np.where(combine_df["build_year"]==1691, 1961,combine_df["build_year"])
combine_df["build_year"]=np.where(combine_df["build_year"]==20052009, 2009,combine_df["build_year"])
combine_df["build_year"]=np.where(combine_df["build_year"]==4965, 1965,combine_df["build_year"])
combine_df["build_year"]=np.where(combine_df["build_year"]==215, 2015,combine_df["build_year"])
combine_df["build_year"]=np.where(combine_df["build_year"]==2, 2014,combine_df["build_year"])
combine_df["build_year"]=np.where(combine_df["build_year"]==3, 2013,combine_df["build_year"])
combine_df["build_year"]=np.where(combine_df["build_year"]==20, 2014,combine_df["build_year"])
combine_df["build_year"]=np.where(combine_df["build_year"]==0, np.nan,combine_df["build_year"])
combine_df["build_year"]=np.where(combine_df["build_year"]==1, np.nan,combine_df["build_year"])

 # drop Hospital beds rion hospital_beds_raion

combine_df = combine_df.drop([ "cafe_sum_500_min_price_avg"], axis = 1)
 
 ## Full_Sq and life_sq
 ## there are houses with life_sq more than full_sq. That is a data error
 
tmp= combine_df[combine_df["life_sq"]>combine_df["full_sq"]]
 
ax=sns.lmplot(x="full_sq",y="price_doc",data=tmp) 
 
ax=sns.lmplot(x="life_sq",y="price_doc",data=tmp) 
 
 ##Where the life_sq<full_sq make it the same as full_sq
 
combine_df["full_sq"]=np.where(combine_df["life_sq"]>combine_df["full_sq"],combine_df["life_sq"],combine_df["full_sq"])

## Full_sq has houses with 0. Convert them to NA
 
tmp= combine_df[combine_df["full_sq"]==0] 
 
 
 
 ## life_sq
 #If life_sq = 0 then its not a residential property
 
combine_df[combine_df["life_sq"]==0]["index"].count()

combine_df[combine_df["life_sq"].isnull()]["index"].count()

#49 properties with 0 life_sq -
#7559 properties with life_sq as null. Make the 0 to NA. 

combine_df["life_sq"]=np.where(combine_df["life_sq"]==0,np.nan,combine_df["life_sq"])

bins=[0,1,10,20,30,40,50,60,70,80,90,100,200,300,400,500,1000,1500,2000,3000,4000,5000,80000]
combine_df.groupby(pd.cut(combine_df["life_sq"],bins,right=False))["life_sq"].count()

combine_df.groupby(pd.cut(combine_df["life_sq"],bins,right=False))["price_doc"].median()
#almost all thes values are between 0 and 10

#Analyzing missing values

train_imp_df[train_imp_df["life_sq"].isna()]["price_doc"].median()
##Looking at the median prices, the life_sq probably ftts between 20 and 30

#Analyzing the outliersss

tmp=train_imp_df[train_df["life_sq"]>100]

train_imp_df[train_imp_df["life_sq"]>100]["life_sq"].count()
test_imp_df[test_imp_df["life_sq"]>100]["life_sq"].count()

#257 houses that have a higher tha usual life_sq in the train data set and 39 in test. Delete them?

ax=sns.lmplot(x="life_sq",y="price_doc",data=tmp )
ax.xaxis.set_major_locator(ticker.MultipleLocator(15))

##No pattern among the high price houses. 

ax=sns.stripplot(x="life_sq",y="price_doc",data=train_imp_df)
ax.xaxis.set_major_locator(ticker.MultipleLocator(20))
ax.xaxis.set_major_formatter(ticker.ScalarFormatter())

#Remove the outlier that is more than 7k

combine_df["life_sq"]=np.where(combine_df["life_sq"]>6000,np.nan,combine_df["life_sq"])

#Distplot
sns.distplot(combine_df["life_sq"].dropna(), kde = False)
 
sns.distplot(np.log1p(combine_df["life_sq"].dropna()), kde = False)
 
## Needs to be transformed ?
 
#Floor 
combine_df.floor.unique()  
sns.distplot(combine_df.floor.dropna(),kde=False)
sns.distplot(np.log1p(combine_df.floor.dropna()),kde=False)

sns.lmplot(x="floor",y="price_doc",data=train_imp_df)


train_imp_df.groupby("floor")["floor"].count()
#Remove outliers and plot

tmp= train_imp_df[["floor","price_doc"]]

tmp["floor"]=np.where(tmp.floor>25,np.nan,tmp.floor)


sns.lmplot(x="floor",y="price_doc",data=tmp)
## Floors > 30 seem to be putliers

train_imp_df.groupby("floor")["price_doc"].median()
train_imp_df[train_imp_df["floor"].isna()]["price_doc"].mean()

##Dnont find annything alarming to clean the data
# The nulls can be imputer after analyzing other variables and running a baseline model

## max_floor

combine_df.max_floor.unique()

sns.distplot(combine_df.max_floor.dropna(),kde=False)


combine_df.groupby("max_floor")["max_floor"].count()

sns.lmplot(x="max_floor",y="price_doc",data=train_imp_df)


train_imp_df.groupby("max_floor")["price_doc"].median()
train_imp_df[train_imp_df["max_floor"].isna()]["price_doc"].mean()


##All the missing values probably fit between 0 and 1 floor based on the median price

##Material

combine_df.material.unique()
combine_df.groupby("material")["material"].count()
sns.boxplot(x="material",y="price_doc",data=train_imp_df)

##Kitch_Sq
#Kitch sq =0 is probably non commmericial space
#Check if any of the apartments have kitchen space greater than fullsq

combine_df[(combine_df["kitch_sq"]>combine_df["full_sq"])]
#17 rows have this. Its a data error.make them null
combine_df["kitch_sq"]=np.where(combine_df["kitch_sq"]>combine_df["full_sq"],np.nan,combine_df["kitch_sq"])

sns.lmplot(x="kitch_sq",y="price_doc",data=combine_df[combine_df.kitch_sq.values<25] )

##Kitchen does not seem to have too many outliers

##Subarea - Drop aubarea as it dosent provide much information

combine_df=combine_df.drop(labels="sub_area",axis=1)


############## End of EDA for demographc varlables with missing values############



####### EDA of Neighbourhood varaibles  ##############

filter_col= [col for col in combine_df if col.endswith('km')]
filter_col.append('price_doc')

nd = pd.melt(combine_df, value_vars = filter_col)
n1 = sns.FacetGrid (nd, col='variable', col_wrap=3, sharex=False, sharey = False)
n1 = n1.map(sns.distplot, 'value')
 
###Plotting neihhbourhood vs price_doc
perm=pd.DataFrame()
for f in filter_col:
    tmp=pd.DataFrame()
    tmp=pd.melt(train_imp_df,value_vars=f)
    tmp['price_doc']=train_imp_df['price_doc']
    perm=perm.append(tmp)

n1 = sns.FacetGrid (perm, col='variable',  col_wrap=3, sharex=False, sharey = False)
n1 = n1.map(plt.scatter, "value", "price_doc")


corr = train_imp_df[filter_col].corr()

print (corr['price_doc'].sort_values(ascending=False) , '\n') #top 15 values
print ('----------------------')

##Observation:
## As 



###### End of Neibourhood Analaysis#####

######### Data Preprocessing  ############


combine_df=combine_df.drop(["level_0","index"],axis=1)


##Split into test and train for the model

train_model_df=combine_df.iloc[train_df["id"]]

imputer = Imputer()
values=train_model_df.values

transformed_values = imputer.fit_transform(values)

transformed_values=pd.DataFrame(transformed_values)

y= transformed_values.iloc[:,-1].values

X_features=train_model_df.columns[0:34]

X_train, X_validation, y_train, y_validation = train_test_split(transformed_values.iloc[:,0:34], y, test_size=0.2)


test_model_df= combine_df[combine_df.price_doc.values==0]

 

values=test_model_df.values

transformed_values = imputer.fit_transform(values)

transformed_values=pd.DataFrame(transformed_values)

X_test=transformed_values.iloc[:,0:34]

######### End of Data Preprocessing ########

########## Model building #################

##RandomForest
model.rgr=RGR()
param_grid = { 
           "n_estimators" : [9, 18, 27, 36, 45, 54, 63],
           "max_depth" : [1, 5, 10, 15, 20, 25, 30],
           "min_samples_leaf" : [1, 2, 4, 6, 8, 10]}
 

grid_search=GridSearchCV(model.rgr,param_grid,cv=5,scoring='neg_mean_squared_error')

grid_search.fit(X_train,y_train)

grid_search.best_params_

grid_search.best_estimator_

cvres=grid_search.cv_results_

for mean_score, params in zip(cvres['mean_test_score'],cvres['params']):
    print (np.sqrt(-mean_score),params)

predictions=grid_search.predict(X_train)
 
rgr_mse=mean_squared_error(y_train,predictions)
rgr_mse=np.sqrt(rgr_mse)




############# Validation Sets ################







########### Submissions ##################

y_pred=grid_search.predict(X_test)    

df_sub = pd.DataFrame({'id': test_df['id'], 'price_doc': y_pred})

df_sub.to_csv('sub.csv', index=False)


########### End of Model Function ##############

## Catergorical variables
categorical_train_df = train_imp_df.select_dtypes(include = [np.object])
categorical_train_df.apply(lambda x: x.nunique())

## sub area has 146 categories.Cannot be used in any tree based alogorithms.Too less variance(maybe drop ? )

# Unique values of objects.Checking to if any categorical variables have been encoded as numeric
unique_values_df = numeric_df.apply(lambda x: x.nunique()).reset_index()
unique_values_df.columns = ["column_name", "count"]

## Checking for categories within the numeric fields
unique_values_df[unique_values_df["count"] == 2]

## None of the numerical columns have categories in them