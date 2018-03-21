 # Import the required libraries

 import pandas as pd
 import numpy as np
 import matplotlib.pyplot as plt
 import seaborn as sns
 from pandas import DataFrame, Series
 import matplotlib.ticker as ticker

 from sklearn import model_selection, preprocessing
 from sklearn.feature_selection import VarianceThreshold
 import xgboost as xgb
 from xgboost import XGBClassifier
 
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

 ### XGB model to get important features
 
 train_mod_df=train_df

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

 ## Missing values

 combine_df_missing = combine_df.isnull().sum().reset_index()
 combine_df_missing.columns = ["Column_Name", "Count"]
 combine_df_missing["Percentage"] = combine_df_missing.Count / 30471

 ## Columns having missing values

 combine_df_missing[combine_df_missing.Count != 0]## 11 columns have missing values in them 

 combine_df_missing.Column_Name[combine_df_missing.Percentage > 0.4].reset_index()

 ## Keep build year and state.Might be important variables

 combine_df.groupby("state")["state"].count()

 ## State = 33 is an error convert into 3. Make missing state as 0. State defines the condition of the houses

 combine_df["state"] = np.where(combine_df["state"] == 33, 3, combine_df["state"])

 ## From the median price it looks like the state = 1 isthe worse condition and 4 being the best

 combine_df.groupby("state")["price_doc"].median()

 # filling missing values for state by imputing 0
 combine_df["state"] = combine_df["state"].fillna(0)## Result of imputing seems accurate.All the houses that have missing values seem to have the lowest median price


del tmp

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

 combine_df = combine_df.drop(["hospital_beds_raion"], axis = 1)
 
 ## Full_Sq and life_sq
 ## there are houses with life_sq more than full_sq. That is a data error
 
 tmp= combine_df[combine_df["life_sq"]>combine_df["full_sq"]]
 
 ax=sns.lmplot(x="full_sq",y="price_doc",data=tmp) 
 
 ax=sns.lmplot(x="life_sq",y="price_doc",data=tmp) 
 
 ##Where the life_sq<full_sq make it the same as full_sq
 
 combine_df["full_sq"]=np.where(combine_df["life_sq"]>combine_df["full_sq"],combine_df["life_sq"],combine_df["full_sq"])
 
 del tmp
## Full_sq has houses with 0. Convert them to NA
 
tmp= combine_df[combine_df["full_sq"]==0] 
 
 
 
 ## life_sq
 ## life_sq has values that start with 0. If this field is the size of living area then it should not be 0

#Make 0 life_sq as NA. They have been incorrectly coded as 0

combine_df["life_sq"]=np.where(combine_df["life_sq"]==0,np.nan,combine_df["life_sq"])


bins=[0,1,10,20,30,40,50,60,70,80,90,100,200,300,400,500,1000,1500,2000,3000,4000,5000,80000]
combine_df.groupby(pd.cut(combine_df["life_sq"],bins,right=False))["life_sq"].count()

combine_df.groupby(pd.cut(combine_df["life_sq"],bins,right=False))["price_doc"].median()
#almost all thes values are between 0 and 100. Median prices of some range is not propotional to the size

#Analyzing missing values

train_imp_df[train_imp_df["life_sq"].isna()]["price_doc"].mean()

#Analyzing the outliersss

tmp=train_df[train_df["life_sq"]>100]

ax=sns.stripplot(x="life_sq",y="price_doc",data=tmp )
ax.xaxis.set_major_locator(ticker.MultipleLocator(15))
 
 
 ax=sns.stripplot(x="life_sq",y="price_doc",data=train_imp_df)
ax.xaxis.set_major_locator(ticker.MultipleLocator(20))
ax.xaxis.set_major_formatter(ticker.ScalarFormatter())


 ## Catergorical variables
 categorical_train_df = train_imp_df.select_dtypes(include = [np.object])
 categorical_train_df.apply(lambda x: x.nunique())

 ## sub area has 146 categories.Cannot be used in any tree based alogorithms.Too less variance(maybe drop ? )
 numeric_df = train_imp_df.select_dtypes(include = [np.number])

 # Unique values of objects.Checking to if any categorical variables have been encoded as numeric
 unique_values_df = numeric_df.apply(lambda x: x.nunique()).reset_index()
 unique_values_df.columns = ["column_name", "count"]

 ## Checking for categories within the numeric fields
 unique_values_df[unique_values_df["count"] == 2]

 ## None of the numerical columns have categories in them