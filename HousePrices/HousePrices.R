#HousePrices comptetion on Kaggle#
##Predict Sales Prices in Iowa##

## Data Set has Factor variables, count discriptors, continuous variables, time series variables




##setwd("C:\\Users\\KV352JE\\Documents\\ML_Projects\\Houseprices")

##test <- read.csv("test.csv")
##train <- read.csv("train.csv")

# LoadPackages/files ------------------------------------------------------

#load required libraries function

ipak <- function(pkg) {
  new.pkg <- pkg[!(pkg %in% installed.packages()[, "Package"])]
  if (length(new.pkg))
    install.packages(new.pkg, dependencies = TRUE)
  sapply(pkg, require, character.only = TRUE)
}

# usage

packages <-
  c(
    "dplyr",
    "caret",
    "ggplot2",
    "e1071",
    "lazyeval",
    "plotROC",
    "lift",
    "RANN",
    "pROC",
    "ROCR",
    "corrplot",
    "glmnet",
    "caretEnsemble",
    "lubridate",
    "outliers",
    "Boruta"
  )

ipak(packages)

train <- read.csv("train.csv")
test <- read.csv("test.csv")
test$SalePrice <- 1
combined <- rbind(train, test)

# Function------------

impute.median <-
  function(x)
    replace(x, is.na(x), median(x, na.rm = TRUE))

Histplots <- function(column) {
  g <-
    ggplot(continuous.variables, aes_string(x = column)) + geom_histogram(col = "white", fill = "firebrick") + theme_light()
  
  print(g)
}

readinteger <- function()
{
  n <- readline(prompt = "Enter an integer: ")
  return(as.integer(n))
  
}

# EDA - Feature Engineering --------------------------------------------------


## Factor Variables --------------------------------------------------------


#Check for NA and remove columns where more than 40% are NA

threshold <- nrow(combined) * .4

colnames <- "id"

for (i in 1:ncol(combined)) {
  x <- combined[!complete.cases(combined[i]),][i]
  
  if (nrow(x) > threshold) {
    colnames <- rbind(colnames, colnames(x))
  }
  
}
colnames <- colnames[-1]

##Remove columns with more than 40%NA
combined <- combined %>% select(-one_of(colnames))


combined$MSSubClass <- as.factor(combined$MSSubClass)
#get factor columns
factor.variables <- combined[, lapply(combined, is.factor) == TRUE]

factor.variables$SalePrice <- combined$SalePrice

##remove ordinal factors from data imputation

factor.variables <-
  factor.variables %>% select(
    -one_of(
      "OverallCond",
      "OverallQual",
      "ExterQual",
      "ExterCond",
      "GarageQual",
      "GarageCond",
      "BsmtQual",
      "BsmtCond",
      "HeatingQC",
      "KitchenQual",
      "Neighborhood"
    )
  )


## Check freq and dis of every level in each column ------------
data.levels <- data.frame(
  colname = "x",
  levels = 1,
  Var1 = "x",
  Freq = 1,
  AvgSalePrice = 1
)

data.levels.tmp <- data.levels



for (i in 1:(ncol(factor.variables) - 1))
{
  factor.variables[, i] <- as.character(factor.variables[, i])
  factor.variables[which(is.na(factor.variables[, i]) == TRUE), i] <-
    "Missing"
  factor.variables[, i] <-
    as.factor(as.character(factor.variables[, i]))
  
  colname.group <- colnames(factor.variables)[i]
  data.levels.tmp <-
    data.frame(
      colname = colnames(factor.variables[i]),
      levels = length(unlist(levels(
        factor.variables[[i]]
      ))),
      as.data.frame(table(factor.variables[i]))
    )
  
  
  x <-
    factor.variables %>% select(i, ncol(factor.variables)) %>% group_by_(colname.group) %>% dplyr::summarise(AvgSalePrice =
                                                                                                               mean(SalePrice)) %>%
    rename_(Var1 = colname.group)
  data.levels.tmp <- data.levels.tmp %>% inner_join(x, "Var1")
  
  data.levels <- rbind(data.levels, data.levels.tmp)
}

rm(data.levels.tmp)
rm(x)
data.levels <- data.levels[-1,]
data.levels$Dist <-
  (data.levels$Freq / nrow(factor.variables)) * 100

colname.remove <-
  data.levels %>% filter(Dist > 95) %>% select(colname)

colname.remove <- as.vector(colname.remove$colname)

##Remove columns where distribution of levels  is more than 95%

factor.variables <-
  factor.variables %>% select(-one_of(colname.remove))

data.levels <-
  data.levels %>% group_by(colname) %>% mutate(rnk = min_rank((Dist))) %>% arrange(colname, rnk)
data.levels <-
  data.levels %>% mutate(cumDist = cumsum(Dist)) %>% ungroup()

data.levels <-
  data.levels %>% mutate(newlevel = ifelse(cumDist < 5, "Other", as.character(Var1)))

## Merge levels where distribution is less than 5% --------------------

new <- factor.variables
new[] <-
  lapply(factor.variables, function(x)
    data.levels$newlevel[match(x, data.levels$Var1)])
factor.variables <- new[-26]
factor.variables <- lapply(factor.variables, as.factor)
factor.variables <- as.data.frame(factor.variables)



##Label Encoding of ordinal Variables


combined <-
  combined %>% mutate(ExterQual = ifelse(ExterQual == 'Ex', 5, ifelse(
    ExterQual == 'Gd', 4, ifelse(ExterQual == 'TA', 3, ifelse(
      ExterQual == 'Fa', 2, ifelse(ExterQual == 'Po', 1, 0)
    ))
  )))

factor.variables$ExterQual = factor(combined$ExterQual,
                                    levels = c(0, 1, 2, 3, 4, 5),
                                    ordered = TRUE)

combined <-
  combined %>% mutate(GarageQual = ifelse(GarageQual == 'Ex', 5, ifelse(
    GarageQual == 'Gd', 4, ifelse(GarageQual == 'TA', 3, ifelse(
      GarageQual == 'Fa', 2, ifelse(GarageQual == 'Po', 1, 0)
    ))
  )))

combined <-
  combined %>% mutate(GarageQual = replace(GarageQual, is.na(GarageQual) ==
                                             TRUE, 0))


factor.variables$GarageQual = factor(combined$GarageQual,
                                     levels = c(0, 1, 2, 3, 4, 5),
                                     ordered = TRUE)



combined <-
  combined %>% mutate(GarageCond = ifelse(GarageCond == 'Ex', 5, ifelse(
    GarageCond == 'Gd', 4, ifelse(GarageCond == 'TA', 3, ifelse(
      GarageCond == 'Fa', 2, ifelse(GarageCond == 'Po', 1, 0)
    ))
  )))



combined <-
  combined %>% mutate(GarageCond = replace(GarageCond, is.na(GarageCond) ==
                                             TRUE, 0))

factor.variables$GarageCond = factor(combined$GarageCond,
                                     levels = c(0, 1, 2, 3, 4, 5),
                                     ordered = TRUE)

combined <-
  combined %>% mutate(BsmtQual = ifelse(BsmtQual == 'Ex', 5, ifelse(
    BsmtQual == 'Gd', 4, ifelse(BsmtQual == 'TA', 3, ifelse(
      BsmtQual == 'Fa', 2, ifelse(BsmtQual == 'Po', 1, 0)
    ))
  )))

combined <-
  combined %>% mutate(BsmtQual = replace(BsmtQual, is.na(BsmtQual) == TRUE, 0))


factor.variables$BsmtQual = factor(combined$BsmtQual,
                                   levels = c(0, 1, 2, 3, 4, 5),
                                   ordered = TRUE)

combined <-
  combined %>% mutate(BsmtCond = ifelse(BsmtCond == 'Ex', 5, ifelse(
    BsmtCond == 'Gd', 4, ifelse(BsmtCond == 'TA', 3, ifelse(
      BsmtCond == 'Fa', 2, ifelse(BsmtCond == 'Po', 1, 0)
    ))
  )))

combined <-
  combined %>% mutate(BsmtCond = replace(BsmtCond, is.na(BsmtCond) == TRUE, 0))



factor.variables$BsmtCond = factor(combined$BsmtCond,
                                   levels = c(0, 1, 2, 3, 4, 5),
                                   ordered = TRUE)

combined <-
  combined %>% mutate(ExterCond = ifelse(ExterCond == 'Ex', 5, ifelse(
    ExterCond == 'Gd', 4, ifelse(ExterCond == 'TA', 3, ifelse(
      ExterCond == 'Fa', 2, ifelse(ExterCond == 'Po', 1, 0)
    ))
  )))

combined <-
  combined %>% mutate(ExterCond = replace(ExterCond, is.na(ExterCond) == TRUE, 0))

factor.variables$ExterCond = factor(combined$ExterCond,
                                    levels = c(0, 1, 2, 3, 4, 5),
                                    ordered = TRUE)

##convert as ordered factor overall condition and overall qual

factor.variables$OverallCond <-
  factor(combined$OverallCond,
         levels = c(1, 2, 3, 4, 5, 6, 7, 8, 9),
         ordered = TRUE)


factor.variables$OverallQual <-
  factor(
    combined$OverallQual,
    levels = c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
    ordered = TRUE
  )


##convert into factors

factor.variables$YearBuilt <- as.factor(combined$YearBuilt)
factor.variables$YearRemodAdd <- as.factor(combined$YearRemodAdd)
factor.variables$GarageYrBlt <- combined$GarageYrBlt

factor.variables <-
  factor.variables %>% mutate(GarageYrBlt = ifelse(is.na(GarageYrBlt) == TRUE, as.numeric(YearBuilt), GarageYrBlt))
factor.variables$GarageYrBlt <-
  as.factor(factor.variables$GarageYrBlt)
factor.variables$YrSold <- as.factor(combined$YrSold)

factor.variables$MoSold <- as.factor(combined$MoSold)
factor.variables$MoSold <-
  factor(factor.variables$MoSold,
         levels = c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12))

##Age of the houses
factor.variables$Age <-
  as.numeric(as.character(factor.variables$YrSold)) - as.numeric(as.character(factor.variables$YearBuilt))


## Removing additional near zero variance predictors

nearzero <- nearZeroVar(factor.variables)
nearzero.colnames <- names(factor.variables)[nearzero]


factor.variables <-
  factor.variables %>% select(-one_of(nearzero.colnames))

## Continuous Variables ----------------------------------------------------
## Check for NA's in continuous variables and impute -----------
continuous.variables <-
  combined[, lapply(combined, is.numeric) == TRUE]

continuous.variables <- continuous.variables[, -1]



continuous.variables$Age <-
  as.numeric(as.character(factor.variables$Age))
common_cols <-
  intersect(colnames(factor.variables), colnames(continuous.variables))
continuous.variables <-
  continuous.variables %>% select(-(one_of(common_cols)))

for (i in 1:ncol(continuous.variables))
{
  continuous.variables[[i]] <-
    impute.median(continuous.variables[[i]])
}
 

##Combine all porches
continuous.variables <-
  continuous.variables %>% mutate(totalporch = ScreenPorch + X3SsnPorch +
                                    EnclosedPorch + OpenPorchSF)

##Number of floors  and remove seperate floor area


continuous.variables <-
  continuous.variables %>% mutate(nmbroffloors =
                                    ifelse(
                                      X1stFlrSF != 0 &
                                        X2ndFlrSF == 0,
                                      1,
                                      ifelse(X1stFlrSF != 0 &
                                               X2ndFlrSF != 0, 2, 0)
                                    ))

##combine All  Areas
continuous.variables$TotalArea <-
  continuous.variables$LotFrontage + continuous.variables$LotArea + continuous.variables$MasVnrArea + continuous.variables$BsmtFinSF1 +
  continuous.variables$BsmtFinSF2 + continuous.variables$BsmtUnfSF + continuous.variables$TotalBsmtSF + continuous.variables$X1stFlrSF +
  continuous.variables$X2ndFlrSF + continuous.variables$GrLivArea + continuous.variables$GarageArea + continuous.variables$WoodDeckSF +
  continuous.variables$OpenPorchSF + continuous.variables$EnclosedPorch + continuous.variables$X3SsnPorch +
  continuous.variables$ScreenPorch + continuous.variables$LowQualFinSF + continuous.variables$PoolArea

##Area of both floors
continuous.variables$TotalArea1st2nd <-
  continuous.variables$X1stFlrSF + continuous.variables$X2ndFlrSF

##TotalNumberofbaths

continuous.variables$Totalbath <-
  continuous.variables$BsmtFullBath + continuous.variables$BsmtHalfBath *
  0.5 + continuous.variables$FullBath + continuous.variables$HalfBath * 0.5

#Remove Non Zero Var
nearzero <- nearZeroVar(continuous.variables)
nearzero.colnames <- names(continuous.variables)[nearzero]
continuous.variables <-
  continuous.variables %>% select(-one_of(nearzero.colnames))

#Find collinearity
M <- cor(continuous.variables)
corrplot(M, method = "circle")
collinear <- findCorrelation(M, cutoff = 0.75)

collinear.names <- names(continuous.variables)[collinear]

continuous.variables <-
  continuous.variables %>% select(
    -one_of(
       "GarageCars",
      "X1stFlrSF",
      "X2ndFlrSF",
      "GrLivArea",
      "LotArea"
    )
  )
summary(continuous.variables)


#Remove these variables
#MiscVal,PoolArea,BsmtFinSF2,LowQualFinSF,ScreenPorch,X3SsnPorch,EnclosedPorch,OpenPorchSF,X1stFlrSF,X2ndFlrSF,BsmtUnfSF,BsmtFinSF1

continuous.variables <-
  continuous.variables %>% select(
    -one_of(
      "MiscVal",
      "PoolArea",
      "BsmtFinSF2",
      "LowQualFinSF",
      "ScreenPorch",
      "X3SsnPorch",
      "EnclosedPorch",
      "OpenPorchSF",
      "X1stFlrSF",
      "X2ndFlrSF",
      "BsmtFinSF1",
      "BsmtUnfSF",
      "WoodDeckSF"
    )
  )



# Outlier Detection -------------------------------------------------------


train.outliers <- continuous.variables[1:nrow(train), ]

chisq.out.test(train.outliers$LotFrontage,
               variance = train.outliers$LotFrontage,
               opposite = FALSE)


#Apply log to the following variables
#LotArea,#LotFrontage,#SalePrice,#GarageArea,GrLivArea,BsmtUnfSF,TotalBsmtSF,WoodDeckSF
#

continuous.variables$LotArea <-
  log(continuous.variables$LotArea + 1)
Histplots("LotArea")

continuous.variables$LotFrontage <-
  log(continuous.variables$LotFrontage + 1)
Histplots("LotFrontage")

continuous.variables$GarageArea <-
  log(continuous.variables$GarageArea + 1)
Histplots("GarageArea")

continuous.variables$GrLivArea <-
  log(continuous.variables$GrLivArea + 1)
Histplots("GrLivArea")

continuous.variables$BsmtUnfSF <-
  log(continuous.variables$BsmtUnfSF + 1)
Histplots("BsmtUnfSF")

continuous.variables$TotalBsmtSF <-
  log(continuous.variables$TotalBsmtSF + 1)
Histplots("TotalBsmtSF")

continuous.variables$TotalArea <-
  log(continuous.variables$TotalArea + 1)

continuous.variables$totalporch <-
  log(continuous.variables$totalporch + 1)

continuous.variables$MasVnrArea <-
  log(continuous.variables$MasVnrArea + 1)





####-- Split train and test records and perform onehotencoding on train data set
### OnehotEncoding ----------------------------------------------------------

dmy <-
  dummyVars(
    ~  MSSubClass + MSZoning   + LotShape + LotConfig  + Condition1 +
      BldgType + HouseStyle + RoofStyle + Exterior1st + Exterior2nd + MasVnrType +
      Foundation + BsmtExposure + BsmtFinType1 + CentralAir + Electrical + GarageType +
      GarageFinish + SaleType + SaleCondition ,
    data = factor.variables
  )
trsf <- data.frame(predict(dmy, newdata = factor.variables))


colnames.excl<-c("ExterQual", "GarageQual"   , "GarageCond"   
, "BsmtQual"  ,    "ExterCond"   ,  "OverallCond"  , "OverallQual"  
"Age")



###Feature selection




set.seed(7)
control <-
  rfeControl(functions = rfFuncs,
             method = "repeatedcv",
             number = 10)
# run the RFE algorithm
results <-
  rfe(train.predict[, !names(train.predict) %in% c("SalePrice", "Id")],
      train.predict[, names(train.predict) %in% "SalePrice"],
      sizes = c(1:49),
      rfeControl = control)

predictors(results)

colnames.optvar <- results$optVariables
colnames.optvar[46] <- "SalePrice"
train.predict <- train.predict %>% select(one_of(colnames.optvar))




##with one hot encoding

encode.onehot <- readinteger()

if (encode.onehot == 1) {
  combined.clean <- cbind(trsf, continuous.variables)
} else {
  combined.clean <- cbind(factor.variables, continuous.variables)
  
}
combined.clean$Id <- combined$Id
combined.NA <- combined.clean[!complete.cases(combined.clean),]
summary(combined.clean)



## Removing additional near zero variance predictors

nearzero <- nearZeroVar(combined.clean)
nearzero.colnames <- names(combined.clean)[nearzero]
combined.clean <-
  combined.clean %>% select(-one_of(nearzero.colnames))






train.predict <-
  combined.clean %>% filter(Id %in% train[["Id"]]) %>% select(-one_of("YearBuilt", "YearRemodAdd", "GarageYrBlt")) %>%
  droplevels()


test.predict <-
  combined.clean %>% filter(Id %in% test[["Id"]]) %>% select(-one_of("YearBuilt", "YearRemodAdd", "GarageYrBlt")) %>%
  droplevels()




# summarize the results



# Model Building --------------------------------------------------
## Try out different models with CV and tuning parameters
train.predict <- train.predict %>% select(-one_of("Id"))

# predictors <- SalePrice ~ .
#
# ##Linear Regression
# trcntrl.lm <- trainControl(
#   method = "repeatedcv",
#   number = 10,
#   repeats = 3,
#   savePredictions = TRUE
# )
#
# set.seed(1234)
# baselineLM <- train(
#   predictors,
#   data = train.predict,
#   method = "lm",
#   trControl = trcntrl,
#   tuneLength = 5,
#   preProcess = c("center", "scale")
# )
#
# #Ridge
#
# trcntrl.ridge <- trainControl(method = "cv",
#                               number = 10, savePredictions = TRUE)
# # Set seq of lambda to test
# lambdaGrid <- expand.grid(lambda = 10 ^ seq(10,-2, length = 100))
# set.seed(1234)
# ridge <- train(
#   predictors,
#   data = train.predict,
#   method = 'ridge',
#   trControl = trcntrl.ridge,
#   tuneGrid = lambdaGrid
# )
#
# #lasso
#
# predictors <- SalePrice ~ .
#
#
#
# tuneGrid = expand.grid(.alpha = 1,
#                        .lambda = seq(0, 100, by = 0.1))
#
# set.seed(1234)
# lasso <- train(
#   predictors,
#   data = train.predict,
#   method = 'glmnet',
#   trControl = trcntrl.ridge,
#   tuneGrid = tuneGrid
# )

#GBM


predictors <-
  SalePrice ~  TotalArea + GarageArea + TotalBsmtSF + Totalbath + Foundation.PConc + Fireplaces +
  FullBath + CentralAir.N + Foundation.Other + CentralAir.Y + GarageType.Attchd +
  TotRmsAbvGrd + HalfBath + BedroomAbvGr + MasVnrArea + HouseStyle.2Story +
  GarageType.Detchd + nmbroffloors + totalporch + BsmtFullBath

trcntrl.ridge <- trainControl(method = "loocv",
                              number = 10,
                              savePredictions = TRUE)

set.seed(1232)
gbmGrid <-  expand.grid(
  interaction.depth = c(1, 5, 9),
  n.trees = (1:30) * 50,
  shrinkage = 0.1,
  n.minobsinnode = 20
)
set.seed(2345)
gbm <- train(
  SalePrice ~ .,
  data = train.predict,
  method = 'gbm',
  trControl = trcntrl.ridge,
  tuneGrid = gbmGrid ,
  preProcess = c("center", "scale")
)



##XGBtree

set.seed(3567)
cv.ctrl <-
  trainControl(method = "cv",
               repeats = 10,
               number = 3)

train.predict.xgb <- train.predict %>% select(-one_of("SalePrice"))
trainx <- Matrix(data.matrix(train.predict.xgb), sparse = TRUE)
trainy <- as.numeric(train.predict$SalePrice)
#inputValid <- Matrix(data.matrix(validationdata[,c(1:numCol),with=FALSE]), sparse=TRUE)

xgbGrid <- expand.grid(
  nrounds = c(10000),
  max_depth = seq(3, 6, by = 1),
  eta = seq(0.03, 0.05, by = 0.01),
  gamma = seq(0, 1, by = 1),
  colsample_bytree = seq(0.4, 0.6, by = 0.1),
  min_child_weight = seq(1, 1, by = 0.5),
  subsample = seq(0.4, 0.6, by = 0.1)
)

xgb_tune <- train(
  predictors,
  data = train.predict,
  method = "xgbTree",
  trControl = cv.ctrl,
  tuneGrid = xgbGrid,
  verbose = T,
  
  nthread = 3
)





# Model Evaluation --------------------------------------------------------




plot.rsquare <-
  gbm$pred %>% filter(n.trees == 100,
                      interaction.depth == 9,
                      shrinkage == 0.1,
                      n.minobsinnode == 20) %>% select(pred, obs)

plot.rsquare <- plot.rsquare %>% mutate(Rsquare = caret::R2(pred, obs))

### plotting observed vs predicted

ggplot(plot.rsquare) + aes(obs, pred) + geom_point() + geom_smooth(method =
                                                                     'lm', formula = y ~ x)


RMSE(log(plot.rsquare$pred), log(plot.rsquare$obs))

# Ensemble model-----------------------------------------------------------

control <-
  trainControl(
    method = "repeatedcv",
    number = 10,
    repeats = 3,
    savePredictions = TRUE
  )
algorithmList <- c('lda', 'gbm')
set.seed(7)
models <-
  caretList(
    predictors ~ .,
    data = train.predict,
    trControl = control,
    methodList = algorithmList
  )
results <- resamples(models)
summary(results)
dotplot(results)




##----  CSV output

rf.model <- predict(randomforest, test.predict)
rf.model <- as.data.frame(rf.model)
rf.model <-
  data.frame(id = test$Id, SalePrice = rf.model$rf.model)
write.csv(rf.model, "rf.model.csv", row.names = FALSE)

gbm.model1 <- predict(gbm, test.predict)
gbm.model1 <- as.data.frame(gbm.model1)
gbm.model1 <-
  data.frame(id = test$Id, SalePrice = gbm.model1$gbm.model1)



write.csv(gbm.model1, "gbm.model1.csv", row.names = FALSE)



xgb.model <- predict(xgb_tune, test.predict)
xgb.model <- as.data.frame(xgb.model)
xgb.model <-
  data.frame(id = test$Id, SalePrice = xgb.model$xgb.model)
write.csv(xgb.model, "xgb.model.csv", row.names = FALSE)


saveRDS(gbm.model, "gbm.model")
saveRDS(gbm.model1, "gbm.model1")