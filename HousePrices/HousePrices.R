#HousePrices comptetion on Kaggle#
##Predict Sales Prices in Iowa##

## Data Set has Factor variables, count discriptors, continuous variables, time series variables




##setwd("C:\\Users\\sivarbh\\Documents\\ML_Projects\\Houseprices")

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
    "glmnet"
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
      "KitchenQual"
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


## Removing additional near zero variance predictors

nearzero <- nearZeroVar(factor.variables)
nearzerocolnames <- NA
for (i in (1:length(nearzero))) {
  x <- colnames(factor.variables)[nearzero[[i]]]
  nearzerocolnames <- rbind(nearzerocolnames, x)
}
nearzerocolnames <- nearzerocolnames[-1]
factor.variables <-
  factor.variables %>% select(-one_of(nearzerocolnames))


####-- Split train and test records and perform onehotencoding on train data set
### OnehotEncoding ----------------------------------------------------------

dmy <-
  dummyVars(
    ~ MSSubClass + MSZoning + LotShape + LotConfig + Neighborhood + Condition1 +
      BldgType + HouseStyle + RoofStyle + Exterior1st + Exterior2nd + MasVnrType +
      Foundation + BsmtExposure + BsmtFinType1 + CentralAir + Electrical + GarageType +
      GarageFinish + PavedDrive + SaleType + SaleCondition + YrSold:MoSold,
    data = factor.variables
  )
trsf <- data.frame(predict(dmy, newdata = factor.variables))

## Continuous Variables ----------------------------------------------------
## Check for NA's in continuous variables and impute -----------
continuous.variables <-
  combined[, lapply(combined, is.numeric) == TRUE]
continuous.variables <- continuous.variables[, -1]

for (i in 1:ncol(continuous.variables))
{
  continuous.variables[[i]] <-
    impute.median(continuous.variables[[i]])
}

for (i in 1:ncol(continuous.variables))
{
  colname <- colnames(continuous.variables[i])
  
  Histplots(colname)
  
}





#combine All Porches and remove individual porches
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
#
# #PriceperSqft
#
# continuous.variables<-continuous.variables%>%mutate(pricesqft=SalePrice/LotArea)
#



#Remove Non Zero Var
nearzero <- nearZeroVar(continuous.variables)
nearzerocolnames <- NA
for (i in (1:length(nearzero))) {
  x <- colnames(continuous.variables)[nearzero[[i]]]
  nearzerocolnames <- rbind(nearzerocolnames, x)
}
nearzerocolnames <- nearzerocolnames[-1]
continuous.variables <-
  continuous.variables %>% select(-one_of(nearzerocolnames))

#Find collinearity
M <- cor(continuous.variables)
corrplot(M, method = "circle")

findCorrelation(M, cutoff = 0.75, verbose = TRUE)


#There is not much colinearity betweeb variables. Removing TotalRmsabvgrnd and Numberof Garage cars


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


#
# for (i in 1:ncol(continuous.variables)){
# x<-BoxCoxTrans(continuous.variables[[i]])
# continuous.variables[[i]]<-predict(x,continuous.variables[[i]])
# }
##Combine continuous and factor variables

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


train.predict <- combined.clean %>% filter(Id %in% train[["Id"]])
test.predict <- combined.clean %>% filter(Id %in% test[["Id"]])




# Model Building --------------------------------------------------
## Try out different models with CV and tuning parameters
train.predict <- train.predict %>% select(-one_of("Id"))

predictors <- SalePrice ~ .

##Linear Regression
trcntrl.lm <- trainControl(
  method = "repeatedcv",
  number = 10,
  repeats = 3,
  savePredictions = TRUE
)

set.seed(1234)
baselineLM <- train(
  predictors,
  data = train.predict,
  method = "lm",
  trControl = trcntrl,
  tuneLength = 5,
  preProcess = c("center", "scale")
)

#Ridge

trcntrl.ridge <- trainControl(method = "cv",
                              number = 10)
# Set seq of lambda to test
lambdaGrid <- expand.grid(lambda = 10 ^ seq(10, -2, length = 100))
set.seed(1234)
ridge <- train(
  predictors,
  data = train.predict,
  method = 'ridge',
  trControl = trcntrl.ridge,
  tuneGrid = lambdaGrid
)

#lasso

predictors <- SalePrice ~ .



tuneGrid = expand.grid(.alpha = 1,
                       .lambda = seq(0, 100, by = 0.1))

set.seed(1234)
lasso <- train(
  predictors,
  data = train.predict,
  method = 'glmnet',
  trControl = trcntrl.ridge,
  tuneGrid = tuneGrid
)

#GBM
predictors <- SalePrice ~ .


gbmGrid <-  expand.grid(
  interaction.depth = c(1, 5, 9),
  n.trees = (1:30) * 50,
  shrinkage = 0.1,
  n.minobsinnode = 20
)
set.seed(1234)
gbm <- train(
  predictors,
  data = train.predict,
  method = 'gbm',
  trControl = trcntrl.ridge,
  tuneGrid = gbmGrid
)


#RandomForest



gbm.model <- predict(gbm, test.predict)
gbm.model <- as.data.frame(gbm.model)
gbm.model <-
  data.frame(id = test$Id, SalePrice = gbm.model$gbm.model)
write.csv(gbm.model, "gbm.model.csv", row.names = FALSE)
