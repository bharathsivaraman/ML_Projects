#HousePrices comptetion on Kaggle#
##Predict Sales Prices in Iowa##

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




model.predict <-
  function(trcntrl,
           predictors,
           modeltype,
           traindata,
           metric)  {
    set.seed(1234)
    
    modelout <-    train(
      predictors,
      data = traindata,
      method = modeltype,
      trControl = trcntrl,
      tuneLength = 5 ,
      metric = metric,
      preProcess = c("center", "scale"),
      tuneGrid = tunegrid
    )
    
    return(modelout)
    
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

#convert into factors
combined$YearBuilt <- as.factor(combined$YearBuilt)
combined$YearRemodAdd <- as.factor(combined$YearRemodAdd)
combined$GarageYrBlt <- as.factor(combined$GarageYrBlt)
combined$YrSold <- as.factor(combined$YrSold)
combined$MSSubClass <- as.factor(combined$MSSubClass)
combined$MoSold <- as.factor(combined$MoSold)

 
#get factor columns
factor.variables <- combined[, lapply(combined, is.factor) == TRUE]

factor.variables$SalePrice <- combined$SalePrice

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
    factor.variables %>% select(i, ncol(factor.variables)) %>% group_by_(colname.group) %>% summarize(AvgSalePrice =
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
factor.variables <- new[-38]
factor.variables <- lapply(factor.variables, as.factor)
factor.variables <- as.data.frame(factor.variables)

## Removing additional near zero variance predictors

nearzero <- nearZeroVar(factor.variables)
nearzerocolnames <- NA
for (i in (1:length(nearzero))) {
  x <- colnames(factor.variables)[nearzero[[i]]]
  nearzerocolnames <- rbind(nearzerocolnames, x)
}
nearzerocolnames<-nearzerocolnames[-1]
factor.variables <- factor.variables%>%select(-one_of(nearzerocolnames))

##remove year column to remove time series

factor.variables<-factor.variables%>%select(-contains("year"),-contains("yr"),-contains("Mo"))

  
####-- Split train and test records and perform onehotencoding on train data set
  ### OnehotEncoding ----------------------------------------------------------

dmy <- dummyVars(" ~ .", data = factor.variables)
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


table(continuous.variables$MiscVal) #--Remove variable
table(continuous.variables$PoolArea)#--Remove variable

table(continuous.variables$WoodDeckSF)#-Remove variable
table(continuous.variables$MasVnrArea)
table(continuous.variables$OverallCond)
table(continuous.variables$OverallQual)


#
# ##convert as ordered factor overall condition and overall qual
#
# continuous.variables$OverallCond <-
#   factor(
#     continuous.variables$OverallCond,
#     levels = c(1, 2, 3, 4, 5, 6, 7, 8, 9),
#     ordered = TRUE
#   )
#
#
# continuous.variables$OverallQual <-
#   factor(
#     continuous.variables$OverallQual,
#     levels = c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
#     ordered = TRUE
#   )

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

#PriceperSqft

continuous.variables<-continuous.variables%>%mutate(pricesqft=SalePrice/LotArea)



#Remove these variables
#MiscVal,PoolArea,BsmtFinSF2,LowQualFinSF,ScreenPorch,X3SsnPorch,EnclosedPorch,OpenPorchSF,X1stFlrSF,X2ndFlrSF,BsmtUnfSF,BsmtFinSF1



#Apply log to the following variables
#LotArea,#LotFrontage,#SalePrice,#GarageArea,GrLivArea,BsmtUnfSF,TotalBsmtSF,WoodDeckSF

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
      "X2ndFlrSF","BsmtFinSF1","BsmtUnfSF",
      "WoodDeckSF",
    )
  )

#
# scale.colname <-
#   c(
#     "LotArea",
#     "LotFrontage",
#     "GarageArea",
#     "GrLivArea",
#     "BsmtUnfSF",
#     "TotalBsmtSF",
#     "WoodDeckSF"
#   )
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


M <- cor(corframe)
corrplot(M, method = "circle")

summary(continuous.variables)

##Combine continuous and factor variables

##with one hot encoding

encode <- readinteger()

if (encode == 1) {
  combined.clean <- cbind(trsf, continuous.variables)
} else {
  combined.clean <- cbind(factor.variables, continuous.variables)
}
combined.clean$Id <- combined$Id
combined.NA <- combined.clean[!complete.cases(combined.clean),]
summary(combined.clean)


train.predict <- combined.clean %>% filter(Id %in% train[["Id"]])
test.predict <- combined.clean %>% filter(Id %in% test[["Id"]])




# Baseline Models--------------------------------------------------
##Baseline model built on averaging the price across neighbourhood

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
#Model Evaluation-----




##Output------
ridge.model <- predict(ridge, test.predict)
ridge.model <- as.data.frame(ridge.model)
ridge.model <-
  data.frame(id = test$Id, SalePrice = ridge.model$ridge.model)
write.csv(ridge.model, "ridge.model.csv", row.names = FALSE)


lasso.model <- predict(lasso, test.predict)
lasso.model <- as.data.frame(lasso.model)
lasso.model <-
  data.frame(id = test$Id, SalePrice = lasso.model$lasso.model)
write.csv(lasso.model, "lasso.model.csv", row.names = FALSE)
