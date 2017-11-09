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
  c("dplyr",
    "caret",
    "ggplot2",
    "lazyeval",
    "plotROC",
    "lift",
    "RANN",
    "pROC",
    "ROCR")

ipak(packages)

train <- read.csv("train.csv")
test <- read.csv("test.csv")
test$SalePrice <- 1
combined <- rbind(train, test)


# EDA - Feature Engineering --------------------------------------------------

#Check for NA and remove columns where more than 40% are NA

threshold <- nrow(combined) * .4

colnames <- "id"

for (i in 1:ncol(combined)) {
  x <- combined[!complete.cases(combined[i]), ][i]
  
  if (nrow(x) > threshold) {
    colnames <- rbind(colnames, colnames(x))
  }
  
}
colnames <- colnames[-1]

combined <- combined %>% select(-one_of(colnames))





##SalePrice is right skewed and taking log removes the skewness
ggplot(train) + aes(SalePrice) + geom_histogram(col = "white") + theme_light()
ggplot(train) + aes(log(SalePrice)) + geom_histogram(col = "white", fill =
                                                       "firebrick") + theme_light()
train$MSSubClass <- as.factor(train$MSSubClass)

continuous.variable <- train[, lapply(train, is.numeric) == TRUE]
continuous.variable <- continuous.variable[-1]

factor.variables <- train[, lapply(train, is.factor) == TRUE]
factor.variables$SalePrice <- train$SalePrice
factor.variables$SalePrice <- train$SalePrice
##Neighbourhood
ggplot(factor.variables) + aes(Neighborhood, SalePrice) + geom_boxplot()


#Baseline Model----
##Baseline model built on averaging the price across neighbourhood

impute.mean <-
  function(x)
    replace(x, is.na(x), mean(x, na.rm = TRUE))

#continuous.variable$YearBuilt<-as.factor(continuous.variable$YearBuilt)
continuous.variable <-
  continuous.variable %>% group_by(YearBuilt) %>% mutate(SalePrice =
                                                           impute.mean(SalePrice)) %>% ungroup()

predictors <- SalePrice ~ .

trcntrl <- trainControl(
  method = "repeatedcv",
  number = 10,
  repeats = 3,
  summaryFunction = twoClassSummary,
  classProbs = TRUE,
  savePredictions = TRUE
)

baselineLM <- train(
  predictors,
  A =
    data = continuous.variable,
  method = "lm",
  trControl = trcntrl,
  tuneLength = 5 ,
  metric = "Accuracy",
  preProcess = c("center", "scale")
)
