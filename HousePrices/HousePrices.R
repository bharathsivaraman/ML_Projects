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

# Function------------

impute.median <-
  function(x)
    replace(x, is.na(x), median(x, na.rm = TRUE))


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

##Remove columns with more than 40%NA
combined <- combined %>% select(-one_of(colnames))

#convert into factors
combined$YearBuilt <- as.factor(combined$YearBuilt)
combined$YearRemodAdd <- as.factor(combined$YearRemodAdd)
combined$GarageYrBlt <- as.factor(combined$GarageYrBlt)
combined$YrSold <- as.factor(combined$YrSold)
combined$MSSubClass <- as.factor(combined$MSSubClass)

#get factor columns
factor.variables <- combined[, lapply(combined, is.factor) == TRUE]

factor.variables$SalePrice <- combined$SalePrice

##-------- Check freq and dis of every level in each column
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
data.levels <- data.levels[-1, ]
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

####----- Merge levels where distribution is less than 5%

new <- factor.variables
new[] <-
  lapply(factor.variables, function(x)
    data.levels$newlevel[match(x, data.levels$Var1)])
factor.variables <- new[-38]
factor.variables <- lapply(factor.variables, as.factor)
factor.variables <- as.data.frame(factor.variables)

##-----------Check for NA's in continuous variables and impute
continuous.variables <-
  combined[, lapply(combined, is.numeric) == TRUE]
continuous.variables <- continuous.variables[,-1]

for (i in 1:ncol(continuous.variables))
{
  continuous.variables[[i]] <-
    impute.median(continuous.variables[[i]])
}


combined.clean <- cbind(factor.variables, continuous.variables)
combined.clean$Id <- combined$Id
combined.NA <- combined.clean[!complete.cases(combined.clean), ]
summary(combined.clean)


train.predict <- combined.clean %>% filter(Id %in% train[["Id"]])
test.predict <- combined.clean %>% filter(Id %in% test[["Id"]])

##SalePrice is right skewed and taking log removes the skewness
ggplot(train.predict) + aes(SalePrice) + geom_histogram(col = "white") + theme_light()
ggplot(train.predict) + aes(log(SalePrice)) + geom_histogram(col = "white", fill = "firebrick") + theme_light()

train.predict$SalePrice <- log(train.predict$SalePrice)


# Baseline Model--------------------------------------------------
##Baseline model built on averaging the price across neighbourhood

train.predict <- train.predict %>% select(-one_of("Id"))

predictors <- SalePrice ~ .

trcntrl <- trainControl(
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

baseline <- predict(baselineLM, test.predict)

