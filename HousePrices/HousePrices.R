#HousePrices comptetion on Kaggle#
##Predict Sales Prices in Boston##




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
str(train)


# EDA - Factor Variables --------------------------------------------------
##SalePrice is right skewed and taking log removes the skewness
ggplot(train)+aes(SalePrice)+geom_histogram(col="white")+theme_light()
ggplot(train)+aes(log(SalePrice))+geom_histogram(col="white")+theme_light()

continuous.variable<-train[, lapply(train, is.numeric) == TRUE]

factor.variables <- train[, lapply(train, is.factor) == TRUE]
factor.variables$SalePrice <- train$SalePrice
factor.variables$SalePrice <- train$SalePrice
##Neighbourhood
ggplot(factor.variables) + aes(Neighborhood, SalePrice) + geom_boxplot()

##YearBuild
ggplot(train) + aes(YearBuilt, SalePrice) + geom_point()


x <-
  factor.variables %>% group_by(Neighborhood) %>% summarise(avg.salesprice = mean(SalePrice)) %>%
  mutate(rnk = rank((avg.salesprice)))

x.top2 <- x %>% select(1) %>% top_n(2)
x.top2$Neighborhood <- as.character(x.top2$Neighborhood)
x.bottom2 <- x %>% select(1) %>% top_n(-2)
x.bottom2$Neighborhood <- as.character(x.bottom2$Neighborhood)

neighbourhood <-
  train %>% filter(
    as.character(Neighborhood) %in% x.top2$Neighborhood |
      Neighborhood %in% x.bottom2$Neighborhood
  )

neighbourhood$Neighborhood <-
  as.character(neighbourhood$Neighborhood)

table(neighbourhood$Neighborhood, neighbourhood$YearBuilt)
table(neighbourhood$Neighborhood, neighbourhood$Utilities)

