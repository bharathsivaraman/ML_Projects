setwd("C:\\Users\\sivarbh\\Documents\\COE")
library(dplyr)
library(ggplot2)
library(AppliedPredictiveModeling)
library(e1071)
#install.packages("pbkrtest")
library(caret)
library(corrplot)
library(ggcorrplot)
library(Hmisc)
library(missForest)
library(dummies)
library(rpart)
#AppliedPredictiveModeling::scriptLocation()
data.train <-
  read.csv("C:\\Users\\sivarbh\\Documents\\ML_Projects\\Train_Mart.csv")

data.test <-
  read.csv("C:\\Users\\sivarbh\\Documents\\ML_Projects\\Test_Mart.csv")

##Data Pre Processing##

#combine test and train

data.test$Item_Outlet_Sales <- 1

data <- rbind(data.train, data.test)


#data imputation using HMISC package#

summary(data)

#impute with mean
data$Item_Weight <- with(data, impute(Item_Weight, median))

#check for null
data.null <- data[!complete.cases(data), ]
summary(data.null)

#data clean up


table(data$Outlet_Size, data$Outlet_Type)
#levels(data$Outlet_Size)[1] <- "Other"

data <-
  data %>%
  mutate(
    Item_Fat_Content = ifelse(
      as.character(Item_Fat_Content) %in% c("LF", "low fat"),
      "Low Fat",
      ifelse (
        as.character(Item_Fat_Content) %in% c("reg", "Regular"),
        "Regular",
        as.character(Item_Fat_Content)
      )
      
    ),
    Item_Visibility = ifelse(
      Item_Visibility == 0,
      median(Item_Visibility),
      data$Item_Visibility
    ),
    Outlet_Size = ifelse(
      as.character(Outlet_Size) == "",
      "Other",
      as.character(Outlet_Size)
    )
  )
data$Item_Fat_Content <- with(data, as.factor(Item_Fat_Content))
summary(data)


#remove dependent variable and identifiers

#hotencoding factors into numericals


#
# num <- sapply(data.new, is.numeric)
# data.num <- data.new[num == TRUE]
#
#
data.new <-
  data %>% select(-c(Item_Outlet_Sales, Item_Identifier, Outlet_Identifier))
data.new[sapply(data.new, is.factor)] <-
  lapply(data.new[sapply(data.new, is.factor)], as.character)


data.new <-
  dummy.data.frame(
    data.new,
    names = c(
      "Item_Fat_Content",
      "Item_Type",
      "Outlet_Size",
      "Outlet_Location_Type",
      "Outlet_Type"
    )
  )

#
# data.standardize <- preProcess(data.num, method = "BoxCox")
# data.model <- predict(data.standardize, data.num)

# PCA

pca.train <- data.new[1:nrow(data.train), ]
pca.test <- data.new[-(1:nrow(data.test)), ]



pca = prcomp(pca.train, scale = T)
names(pca)

pca$rotation

x<-pca$rotation

 # variance

std_dev <- pca$sdev
pr_var = (pca$sdev) ^ 2

# % of variance
prop_varex = pr_var / sum(pr_var)

# Plot


plot(prop_varex,
     xlab = "Principal Component",
     ylab = "Proportion of Variance Explained",
     type = "b")

plot(cumsum(prop_varex))

train.data<-data.frame(Item_Outlet_Sales=data.train$Item_Outlet_Sales,pca$x)

  
rpart.model<-rpart(Item_Outlet_Sales~.,data=train.data,method="anova")

fancyRpartPlot(rpart.model,main=paste('RPART:'),sub=cName)
plotcp(rpat.model)
