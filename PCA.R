setwd("C:\\Users\\sivarbh\\Documents\\COE")
library(dplyr)
library(ggplot2)
library(AppliedPredictiveModeling)
library(e1071)
#install.packages("pbkrtest")
library(caret)
library(corrplot)
library(ggcorrplot)


#AppliedPredictiveModeling::scriptLocation()
data <-
  read.csv("C:\\Users\\sivarbh\\Documents\\COE\\movie_metadata.csv")

##Data Pre Processing##

num <- sapply(data, is.numeric)
data.num <- data[num == TRUE]

#Seperate NA's for analysis
data.clean <- data.num[complete.cases(data.num), ]
data.null <- data.num[!complete.cases(data.num), ]

data.standardize <- preProcess(data.clean, method = "BoxCox")
data.trans <- predict(data.standardize, data.clean)


# Spliting training and testing dataset

# Spliting training and testing dataset
index = sample( 1:nrow( data.trans ), nrow( data.trans ) * 0.6, replace = FALSE ) 

trainset = data.trans[ index, ]
test = data.trans[ -index, ]


data.train<-trainset%>%select(-imdb_score)
data.test<-test
M <- cor(data.train)

# corrplot(
#   M,
#   method = "color",
#   outline = T,
#   addgrid.col = "darkgray",
#   order = "hclust",
#   addrect = 3,
#  # rect.lwd = 5,
#  # cl.pos = "b",
#   tl.col = "indianred4",
#   addCoef.col = "white",
#   number.digits = 2,
#   number.cex = 0.75
# )
ggcorrplot(
  M,
  hc.order = TRUE,
  type = "lower",
  outline.col = "white",
  ggtheme = ggplot2::theme_gray,
  colors = c("#6D9EC1", "white", "#E46726"),
  lab = T
)



