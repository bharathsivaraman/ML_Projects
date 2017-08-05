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
M <- cor(data.trans)

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


