#load required libraries function

ipak <- function(pkg){
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
    "pROC","ROCR")

ipak(packages)

#load required libraries function


lapply(packages, require, character.only = TRUE)



train <- read.csv("train.csv")
test <- read.csv("test.csv")




#EDA ---------------

data.complete <- rbind(train[-13], test)

summary(data.complete)

data.null <- data.complete[!complete.cases(data.complete),]



##NA Imputation-------

Imputedata <- 0
if (Imputedata == 1)
{
  data.complete <- rbind(train[-13], test)
  
  data.complete$Gender <- as.character(data.complete$Gender)
  data.complete$Married <- as.character(data.complete$Married)
  data.complete$Dependents <-
    as.character(data.complete$Dependents)
  data.complete$Self_Employed <-
    as.character(data.complete$Self_Employed)
  
  
  data.complete <-
    data.complete %>% mutate(
      Gender = ifelse(
        Gender == '' & ApplicantIncome > 5256,
        "Male",
        ifelse(Gender == '' &
                 ApplicantIncome < 5256, "Female", Gender)
      ),
      Married = ifelse(
        Married == '' & Dependents != '',
        "Yes",
        ifelse(Married == '' &
                 Dependents == '', "No" , Married)
      ),
      Dependents = ifelse(
        Dependents == '' & ApplicantIncome > 7000,
        "3+",
        ifelse(
          Dependents == '' &
            ApplicantIncome < 5000,
          "0",
          ifelse(
            Dependents == '' &
              (ApplicantIncome < 7000  &
                 ApplicantIncome > 5000),
            "2",
            Dependents
          )
          
        )
      ),
      Self_Employed = ifelse(
        Self_Employed == '' & ApplicantIncome > 6900,
        "Yes",
        ifelse(
          Self_Employed == '' &
            ApplicantIncome < 6900,
          "No",
          Self_Employed
        )
        
        
      ),
      
      Credit_History = ifelse(
        is.na(Credit_History) &
          ApplicantIncome > 5200,
        1,
        ifelse(
          is.na(Credit_History) &
            ApplicantIncome < 5200,
          0,
          Credit_History
        )
      ),
      Dependents = ifelse(Dependents == '3+', 3, Dependents)
    )
  
  
  
  data.complete$Gender <- as.factor(data.complete$Gender)
  data.complete$Married <- as.factor(data.complete$Married)
  data.complete$Dependents <- as.factor(data.complete$Dependents)
  data.complete$Self_Employed <-
    as.factor(data.complete$Self_Employed)
  data.complete$Credit_History <-
    as.factor(data.complete$Credit_History)
  data.complete$Dependents <- as.factor(data.complete$Dependents)
  
  data.complete <-
    data.complete %>% group_by(Gender, Property_Area) %>% mutate(LoanAmount =
                                                                   impute.mean(LoanAmount)) %>% ungroup()
  
  data.complete$Loan_Amount_Term[is.na(data.complete$Loan_Amount_Term) == TRUE] <-
    360
  
  data.complete$Loan_Amount_Term <-
    as.factor(data.complete$Loan_Amount_Term)
} else{
  data.complete <- rbind(train[-13], test)
  preProcValues <-
    preProcess(data.complete, method = c("knnImpute"))
  data.complete <- predict(preProcValues, data.complete)
}


summary(data.complete)


#EDAPlots------------


data.plot <- data.complete[1:nrow(train),]
data.plot$Loan_Status <- train$Loan_Status

ggplot(data.plot) + aes(ApplicantIncome, LoanAmount, color = Self_Employed) + geom_point()
ggplot(data.plot) + aes(Loan_Status, CoapplicantIncome) + geom_boxplot()
ggplot(data.plot) + aes(ApplicantIncome + CoapplicantIncome, LoanAmount, color =
                          Education) +
  geom_point() + geom_smooth(method = 'lm')
ggplot(data.plot) + aes(Self_Employed, LoanAmount) + geom_boxplot()
ggplot(data.plot) + aes(Loan_Amount_Term, color = Loan_Status) + geom_histogram(stat = "count")
ggplot(data.plot) + aes(Loan_Status, fill = Education) + geom_bar(stat =
                                                                    "count")
ggplot(data.plot) + aes(Loan_Amount_Term, ApplicantIncome) + geom_boxplot()
ggplot(data.plot) + aes(LoanAmount, ApplicantIncome) + geom_point()

table(data.plot$Credit_History, data.plot$Loan_Status)



#FeatureEngineering------------------

##TotalIncome-----
data.feature <-
  data.complete %>% mutate(totalincome = ApplicantIncome + CoapplicantIncome)
##Loan Amount to Income ratio-----------
data.feature <-
  data.feature %>% mutate(incomedependentsratio = totalincome / as.numeric(Dependents))
data.feature <-
  data.feature %>% mutate(incomeloanration = LoanAmount / totalincome)

##Monthly Payment--------

EMI <- function(df, R) {
  r <- R / (12 * 100)
  power <- '^'(1 + r, as.numeric(df$Loan_Amount_Term))
  power_denom <- '^'(1 + r, as.numeric(df$Loan_Amount_Term) - 1)
  EMI = (df$LoanAmount * 1000 * r * power / (power_denom))
}

data.feature <-
  data.feature %>% mutate(EMI = EMI(data.feature, 10.0))

##IncometoEMIratio-----

data.feature <-
  data.feature %>% mutate(IncometoEMIratio = EMI / totalincome)

ggplot(data.feature) + aes(totalincome, EMI) + geom_point()
#OnehotEncoding-------------

encode<-readinteger()

if (encode==1){

dmy <- dummyVars("~.", data = data.feature[-1])
trnsf <- data.frame(predict(dmy, newdata = data.feature))
} else {trnsf<-data.feature[-1] }

#ModelBuilding--------

##ModelBuild and analysis-------------

### Split training and test data set
train.data <-
  trnsf[1:nrow(train), ]
train.data$Loan_Status <- train$Loan_Status
test.data <-
  trnsf[(1:nrow(test)), ]

### predictor selection using VarImp from RF
predictors <- Loan_Status ~
  
  Credit_History        +
  totalincome            +
  incomedependentsratio   +
  ApplicantIncome         +
  IncometoEMIratio        +
  incomeloanration

#+
# EMI                     +
# LoanAmount              +
# CoapplicantIncome

# Loan_Status ~ Credit_History.1 +
# Credit_History.0 +
# incomeloanration +
# IncometoEMIratio  +
# totalincome     +
# incomedependentsratio  +
# EMI                 +
# LoanAmount

#incomeloanration + totalincome + incomedependentsratio + Credit_History.1 + Credit_History.0 + LoanAmount
### Random Forest

rf.trcntrl <- trainControl(
  method = "LOOCV",
  number = 10,
  repeats = 3,
  summaryFunction = twoClassSummary,
  classProbs = TRUE,
search = "grid", savePredictions = TRUE
  )

tunegrid <- expand.grid(.mtry = c(1:15))
model.rf <-
  model.predict(rf.trcntrl, predictors, "rf", train.data, metric = 'ROC')

### Rpart
model.rpart <-
  model.predict("LOOCV", predictors, "rpart", train.data, metric =
                  "Kappa")

### GBM
model.gbm <-
  model.predict("LOOCV", predictors, "gbm", train.data, metric = 'Accuracy')

### XGboost
model.xgb <-
  model.predict("repeatedcv", predictors, "xgbTree", train.data, metric = 'Accuracy')

### LogisticRegression
model.glm <-
  model.predict("LOOCVs", predictors, "glm", train.data, metric = 'Accuracy')



###ModelEvaluation------------


#confusion Matrix


confusionMatrix(model.rf$pred$pred,
                reference = model.rf$pred$obs,
                positive = "Y")

#Calibration plot

calY <- calibration(obs ~ Y, data = model.rf$pred, class = "Y")
xyplot(calY)

calN <- calibration(obs ~ N, data = model.rf$pred, class = "N")
xyplot(calN)

#ROC

selectedIndices <- model.rf$pred$mtry == 1



g <-
  ggplot(model.rf$pred[selectedIndices,], aes(m = Y, d = factor(obs, levels = c("N", "Y")))) +
  geom_roc(n.cuts = 0) +
  coord_equal() +
  style_roc()

g + annotate("text",
             x = 0.75,
             y = 0.25,
             label = paste("AUC =", round((calc_auc(
               g
             ))$AUC, 4)))

resamps <-
  resamples(list(GBM = model.gbm, RPART = model.rpart, RF = model.rf))


##Prediction and submission ----------------
rf.predict <- predict(model.rf, newdata = test.data)
write.output(rf.predict, "rf.sub.csv")


rpart.predict <- predict(model.rf, newdata = test.data)
write.output(rf.predict, "rpart.sub.csv")

gbm.predict <- predict(model.gbm, newdata = test.data)
write.output(rf.predict, "gbm.sub.csv")


#Functions------------


impute.mean <-
  function(x)
    replace(x, is.na(x), mean(x, na.rm = TRUE))


impute.median <-
  function(x)
    replace(x, is.na(x), median(x, na.rm = TRUE))




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



write.output <- function (modelframe, filename)
{
  output <-  data.frame(Loan_ID = test$Loan_ID,
                        Loan_Status = modelframe)
  
  write.csv(output, filename, row.names = FALSE)
}

readinteger <- function()
{
  n <- readline(prompt = "Enter an integer: ")
  return(as.integer(n))
}
print(readinteger())


 
