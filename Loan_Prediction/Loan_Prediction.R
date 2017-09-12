#load required libraries function


packages <- c("dplyr", "caret", "ggplot2", "lazyeval")
lapply(packages, require, character.only = TRUE)



train <- read.csv("train.csv")
test <- read.csv("test.csv")




#EDA ---------------

data.complete <- rbind(train[-13], test)

summary(data.complete)

data.null <- data.complete[!complete.cases(data.complete),]



##NA Imputation

calc.average(data.complete, "Credit_History", "ApplicantIncome")

data.complete$Gender <- as.character(data.complete$Gender)
data.complete$Married <- as.character(data.complete$Married)
data.complete$Dependents <- as.character(data.complete$Dependents)
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
      ifelse(Self_Employed == '' &
               ApplicantIncome < 6900, "No", Self_Employed)
      
      
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
    )
  )



data.complete$Gender <- as.factor(data.complete$Gender)
data.complete$Married <- as.factor(data.complete$Married)
data.complete$Dependents <- as.factor(data.complete$Dependents)
data.complete$Self_Employed <-
  as.factor(data.complete$Self_Employed)

data.complete <-
  data.complete %>% group_by(Gender, Property_Area) %>% mutate(LoanAmount =
                                                                 impute.mean(LoanAmount)) %>% ungroup()

data.complete$Loan_Amount_Term[is.na(data.complete$Loan_Amount_Term) == TRUE] <-
  360

summary(data.complete)

#
# ###EDAPlots------------
#
# factor.variables <- data.complete[1:nrow(train),][,(sapply(data.complete,is.factor))==TRUE]
#
# factor.variables$Status=train$Loan_Status
# ggplot(factor.variables)+aes(Status,Gender)+geom_boxplot()



data.train <- data.complete[1:nrow(train), ]
data.train$Loan_status <- train$Loan_Status
data.train <- data.train %>% select(-c(Loan_ID))

data.test <- data.complete[1:nrow(test), ]

#OnehotEncoding-------------

dmy<- dummyVars("~.",data=data.complete[-1])
trnsf<-data.frame(predict(dmy,newdata=data.complete))


#ModelBuilding--------


train.data <-
  trnsf[1:nrow(train), ]
train.data$Loan_Status <- train$Loan_Status
test.data <-
  trnsf[(1:nrow(test)), ]



model.rf<- model.predict("repeatedcv", "rf", train.data,metric='accuracy')
rf.predict <- predict(model.rf, newdata = test.data)
write.output(rf.predict, "rf.sub.csv")



resamples(list(GBM=model.gbm,RPART=model.rpart))


#Functions------------


calc.average <- function(df, groupvar, summarisevar) {
  df %>% group_by_(groupvar) %>% summarise_(mean.income = interp(~ mean(v), v =
                                                                   as.name(summarisevar))) %>%  ungroup()
  
}
impute.mean <-
  function(x)
    replace(x, is.na(x), mean(x, na.rm = TRUE))


impute.median <-
  function(x)
    replace(x, is.na(x), median(x, na.rm = TRUE))




model.predict <- function(cvmethod, modeltype, traindata,metric)
{
  trcntrl <- trainControl(method = cvmethod,
                          number = 10,
                          repeats = 10)
  
  set.seed(99999)
  
  modelout <-    train(
    Loan_Status ~ .,
    data = traindata,
    method = modeltype,
    trControl = trcntrl,
    tuneLength = 5 ,metric=metric )
  
  return(modelout)
  
}



write.output <- function (modelframe, filename)
{
  output <-  data.frame(
    Loan_Id = test$Loan_ID,
    Loan_Status = modelframe
  )
  
  write.csv(output, filename, row.names = FALSE)
}

