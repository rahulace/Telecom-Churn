library(gridExtra)
library(grid)
library(ggplot2)
library(lattice)
library(caret)
library(corrplot)
library(plyr)
library(pROC)
library(rpart)
library(kernlab)
library(randomForest)
library(gbm)


#Loading Data
Data <- read.csv("WA_Fn-UseC_-Telco-Customer-Churn.csv", stringsAsFactors = FALSE, na.strings = c("NA", ""))

#show some lines of data
head(Data, n=5L)

Data <- Data[complete.cases(Data), ]

# MIn and Max value of tenure
paste("The Minimum value is: ",min(Data$tenure)," and the Maximum value is: ",max(Data$tenure))

#Grouping tenure into intervals

CreateGrp <- function(tn){
  if (tn >= 0 & tn <= 12){
    return('0-12')
  }else if(tn > 12 & tn <= 24){
    return('12-24')
  }else if (tn > 24 & tn <= 48){
    return('24-48')
  }else if (tn > 48 & tn <=60){
    return('48-60')
  }else if (tn > 60){
    return('> 60')
  }
}

# apply the Group function to the tenure column
Data$GrpTenure <- sapply(Data$tenure,CreateGrp)

# set as factor the new column
Data$GrpTenure <- as.factor(Data$GrpTenure)

#Remove irrelvent columns
Data <- Data[,-which(names(Data) %in% c("customerID","tenure"))]
head(Data, n=5L)

#Exploring numeric data
numeric.var <- sapply(Data, is.numeric)
corr.matrix <- cor(Data[,numeric.var])
corrplot(corr.matrix, main="\n\nCorrelation Plot for Numerical Variables", method = "number")

# check the frequency for each group bin
table(Data$GrpTenure)
barplot(table(Data$GrpTenure))

#Changing values to readable data
Data$SeniorCitizen <- as.factor(
  mapvalues(Data$SeniorCitizen,
            from=c("0","1"),
            to=c("No", "Yes"))
)


# my plots will be very similar lets simplify using a function for it
createplot <- function(dst, column, name) {
  plt <- ggplot(dst, aes(x=column, fill=(Churn))) + 
    ggtitle(name) + 
    xlab(name) +
    ylab("Percentage")  +
    geom_bar(aes(y = 100*(..count..)/sum(..count..)), width = 0.7) + 
    theme_minimal() +
    theme(legend.position="none", axis.text.x = element_text(angle = 45, hjust = 1))
  return(plt)
}


# Plot 1 by gender 
p1 <- createplot(Data, Data$gender, "Gender")                      
# plot 2 by Senior Citizen
p2 <- createplot(Data, Data$SeniorCitizen, "Senior Citizen")
# plot 3 by Partner
p3 <- createplot(Data, Data$Partner, "Partner")
# plot 4 by Dependents
p4 <- createplot(Data, Data$Dependents, "Dependents")
# plot 5 by Phone Service
p5 <- createplot(Data, Data$PhoneService, "Phone Service")
# plot 6 by Multiple Lines
p6 <- createplot(Data, Data$MultipleLines, "Multiple Lines")
# plot 7 by Internet Service
p7 <- createplot(Data, Data$InternetService, "Internet Service")
# plot 8 by Online Security
p8 <- createplot(Data, Data$OnlineSecurity, "Online Security")

# Plot grid
grid.arrange(p1, p2, p3, p4, p5, p6, p7, p8, ncol=4)



# Plot 1 by OnlineBackup 
p9 <- createplot(Data, Data$OnlineBackup, "Online Backup")                      
# plot 2 by DeviceProtection
p10 <- createplot(Data, Data$DeviceProtection, "Device Protection")
# plot 3 by TechSupport
p11 <- createplot(Data, Data$TechSupport, "Tech Support")
# plot 4 by StreamingTV
p12 <- createplot(Data, Data$StreamingTV, "Streaming TV")
# plot 5 by StreamingMovies
p13 <- createplot(Data, Data$StreamingMovies, "Streaming Movies")
# plot 6 by PaperlessBilling
p14 <- createplot(Data, Data$PaperlessBilling, "Paperless Billing")
# plot 7 by PaymentMethod
p15 <- createplot(Data, Data$PaymentMethod, "Payment Method")
# plot 8 by GrpTenure
p16 <- createplot(Data, Data$GrpTenure, "Grp. Tenure")

# Plot grid
grid.arrange(p9, p10, p11, p12, p13, p14, p15, p16, ncol=4)


###################MODELING######################################

# check column type
sapply(Data, typeof)


# The missing data percentage by variable (exclude Survived)
sapply(Data[,-c(2)], function(x) round((sum(is.na(x))/length(x)*100),2))

# convert to factor
Data$Churn  <- factor(Data$Churn)


# Setting up 10-fold cross validation
control <- trainControl(method="cv", number=10)
metric <- "Accuracy"

# Splitting the dataset into train and test
Data_TrainTest <- Data[, c('Churn','SeniorCitizen','Partner','Dependents','GrpTenure','PhoneService',
                           'MultipleLines','InternetService','OnlineSecurity','OnlineBackup','DeviceProtection',
                           'TechSupport','StreamingTV','StreamingMovies','Contract','PaperlessBilling',
                           'PaymentMethod','MonthlyCharges')]
idxSplit <- createDataPartition(Data$Churn, p = 0.75, list=FALSE)
Train_data <- Data_TrainTest[idxSplit,]
Test_data <- Data_TrainTest[-idxSplit,]

#Normalizing numeric data for KNN
trainX <- Train_data[,names(Train_data) != "Churn"]
preProcValues <- preProcess(x = trainX,method = c("center", "scale"))
preProcValues

# logistic regression
set.seed(1234)
fit.glm <- train(Churn ~ ., data=Train_data, method="glm", metric=metric, trControl=control)

# Dtree
set.seed(1234)
fit.cart <- train(Churn ~ ., data=Train_data, method="rpart", metric=metric, trControl=control)

# kNN
set.seed(1234)
fit.knn <- train(Churn ~ ., data=Train_data, method="knn", 
                 metric=metric, trControl=control, preProcess = c("center","scale"), tuneLength = 5)

# SVM
set.seed(1234)
fit.svm <- train(Churn ~ ., data=Train_data, method="svmRadial", metric=metric, trControl=control)

# Random Forest
set.seed(1234)
fit.rf <- train(Churn ~ ., data=Train_data, method="rf", metric=metric, trControl=control)

# Gradient Boost Machine (GBM)
set.seed(1234)
fit.gbm <- train(Churn ~ ., data=Train_data, method="gbm", 
                 metric=metric, trControl=control, verbose=FALSE)


# summarize accuracy of models
results <- resamples(list(
  glm=fit.glm, 
  cart=fit.cart, 
  knn=fit.knn, 
  svm=fit.svm, 
  rf=fit.rf,
  gbm=fit.gbm
))

#summary of result and comparing accuracy of models
dotplot(results)

#creating fuction to calculate accuracy
AccCalc <- function(TestFit, name) {
  # prediction 
  TestModelClean <- Test_data
  TestModelClean$Churn <- NA
  predictedval <- predict(TestFit, newdata=TestModelClean)
  
  # summarize results with confusion matrix
  cm <- confusionMatrix(predictedval, Test_data$Churn)
  
  # calculate accuracy of the model
  Accuracy<-round(cm$overall[1],2)
  acc <- as.data.frame(Accuracy)
  
  roc_obj <- roc(Test_data$Churn, as.numeric(predictedval))
  acc$Auc <- auc(roc_obj)
  
  acc$FitName <- name
  return(acc)
}

accAll <- AccCalc(fit.glm, "glm")
accAll <- rbind(accAll, AccCalc(fit.cart, "cart"))
accAll <- rbind(accAll, AccCalc(fit.knn, "knn"))
accAll <- rbind(accAll, AccCalc(fit.svm, "svm"))
accAll <- rbind(accAll, AccCalc(fit.rf, "rf"))
accAll <- rbind(accAll, AccCalc(fit.gbm, "gbm"))
rownames(accAll) <- c()
arrange(accAll,desc(Accuracy))
