library(caret)
library(randomForest)
library(Hmisc)
library(rpart)
library(ipred)
library(rattle)
set.seed(831)
    trainurl="https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
    testurl="https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
#download and populate testing and training dataframes
    
    df_train<-read.csv(trainurl, header=T)
    df_test<-read.csv(testurl, header=T)
    
# clean up training - remove low varaiance and predominantly NA columns
    #train_upd<-df_train[,-nearZeroVar(df_train)]
    zVarcols<-colnames(df_train)[nearZeroVar(df_train)]
    naCols<-colnames(df_train)[colSums(is.na(df_train)) > nrow(df_train)*.5]
    time_cols<-colnames(df_train)[grepl("*time*", colnames(df_train))]
    rm_cols<-c(zVarcols,naCols,time_cols,"X","num_window")
    train_upd<-df_train[,!names(df_train) %in% rm_cols]
    
# create training and testing partitions of training data
    inTrain<- createDataPartition(y=train_upd$classe, p=.6, list =F)
    Part_train<-train_upd[inTrain,]
    Part_test<-train_upd[-inTrain,]
#Model Cross-Validation setting
    cv_set<-trainControl(method="cv", number=3, verboseIter=F)
    
    
#model builds
    mod_rf<-train(classe~., method="rf", data=Part_train, trControl=cv_set)
    mod_gbm<-train(classe~., method="gbm", data=Part_train, trControl=cv_set, verbose=FALSE)
    mod_trbag<-train(classe~., method="treebag", data=Part_train, trConrol=cv_set)
    mod_rpart<-train(classe~., method="rpart", data=Part_train, trConrol=cv_set)
#generate predictions    
    pred_rf<-predict(mod_rf, Part_test)
    pred_gbm<-predict(mod_gbm, Part_test)
    pred_rpart<-predict(mod_rpart, Part_test)
#variable importance
    confusionMatrix(pred_rf, Part_test$classe)
    confusionMatrix(pred_gbm, Part_test$classe)
    confusionMatrix(pred_rpart, Part_test$classe)
    
    varImp(mod_rf)
    varImp(mod_gbm)
    plot(varImp(mod_rf), ylab.mar=14)
    plot(varImp(mod_gbm))
#add predictions to frame    
    pred_fm<-data.frame(Part_test, rf=predict(mod_rf, Part_test), gbm=predict(mod_gbm, Part_test),pred_rpart=predict(mod_rpart, Part_test))
    
    pred_fm<-mutate(pred_fm, Rf_predictCorrect=pred_fm$classe==pred_fm$rf)
    
    
    fancyRpartPlot(mod_rpart$finalModel)