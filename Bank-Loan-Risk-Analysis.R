## libraries

library(VIM)
library(glmnet)
library(glmnetUtils)
library(caret)
library(dplyr)
library(ggplot2)
library(rpart)
library(randomForest)
library(utils)
library(pROC)

## reading training data
bank1<-read.csv("train_v3.csv")

## Checking how many attributes has misssing NA values
colMeans(is.na(bank1)) 
bank11<-bank1

########################################### Imputing missing values by median  of train data #######################################################
for(i in 1:ncol(bank1)){
  bank11[is.na(bank11[,i]), i] <- median(bank11[,i], na.rm = TRUE)
}
colMeans(is.na(bank11))

######################################### Dimensionlatiy reduction ################################################################
##  Lasso regression

bank11_scale<-preProcess(bank11,method = c("center","scale"))   ## normalizating the data for lasso regression
bank11_scale1<-predict(bank11_scale,bank11)

model2<- cv.glmnet(loss~.,data=bank11_scale1,alpha=1,nlambda=100)
plot(model2)
model2$lambda.min
coef(model2, s = "lambda.min")              ## observing the importance of the coefficients
## normalization of the data before performing lasso regression gave significant results.

#######################################################################################################################################

## selecting variables predicted from the lasso regression to the training data.
dimred<-select(bank11,f3,f5,f55,f67,f70,f80,f82,f102,f103,f111,f112,f132,f136,f145,f153,f180,f190,f198,f200,f218,f221,f222,f238,f241,f251,f270,f281,f286,f290,f293,f294,f295,f296,f314,f315,f323,f361,f373,f374,f383,f384,f398,f413,f428,f471,f479,f514,f518,f522,f526,f533,f536,f546,f556,f588,f604,f617,f620,f631,f650,f652,f655,f663,f665,f666,f673,f674,f676,f682,f704,f705,f709,f734,f740,f747,f752,f771,f775,f776,loss)

## reading test data 1
banktest<-read.csv("test_scenario_2.csv") ## Test Data 1
banktest1<-banktest

## Median imputation of missing values of the test data 1
for(i in 1:ncol(banktest)){
  banktest1[is.na(banktest1[,i]), i] <- median(banktest1[,i], na.rm = TRUE)  
}
## Selecting the same variables for the test data, variables predicted from the lasso regression.
banktest11<-select(banktest1,f3,f5,f55,f67,f70,f80,f82,f102,f103,f111,f112,f132,f136,f145,f153,f180,f190,f198,f200,f218,f221,f222,f238,f241,f251,f270,f281,f286,f290,f293,f294,f295,f296,f314,f315,f323,f361,f373,f374,f383,f384,f398,f413,f428,f471,f479,f514,f518,f522,f526,f533,f536,f546,f556,f588,f604,f617,f620,f631,f650,f652,f655,f663,f665,f666,f673,f674,f676,f682,f704,f705,f709,f734,f740,f747,f752,f771,f775,f776)

## creating new binary variable for Training classification in 0 and 1.
lossbin<-rep(0,80000)

for(i in 1:nrow(dimred)){
  if(dimred[i,80]>0){
  lossbin[i]<-1
}
}
dimred$lossbin<-lossbin
dimred$lossbin<-factor(dimred$lossbin)

######################################## Classification ############################################################

dimred1<-dimred[,-80]   ## Data that is used for clasification

## To eliminate the bias of the model as there are many  non default values
onlyloss<- dimred1 %>% filter(lossbin==0)
notloss<-dimred1 %>% filter(lossbin==1)
nrow(notloss) ## only 7379 values of defalut.
random_only_loss<-sample(1:nrow(onlyloss),nrow(notloss),replace = FALSE)
final_random_data<-onlyloss[random_only_loss,]
bias_Classification_data<-rbind(final_random_data,notloss)  ## Choosong equal number of the lossbin values to reduce smooching.

## splitting the train data into train and test to find out the model accuracy.
## classificarion partitioning into test and train data.
set.seed(2019)
cdp<-createDataPartition(bias_Classification_data$lossbin,p=0.8,list = FALSE)
rf_class_train1<-bias_Classification_data[cdp,] ## train data
rf_class_test1<-bias_Classification_data[-cdp,] ## Test data

##### RF model

rf_class<-train(lossbin~.,data = rf_class_train1,method='rf',ntree=300)
rf_class
rf_class_pred<-predict(rf_class,rf_class_test1)
table(rf_class_pred,rf_class_test1$lossbin)                  ## 64.57% accuracy
proc_result_rf<-roc(rf_class_test1$lossbin,rf_class_pred)

rf_class_pred_final<-predict(rf_class,banktest11,type="prob")  ## giving probabilities for predicting for cases 1 and 2
rf_pred<-rf_class_pred_final
rff<-as.data.frame(rf_pred)
View(rff)
## our optimal final model used for classification because of accuracy.


## calssification by svm
svm_class<-train(lossbin~.,data = rf_class_train1,method="svmLinear")
svm_class
svm_class_pred<-predict(svm_class,rf_class_test1)
table(svm_class_pred,rf_class_test1$lossbin)           ## 64.2% accuraccy

## classification by generalized
rid_class<-train(lossbin~.,data=rf_class_train1,method='glmnet',tuneGrid=expand.grid(alpha=0,lambda=c(seq(0.1, 2, by =0.1) ,  seq(2, 5, 0.5) , seq(5, 25, 1))))
rid_class
rid_class_pred<-predict(rid_class,rf_class_test1)
table(rid_class_pred,rf_class_test1$lossbin)           ## 64.47% accuracy at lambda=0.1

#rf<-train(lossbin~.,data=dimred1,method='rf',ntree=100)
#rf  ## 90.75% of the accuracy
#rf_pred<-predict(rf,banktest11,type = "prob") ## giving prob of customers who have defaulted or not defaulted

######################################### Regression ############################################################

## considering variables where loss is only greater than 0
loss_0<-filter(dimred,loss>0)
loss_00<-loss_0[,-81]

## splitting the train data into train and test to find out the model accuracy.
set.seed(2019)
cdp1<-createDataPartition(loss_00$loss,p=0.8,list = FALSE)
rf_reg_train1<-loss_00[cdp1,]
rf_reg_test1<-loss_00[-cdp1,]

################################## lasso Regression
rid_reg<-train(loss~.,data = rf_reg_train1,method='glmnet',trControl=trainControl(method = "cv",number=10),tuneGrid=expand.grid(alpha=1,lambda=c(seq(0.001, 0.5, by =0.01))),metric="Rsquared")
rid_reg
rid_reg_pred<-predict(rid_reg,rf_reg_test1)
(cor(rf_reg_test1$loss,rid_reg_pred)^2)      ## 29.68% accuracy.

rid_reg_pred_final<-predict(rid_reg,banktest11)
rf1_pred<-rid_reg_pred_final
rff1<-as.data.frame(rf1_pred)
View(rff1)
## our optimal model to find the LGD for cases 1 and 2.

################################################################ Regression RF
rf_reg<-train(loss~.,data = rf_reg_train1,method='rf',ntree=500,tuneGrid=expand.grid(.mtry=c(seq(15,40,by=5))))
rf_reg
rf_reg_pred<-predict(rf_reg,rf_reg_test1)
(cor(rf_reg_test1$loss,rf_reg_pred)^2)       ## 28.94% accuracy. 

############################################################################# XGboost
boost_reg<-train(loss~.,data = rf_reg_train1,method="blackboost")
boost_reg
boost_reg_pred<-predict(boost_reg,rf_reg_test1)
(cor(rf_reg_test1$loss,boost_reg_pred)^2)       ## 28.58% accuracy. 

######################################## Problem formulation -- one ###########################################
loss<-data.frame(la=rep(0,25471))
for(i in 1:nrow(banktest1)){
   loss$la[i] <- (rff[i,2]*banktest[i,764]*(rff1[i,1]/100))
}

profit<-data.frame(pr=rep(0,25471))
for(i in 1:nrow(banktest1)){
  profit$pr[i] <- (rff[i,1]*banktest[i,764]*5*(4.32/100))
}

case1<-banktest1
pf1<- case1 %>% mutate(profit=profit$pr,loss=loss$la) %>% mutate(totalgain=profit-loss) %>%
  arrange(desc(totalgain)) %>% mutate(cumsum=cumsum(requested_loan))

## Taking total gain cutoff 0 to approve the customer
finalpred1<-data.frame(final=rep(0,25471))
for(i in 1:nrow(banktest1)){
  if(pf1$totalgain[i]>0){
    finalpred1$final[i]<-1
  }
  else{
    finalpred1$final[i]<-0
  }
}

finalsum1<-cbind(pf1,finalpred1)
finalsumm1<- filter(finalsum1,final==1) 
nrow(finalsumm1)                        ## 25123 customers are accepted for loan.
sum(finalsumm1$requested_loan)          ## We are using upto 1.252 billion dollars of the given 1.4 billion

finalcsv1<- finalsum1 %>% arrange(X.1) %>% select(final)

############################################### Problem formulation -- 2 ###################################################################

pf2<-pf1
finalpred2<-data.frame(final=rep(0,25471))
for(i in 1:nrow(banktest1)){
  if(pf2$cumsum[i]<450000000){
    finalpred2$final[i]<-1
  }
  else{
    finalpred2$final[i]<-0
  }
}

finalsum2<-cbind(pf2,finalpred2)
finalsumm2<-filter(finalsum2,final==1)
nrow(finalsumm2)                   ## 6469 customers are approved for the loan
sum(finalsumm2$requested_loan)     ## We are using 449.999 million of given 450 million

finalcsv2<-finalsum2 %>% arrange(X.1) %>% select(final)

################################################ Problem formulation --3 ##############################################################
prop_intrest<-read.csv("test_scenario3.csv")
prop_intrest1<-prop_intrest

## Median imputation of scenario 3
for(i in 1:ncol(prop_intrest)){
  prop_intrest1[is.na(prop_intrest1[,i]),i]<-median(prop_intrest[,i],na.rm=TRUE)
}

## selecting the same variables from the lasso regression
prop_intrest2<-select(prop_intrest1,f3,f5,f55,f67,f70,f80,f82,f102,f103,f111,f112,f132,f136,f145,f153,f180,f190,f198,f200,f218,f221,f222,f238,f241,f251,f270,f281,f286,f290,f293,f294,f295,f296,f314,f315,f323,f361,f373,f374,f383,f384,f398,f413,f428,f471,f479,f514,f518,f522,f526,f533,f536,f546,f556,f588,f604,f617,f620,f631,f650,f652,f655,f663,f665,f666,f673,f674,f676,f682,f704,f705,f709,f734,f740,f747,f752,f771,f775,f776)

rf_pred_prop<-predict(rf_class,prop_intrest2,type = "prob")     ## Classification
rff_prop<-as.data.frame(rf_pred_prop)

rf1_pred_prop<-predict(rid_reg,prop_intrest2)                ## Regression prediction
rff1_prop<-as.data.frame(rf1_pred_prop)

loss_prop<-data.frame(la=rep(0,25471))
for(i in 1:nrow(prop_intrest1)){
  loss_prop$la[i] <- (rff_prop[i,2]*prop_intrest1[i,764]*(rff1_prop[i,1]/100))
}

profit_prop<-data.frame(pr=rep(0,25471))
for(i in 1:nrow(prop_intrest2)){
  profit_prop$pr[i] <- (rff_prop[i,1]*prop_intrest1[i,764]*5*(prop_intrest1$Proposed_Intrest_Rate[i]/100))
}

pf3<- prop_intrest1 %>% mutate(profit=profit_prop$pr,loss=loss_prop$la) %>% mutate(totalgain=profit-loss) %>%
  arrange(desc(totalgain)) %>% mutate(cumsum=cumsum(requested_loan))

## Taking total gain cutoff 0 to approve the customer
finalpred3<-data.frame(final=rep(0,25471))
for(i in 1:nrow(prop_intrest1)){
  if((pf3$totalgain[i])>0){
    finalpred3$final[i]<-1
  }
  else{
    finalpred3$final[i]<-0
  }
}

finalsum3<-cbind(pf3,finalpred3)
finalsumm3<-filter(finalsum3,final==1)
nrow(finalsumm3)                           ## 21767 customers are approved for loan
sum(finalsumm3$requested_loan)             ## Using 1.097 billion of 1.4 billion

finalcsv3<- finalsum3 %>% arrange(X.1) %>% select(final)

##############################################################################################################
write.csv(finalcsv1,"G4_S1.csv")
write.csv(finalcsv2,"G4_S2.csv")
write.csv(finalcsv3,"G4_S3.csv")
