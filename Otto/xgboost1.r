setwd("~/Downloads/MISC/kaggle/otto")
otto <- read.csv("train.csv")
#install.packages("xgboost")
library(xgboost)
library(data.table)
library(methods)
library(magrittr)
str(otto)
otto$target <- gsub("Class_","",otto$target)
otto$target <- as.numeric(otto$target)
head(otto$target, 10)
x <- otto$target - 1
train <- matrix(as.numeric(unlist(otto)),nrow=nrow(otto))
train[1:6,1:5]
train <- train[,-1]
ncol(train)
train <- train[,-94]
dim(train)
class <- max(x)+1
param <- list("objective" = "multi:softprob",
              "eval_metric" = "mlogloss",
              "num_class" = class,
              "nthread" = 8)
nround = 2000
bst.cv <- xgb.cv(param = param, data = train, label = x,nfold = 5, nrounds = nround, eta =0.03)
bst.cv
minimum <- min(bst.cv$test.mlogloss.mean)
index <- which.min(bst.cv$test.mlogloss.mean)
bst <- xgboost(param=param, data=train, label=x, 
               nrounds=index, verbose=0, eta = 0.03) 
test <- read.csv("test.csv")
str(test)
test1 <- matrix(as.numeric(unlist(test)),nrow=nrow(test))
test1 <- test1[,-1]
pred <- predict(bst,test1)
head(pred,10)
pred <- matrix(pred,9,length(pred)/9)
pred[1:6,1:9]
pred <- t(pred)
pred = format(pred, digits=2,scientific=F) # shrink the size of submission
pred = data.frame(1:nrow(pred),pred)
names(pred) = c('id', paste0('Class_',1:9))
write.csv(pred,file='submission2.csv', quote=FALSE,row.names=FALSE)
