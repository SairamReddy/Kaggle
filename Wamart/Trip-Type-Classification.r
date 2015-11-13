# Sairam reddy Reddipalli's Walmart Trip Type Classification Data

# Load the libraries
library(dplyr)
library(magrittr)
library(reshape2)
library(xgboost)
library(Matrix)
library(Ckmeans.1d.dp)
library(DiagrammeR)
library(caret)
library(corrplot)
library(Rtsne)
library(stats)
library(ggplot2)
library(e1071)
library(data.table)

setwd("~/Downloads/MISC/kaggle/walmart")

# read the training and test data sets and convert them into data table
train <- data.table(read.csv("train.csv", header = TRUE, stringsAsFactors = TRUE)) 
test <- data.table(read.csv("testw.csv", header = TRUE, stringsAsFactors = TRUE))

# Summary Stats
summary(train)
summary(test)

# Dependent Variable factor Levels
x <- data.frame(table(train$TripType))
x

# Find the NA values in train and test data set. We find NA values of Upc and FinelineNumber correspond.
sapply(train, function(x) sum(is.na(x))) # 4129 NA values in Upc and FinelineNumber
sapply(test, function(x) sum(is.na(x)))  # 3986 NA values in Upc and FinelineNumber


train.data <- data.frame(train.data)

cor(train.data[, -c(3, 6)]) # triptype and scan count is slightly correlated
Col <- c("VisitNumber", "Upc", "ScanCount", "FinelineNumber")

# plotting the data
# corrplot.mixed(cor(train.data[, -c(3,6)]), lower="circle", upper="color", 
#                tl.pos="lt", diag="n", order="hclust", hclust.method="complete")
ggplot(train.data, aes(x = DepartmentDescription, y = TripType)) + geom_boxplot()


# sort the trip type
sort <- data.table(TripType = sort(unique(train$TripType)))
sort$Index <- seq_along(sort$TripType) - 1

# combine train and test
dt <- data.table(rbind(train, cbind(TripType = -1, test)))
sapply(dt, function(x) sum(is.na(x)))

# addNA's to the NA values
dt$FinelineNumber <- addNA(dt$FinelineNumber)
dt$Upc <- addNA(dt$Upc)

# NULL values in DepartmentDescription
levels(dt$DepartmentDescription)
dt <- dt[, NullDescription:=ifelse (dt$DepartmentDescription == "NULL", 1, 0)]

# Number of NULL values
sum(dt$NullDescription)

dt$ReturnCount <- -dt$ScanCount
dt$ReturnCount[dt$ReturnCount < 0] <- 0
dt$ScanCount[dt$ScanCount < 0] <- 0
dt$ResultCount <- dt$ScanCount - dt$ReturnCount


item <- summarise(group_by(dt, VisitNumber),
                         TotalScan = sum(ScanCount), TotalReturn = sum(ReturnCount), TotalResult = sum(ResultCount))

 
dt.long <- melt.data.table(data = dt, measure.vars = c("ScanCount", "ReturnCount", "ResultCount"),
                           variable.name = "ItemCount")

dt.wide1 <- dcast.data.table(data = dt.long,
                             VisitNumber + TripType + Weekday ~ DepartmentDescription + ItemCount,
                             value.var = "value",
                             fun.aggregate = sum) # %>% arrange(VisitNumber)
 

wd <- model.matrix(~0 + Weekday, data = dt.wide1)

dt.wide1 <- cbind(wd, dt.wide1)
dt.wide1 <- dt.wide1[, Weekday:=NULL]

dt.wide <- dt.wide1

rm(dt.wide1)

dt.wide <- merge(dt.wide, item.counts, by = "VisitNumber")

train <- dt.wide[dt.wide$TripType != -1, ]
test <- dt.wide[dt.wide$TripType == -1, ]

train <- train[, VisitNumber := NULL] # preferred way of deleting data.table columns
test.VisitNumber <- test$VisitNumber
test <- test[, VisitNumber :=  NULL]


y <- plyr::mapvalues(train$TripType, from = outcomes$TripType, to = outcomes$Index)

train <- train[, TripType := NULL]
test <- test[, TripType := NULL]

num.class <- length(unique(y))

param <- list("objective" = "multi:softprob",    # multiclass classification 
              "num_class" = num.class,    # number of classes 
              "eval_metric" = "mlogloss",    # evaluation metric 
              "nthread" = 8,   # number of threads to be used 
              "max_depth" = 16,    # maximum depth of tree 
              "eta" = 0.3,    # step size shrinkage 
              "gamma" = 0,    # minimum loss reduction 
              "subsample" = 1,    # part of data instances to grow tree 
              "colsample_bytree" = 1,  # subsample ratio of columns when constructing each tree 
              "min_child_weight" = 12  # minimum sum of instance weight needed in a child 
)

train.matrix <- as.matrix(train)
train.matrix <- as(train.matrix, "dgCMatrix") # conversion to sparse matrix
dtrain <- xgb.DMatrix(data = train.matrix, label = y)

set.seed(1234)

cv.nround <- 50 
cv.nfold <- 3 

bst.cv <- xgb.cv(param=param, data=dtrain, 
                 nfold=cv.nfold, nrounds=cv.nround, prediction=TRUE) 

tail(bst.cv$dt)


min.error.index = which.min(bst.cv$dt[, test.mlogloss.mean]) 
min.error.index 


bst.cv$dt[min.error.index, ]


nround = min.error.index 
bst <- xgboost(param = param, data = dtrain, nrounds = nround, verbose = TRUE)

model <- xgb.dump(bst, with.stats = T)
model[1:10]


names <- dimnames(train.matrix)[[2]]


importance_matrix <- xgb.importance(names, model = bst)


xgb.plot.importance(importance_matrix[1:20,])


test.matrix <- as.matrix(test)
pred <- predict(bst, test.matrix)


pred <- matrix(pred, nrow=num.class, ncol=length(pred) / num.class)
pred <- data.frame(cbind(test.VisitNumber, t(pred)))


submit <- function(filename) {
  names(pred) <- c("VisitNumber", paste("TripType", outcomes$TripType, sep = "_")) 
  
  write.table(format(pred, scientific = FALSE), paste("./output/", filename, sep = ""), row.names = FALSE, sep = ",")
}
submit("xgboost11.csv")

time.end <- Sys.time()
time.end - time.start

set.seed(1234)

library(mlbench)
library(caret)


zero.var = nearZeroVar(train, saveMetrics=TRUE)

cols <- row.names(zero.var[zero.var$nzv == TRUE, ]) 
colNums <- match(cols, names(train))
ntrain <- select(train, -colNums)

corr <- cor(ntrain)
corr

model <- naiveBayes(ntrain, y)   
pred <- predict(model, test)  


# Decode prediction
pred <- matrix(pred, nrow=num.class, ncol=length(pred) / num.class)
pred <- data.frame(cbind(test.VisitNumber, t(pred)))
