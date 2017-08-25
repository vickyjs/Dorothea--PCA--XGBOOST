
## Dorothea dataset is a sparse binary matrix dataset,this function creates a matrix
## with 800 rows and 100000 col with all 0's, then adds 1 to the index mentioned in th
## orignal data.

## %%%% Change the number of nrow to 350 when loading validation data.%%%%
to.Matrix <- function(filename){
  emp1.mat <- matrix(0, nrow = 800, ncol = 100000)
  for (i in (1:nrow(emp1.mat))){
    sample.one <- readLines(filename)
    test.1 <- strsplit(sample.one[i], " ")
    test.1 <- as.numeric(unlist(test.1))
    for (j in 1:length(test.1)){
      c <- test.1[j]
      emp1.mat[i,c] <- 1
    }
    
  }
  return(emp1.mat)
}
# Give your path to the dataset
train_data <- to.Matrix("C:/Users/Vignesh/Desktop/Backups/All-in-One 15-03-2017/Important Ebook/dorothea_train.csv")
train_labels <- read.csv("C:/Users/Vignesh/Desktop/Backups/All-in-One 15-03-2017/Important Ebook/dorothea_train.labels.csv", header = F)

library(caret)
# Detecting columns(Features) with near variance, worthless
near.zero.variance <- nearZeroVar(train_data, saveMetrics = TRUE)
print(paste("Range", range(near.zero.variance$percentUnique)))
## "Range 0.125" "Range 0.25"

#Removing features with range less than 0.25, very less variance not useful for prediction
train_data <- train_data[,c(as.numeric(rownames(near.zero.variance[near.zero.variance$percentUnique > 0.125,])))]
dim(train_data)
##  800 88119
nzv.index <- as.numeric(rownames(near.zero.variance[near.zero.variance$percentUnique > 0.125,]))

## Function to do Cross Validation on the train set and evaluate the performance of
## the XGBOOST on CVed dataset.
EvaluateAUC <- function(dfEvaluate){
  CVs <- 5
  cvDivider <- floor(nrow(dfEvaluate) / (CVs +1))
  indexCount <- 1
  outComeName <- c("cluster")
  predictors <- names(dfEvaluate)[!names(dfEvaluate) %in% outComeName]
  lsERR <- c()
  lsAUC <- c()
  lsCom <- c()
  for (cv in seq(1:CVs)){
    print(paste("cv", cv))
    datasetIndex <- c((cv * cvDivider): (cv * cvDivider +cvDivider))
    dataTest <- dfEvaluate[datasetIndex,]
    dataTrain <- dfEvaluate[-datasetIndex,]
    bst <- xgboost(data= as.matrix(dataTrain[,predictors]),
                   label = dataTrain[,outComeName],
                   max.depth = 6, eta = 1, verbose = 0,
                   nround = 5, nthread = 4,
                   objective = "reg:linear")
    predictions <- predict(bst, as.matrix(dataTest[,predictors]))
    bin.pred <- ifelse(predictions< 0, -1,1)
    com.err <- length(bin.pred[bin.pred != dataTest[,outComeName]])
    err <- rmse(dataTest[,outComeName], predictions)
    auc <- auc(dataTest[,outComeName], predictions)
    lsERR <- c(lsERR, err)
    lsAUC <- c(lsAUC, auc)
    lsCom <- c(lsCom, com.err)
    gc()
  }
  print(paste("mean Error", mean(lsERR)))
  print(paste("mean AUC", mean(lsAUC)))
  print(paste("mismatch precentage", as.character((mean(lsCom)/length(predictions))*100)))
  print(paste("Accuracy",as.character(100-(mean(lsCom)/length(predictions))*100)))
}

## Lets try to find who our dataset with 8000 features perform to compare it with
## PCA model later.
dfEvaluvate <- cbind(as.data.frame(train_data), cluster = train_labels$V1)

library(xgboost)
library(Metrics)

EvaluateAUC(dfEvaluvate)
   #"mean Error 0.61257337463671"
   #"mean AUC 0.79609606928532"
# Not bad.. 

pca.train <- prcomp(dfEvaluvate[-88120])
s.dv <- pca.train$sdev
pr_var <- s.dv^2

prop_var <- pr_var/sum(pr_var)
prop_var[1:10]
#[1] 0.011065221 0.010183311 0.009122824 0.006288180 0.005696662 0.004998560 0.004674104
#[8] 0.004094805 0.003962705 0.003930076



plot(prop_var[1:100], xlab = "principle component",
     ylab = "prop_var",
     type = "b")
## App. 30 variables explain 90% of variance.

dfComponent <- predict(pca.train, newdata = dfEvaluvate[-88120])
## Testing with 10 Principle components
dfComponent.1 <- as.data.frame(dfComponent[,1:10])
pca.Evaluvate <- cbind(as.data.frame(dfComponent.1), cluster = train_labels$V1)

EvaluateAUC(pca.Evaluvate)
#"mean Error 0.616902458905324"
#"mean AUC 0.767661667609706"


## Testing with 20 Principle Components
dfComponent.1 <- as.data.frame(dfComponent[,1:20])
pca.Evaluvate <- cbind(as.data.frame(dfComponent.1), cluster = train_labels$V1)
EvaluateAUC(pca.Evaluvate)
#"mean Error 0.54157377483409"
#"mean AUC 0.798040127633325"
#"mismatch precentage 7.61194029850746"
#"Accuracy 92.3880597014925"

## Testing with 30 principle Components
dfComponent.1 <- as.data.frame(dfComponent[,1:30])
pca.Evaluvate <- cbind(as.data.frame(dfComponent.1), cluster = train_labels$V1)
EvaluateAUC(pca.Evaluvate)
#"mean Error 0.532920331506397"
# "mean AUC 0.801054683631662"
# "mismatch precentage 8.50746268656716"
# "Accuracy 91.4925373134328"


# Model performs exceptional good. Building a final model using all training data
# and 30 components
final.pca.train <- dfComponent[,1:30]
xg.model <- xgboost(data= as.matrix(final.pca.train),
               label = train_labels$V1,
               max.depth = 6, eta = 1, verbose = 0,
               nround = 5, nthread = 4,
               objective = "reg:linear")


## Now lets try our model with Validation data set

Valid_data <- to.Matrix("C:/Users/Vignesh/Desktop/Backups/All-in-One 15-03-2017/Important Ebook/dorothea_valid.csv")
Valid_data <- Valid_data[,c(nzv.index) ]
Valid_lable <- read.csv("C:/Users/Vignesh/Desktop/Backups/All-in-One 15-03-2017/Important Ebook/dorothea_valid.labels.csv", header = F)
Valid.pc <- predict(pca.train, newdata = as.data.frame(Valid_data))
final.pca.valid <- Valid.pc[,1:20]


predictions.final <- predict(xg.model, newdata = as.matrix(final.pca.valid))
bin.pred <- ifelse(predictions.final < 0, -1,1)
com.err <- length(bin.pred[bin.pred != Valid_lable$V1])
err <- rmse(Valid_lable$V1, predictions.final)
auc <- auc(Valid_lable$V1, predictions.final)

print(paste("mean AUC", as.character(auc)))
print(paste("mismatch precentage", as.character((com.err/length(predictions.final))*100)))
print(paste("Accuracy",as.character(100-((com.err/length(predictions.final))*100))))


#"mean AUC 0.877326880119136"
#"mismatch precentage 7.71428571428571"
#"Accuracy 92.2857142857143"
