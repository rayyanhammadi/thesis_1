library(readxl)
library(randomForest)
library(ggplot2)
set.seed(2023)
raw_data = read_excel('BDD_SummerSchool_BENOIT1.xlsx',sheet="raw_data")
raw_data$y<- raw_data$`USA (Acc_Slow)`
data <- raw_data[,!(names(raw_data) %in% c("dates","USA (Acc_Slow)"))]

covariates <- raw_data[,!(names(raw_data) %in% c("dates","USA (Acc_Slow)","y"))]
target <- data.frame(date = raw_data$dates,ACC_SLO = raw_data$`USA (Acc_Slow)`)
ans <- c()
date_split = 204
step_ahead = 15
predictions <- data.frame(date=raw_data$dates,label=0,preds=0)
predictions <- predictions[date_split:nrow(predictions),]

normalize <- function(x, na.rm = TRUE) {
  return((x- min(x)) /(max(x)-min(x)))
}

logit_predict <- function(date_split,step_ahead){
  pred <- numeric(nrow(data))
  for (id_split in(date_split:nrow(covariates))){
    print(id_split)
    data_train <- data[1:(id_split-step_ahead+1),]
    x_test <- covariates[id_split,]
    logistic <- glm(y~.,data=data_train,family="binomial")
    pred[id_split] <- predict(logistic,newdata=x_test,type='response')
    predictions[id_split-date_split+1,2] <- pred[id_split]
    predictions[id_split-date_split+1,3] <- ifelse(pred[id_split]>0.5,1,0)
  }
  return(predictions)
}

RF_predict <- function(date_split,step_ahead){
  p<-0
  for (id_split in(date_split:nrow(covariates))){
    y_train <- target[1:(id_split-step_ahead+1),2]
    x_train <- covariates[1:(id_split-step_ahead+1),]
    x_test <- covariates[id_split,]
    print(id_split)
    rf <- randomForest(x=x_train,y=y_train, n_tree=2000,random_state=0)
    p <- predict(rf,newdata=x_test,type='response')
    predictions[id_split-date_split+1,2] <- p
    predictions[id_split-date_split+1,3] = ifelse(p>0.5,1,0)
  }
  return(predictions)
}


a=logit_predict(date_split,step_ahead)
plot(a$date,a$label,type="l",xlab="dates",ylab="predictions",col="green",main="RandomForest")
polygon(a$date,data$y[date_split:nrow(data)],col="lightblue",density=100)
abline(h=.5,col="red")
legend("topright", c( "Acceleration", "Slowdown"),
       col = c("lightblue","white"), lty = c(1, 1, 1), lwd = 2, box.lty = 0, bg = "gray95")
