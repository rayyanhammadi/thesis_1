library(readxl)
library(randomForest)
library(ggplot2)
set.seed(2023)
raw_data = read_excel('BDD_SummerSchool_BENOIT1.xlsx',sheet="raw_data")
raw_data$y<- raw_data$`USA (Acc_Slow)`
data <- raw_data[,!(names(raw_data) %in% c("dates","USA (Acc_Slow)"))]

covariates <- raw_data[,!(names(raw_data) %in% c("dates","USA (Acc_Slow)","y"))]
covariates_both <-  covariates[,c("var51","var7","var41","var22","spread105","var34","var53"
                                       ,"var55","var26","var3","var30","var42","spread13m","var31",
                                       "var19","var1","var15","spi","var11","var35","spread102",
                                       "var54","var49","gold","var43","var12","var32")]
cov_col_names <- colnames(covariates)
target <- data.frame(date = raw_data$dates,ACC_SLO = raw_data$`USA (Acc_Slow)`)
ans <- c()
date_split = 204
step_ahead = 15
predictions <- data.frame(date=raw_data$dates,label=0,preds=0)
predictions <- predictions[date_split:nrow(predictions),]

normalize <- function(x, na.rm = TRUE) {
  return((x- min(x)) /(max(x)-min(x)))
}
norm_covariates <- normalize(covariates)
norm_covariates_both <- normalize(covariates_both)
norm_data<-norm_covariates_both
norm_data$y<-raw_data$y


logit_predict <- function(date_split,step_ahead){
  pred <- numeric(nrow(data))
  for (id_split in(date_split:nrow(covariates_both))){
    print(id_split)
    data_train <- norm_data[1:(id_split-step_ahead+1),]
    x_test <- norm_covariates_both[id_split,]
    logistic <- glm(y~.,data=data_train,family=binomial(link="probit"))
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
plot(a$date,a$label,type="l",xlab="dates",ylab="predictions",col="red",main="Logit",lwd=2)
polygon(a$date,data$y[date_split:nrow(data)],col="lightgrey",density=100)
abline(h=.5,col="blue")
legend("topright", c( "Acceleration", "Slowdown"),
       col = c("lightgrey","white"), lty = c(1, 1, 1), lwd = 2, box.lty = 0, bg = "gray95")

b=RF_predict(date_split,step_ahead)
plot(b$date,b$label,type="l",xlab="dates",ylab="predictions",col="red",main="RF")
polygon(a$date,data$y[date_split:nrow(data)],col="lightgrey",density=100)
abline(h=.5,col="red")
legend("topright", c( "Acceleration", "Slowdown"),
       col = c("lightgrey","white"), lty = c(1, 1, 1), lwd = 2, box.lty = 0, bg = "gray95")
