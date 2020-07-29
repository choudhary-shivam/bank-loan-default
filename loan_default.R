# remove all the objects stored
rm(list = ls())

# Importing require Libraries
library(ggplot2)
# Importing library for spliting train and test data
library(caTools)
# Importing some useful functions
#library(dplyr)
# Importing library to built correleation graph
library(corrgram)
# Importing library for confusion matrix, correlation and feature importance
library(mlbench)
library(caret)
# library for LDA
library(MASS)
# Importing Library to evaluate PR and ROC
library(PRROC)
# Importing Library for KNN
library(class)
# Importing Library for Naiye Bayes and SVM
library(e1071)
# Importing Library for Decision Tree and pca
library(C50)
library(rpart)
# Importing Library for Random Forest
library(randomForest)

# set current working directory
setwd("E:/project")

# Load csv data in R
dataset = read.csv("bank-loan.csv", header=T)

# Getting the number of variables and obervation in the data-set
dim(dataset)
nrow(dataset)
ncol(dataset)

# looking at unique values data
sapply(dataset, function(x) length(unique(x)))

################# Missing Value Analysis ####################

table(is.na(dataset))
colSums(is.na(dataset))

# checking null values in Train and Test data set
missing_value = data.frame(apply(dataset,2,function(x){sum(is.na(x))}))
missing_value$Columns = row.names(missing_value)
row.names(missing_value) = NULL
missing_value = missing_value[,c(2,1)]
names(missing_value)[2] =  "count"
missing_value$missing_percent = (missing_value$count/nrow(dataset)) * 100

# Imputing missing values
for(i in colnames(dataset[,-9])){
  #dataset[,i][is.na(dataset[,i])] = mean(dataset[,i], na.rm = T)
  dataset[,i][is.na(dataset[,i])] = median(dataset[,i], na.rm = T)
}

#########################################################################

# separate data in 2 different data frame to predict default value on the bases of our model
train = dataset[which(!is.na(dataset$default)),]
test = dataset[which(is.na(dataset$default)),]
rm(dataset)

# Getting the column names of data-set
colnames(train)
colnames(test)

# Convert data to required types
train$default = as.factor(train$default)
train$ed = as.factor(train$ed)
test$default = as.factor(test$default)
test$ed = as.factor(test$ed)

# checking class
class(train$default)

# Getting the structure of the datasets
str(train)
str(test)

# Getting summary for each columns in data-set
summary(train)
summary(test)

# # checking and removing duplicate rows
# train = distinct(train)

# Encoding the target feature as factor
train$default = factor(train$default, levels = c(0, 1))

# # Encoding categorical data
# dataset$Country = factor(dataset$Country,
#                          levels = c('France', 'Spain', 'Germany'),
#                          labels = c(1, 2, 3))
# dataset$Purchased = factor(dataset$Purchased,
#                            levels = c('No', 'Yes'),
#                            labels = c(0, 1))

# looking at the target variable
table(train$default)

# Saving Numerical and categorical variables in different list
numeric_index = sapply(train,is.numeric) #selecting only numeric
numeric_data = train[,numeric_index]
cat_data = train[,!numeric_index]
num_var = colnames(numeric_data)
cat_var = colnames(cat_data)
rm(numeric_index, numeric_data, cat_data)

# function to plot graph together
grid_plot = function() 
{
  gridExtra::grid.arrange(graph1, graph2, graph3, graph4, 
                          graph5, graph6, graph7, ncol=2)
}

# Plotting target column for visualization of train
ggplot(data = train) + 
  geom_bar(mapping = aes(x = default),
           alpha = 0.5, width=0.4, position = "identity", fill = "steelblue")

# Plotting target column for for ed category
ggplot(data = train) + 
  geom_bar(mapping = aes(x = default, fill = ed),
           alpha = 0.5, width=0.5, position = "identity")
# position = "dodge" places overlapping objects directly beside one another

# Plotting target column for visualization of test
ggplot(data = test) + 
  geom_bar(mapping = aes(x = ed, fill = ed), alpha = 0.7, width=0.5, stat="count")

# This type of plots are used when you need to find a relation between two variables
ggplot(data = train) + 
  geom_point(mapping = aes(x = age, y = income, color = default))

# This type of plots are used when you need to find a relation between two variables
ggplot(data = train) +
  geom_bar(mapping = aes(x = age, y = income, fill = default), stat = "identity", alpha = 0.6)

# plotting distribution of numerical variables of train Dataset
for(i in 1:length(num_var)){
  assign(paste0("graph",i), ggplot(data = train) +
           geom_density(mapping = aes_string(x = num_var[i]),
                        fill="#69b3a2", color="#e9ecef", alpha=0.6))
}
grid_plot()

# plotting distribution of numerical variables of test Dataset
for(i in 1:length(num_var)){
  assign(paste0("graph",i), ggplot(data = test) +
           geom_density(mapping = aes_string(x = num_var[i]),
                        fill="#f2b646", color="#e9ecef", alpha=0.6))
}
grid_plot()

# Plotting Density Distribution of train per class
for(i in 1:length(num_var)){
  assign(paste0("graph",i), ggplot(train) + 
           geom_density(mapping = aes_string(x = num_var[i], fill = "default"),
                        kernel='gaussian', alpha=0.4))
}
grid_plot()

# Boxplot of train dataset
for(i in 1:length(num_var)){
  assign(paste0("graph",i), ggplot(data = train) +
           geom_boxplot(mapping = aes_string(x = "default", y = num_var[i]),
                        outlier.colour="red", fill = "grey"))
}
grid_plot()
rm(list = ls(pattern = "graph"))
rm(grid_plot)

####################### Outlier Analysis ################################

#loop to remove rows with outliers
# for(i in num_var){
#   val1 = train[,i][train[,i] %in% boxplot.stats(train[,i])$out]
#   val2 = test[,i][test[,i] %in% boxplot.stats(test[,i])$out]
#   train = train[which(!train[,i] %in% val1),]
#   test = test[which(!test[,i] %in% val2),]
# }
#Data size will become small to make model more accurate, so not using it

# Replace all outliers in Train and Test with NA
for(i in num_var){
  val1 = train[,i][train[,i] %in% boxplot.stats(train[,i])$out]
  val2 = test[,i][test[,i] %in% boxplot.stats(test[,i])$out]
  train[,i][train[,i] %in% val1] = NA
  test[,i][test[,i] %in% val2] = NA
}

# Imputing missing values with Median
for(i in num_var){
  train[,i][is.na(train[,i])] = median(train[,i], na.rm = T)
  test[,i][is.na(test[,i])] = median(test[,i], na.rm = T)
}
rm(val1, val2)
colSums(is.na(train))

#########################Feature Selection##################################

#Correlation Plot 
corrgram(train[,num_var], order = F,
         upper.panel=panel.pie, text.panel=panel.txt, main = "Correlation Plot")

correlation_matrix = cor(train[,num_var])
print(correlation_matrix)
highly_correlated = findCorrelation(correlation_matrix, cutoff = 0.75)
print(highly_correlated)

# Chi-squared Test of Independence
factor_index = sapply(train,is.factor)
factor_data = train[,factor_index]

for (i in 1:length(factor_data))
{
  print(names(factor_data)[i])
  print(chisq.test(table(factor_data$default,factor_data[,i])))
}
rm(factor_data, factor_index)

set.seed(12345)

############################ Feature Scaling ##########################

# # except target column all other column has been scaled using standardization
# train[,-9] = scale(train[,-9])
# test[,-9] = scale(test[,-9])

# #Normalisation
# for(i in num_var){
#   train[,i] = (train[,i] - min(train[,i]))/(max(train[,i] - min(train[,i])))
#   test[,i] = (test[,i] - min(test[,i]))/(max(test[,i] - min(test[,i])))
# }

#Standardisation
for(i in num_var){
  train[,i] = (train[,i] - mean(train[,i]))/sd(train[,i])
  test[,i] = (test[,i] - mean(test[,i]))/sd(test[,i])
}
rm(i)

################################# Sampling ################################

# Splitting the train data set into the Training set and Test set
split = sample.split(train$default, SplitRatio = 0.75)
training_set = subset(train, split == TRUE)
test_set = subset(train, split == FALSE)
rm(split)

################################### Model Development #################################

##############Logistic Regression###############

# Fitting Logistic Regression to the Training set
logit_classifier = glm(formula = default ~ ., family = "binomial", data = training_set)

# Predicting the Test set results
prob_pred = predict(logit_classifier, type = 'response', newdata = test_set[,-4])
logit_pred = ifelse(prob_pred > 0.5, 1, 0)

# Making the Confusion Matrix
cm_logit = table(test_set[, 4], logit_pred)
confusionMatrix(cm_logit, positive = '1', mode = "everything")

# saving predicted and actual values to different data frame to find ROC and PR.
scores <- data.frame(logit_pred, test_set$default)

# evaluating PR and ROC for Logistic classifier

pr <- pr.curve(scores.class0= scores[scores$test_set.default=="1",]$logit_pred,
               scores.class1= scores[scores$test_set.default=="0",]$logit_pred,
               curve=T)
plot(pr)

roc <- roc.curve(scores.class0= scores[scores$test_set.default=="1",]$logit_pred,
                 scores.class1= scores[scores$test_set.default=="0",]$logit_pred,
                 curve=T)
plot(roc)

################ Naive Bayes #####################

# Fitting Naive Bayes to the Training set
nb_classifier = naiveBayes(x = training_set[,-9],
                           y = training_set$default)

# Predicting the Test set results
nb_pred = predict(nb_classifier, newdata = test_set[,-9])

# Making the Confusion Matrix
cm_nb = table(test_set[, 9], nb_pred)
confusionMatrix(cm_nb, positive = '1', mode = "everything")

#statical way
mean(nb_pred == test_set$default)

# saving predicted and actual values to different data frame to find ROC and PR.
scores <- data.frame(nb_pred, test_set$default)

# evaluating PR and ROC for Naive Bayes classifier

pr <- pr.curve(scores.class0= scores[scores$test_set.default=="1",]$nb_pred,
               scores.class1= scores[scores$test_set.default=="0",]$nb_pred,
               curve=T)
plot(pr)

roc <- roc.curve(scores.class0= scores[scores$test_set.default=="1",]$nb_pred,
                 scores.class1= scores[scores$test_set.default=="0",]$nb_pred,
                 curve=T)
plot(roc)

##################### Random Forest ##################

# Fitting Random Forest Classification to the Training set
rf_classifier = randomForest(default ~ ., training_set, importance = TRUE, ntree = 625)

importance(rf_classifier)

# Predicting the Test set results
rf_pred = predict(rf_classifier, newdata = test_set[-9])

# Making the Confusion Matrix
cm_rf = table(test_set[, 9], rf_pred)
confusionMatrix(cm_rf, positive = '1', mode = "everything")

# saving predicted and actual values to different data frame to find ROC and PR.
scores <- data.frame(rf_pred, test_set$default)

# evaluating PR and ROC for Random forest classifier

pr <- pr.curve(scores.class0= scores[scores$test_set.default=="1",]$rf_pred,
               scores.class1= scores[scores$test_set.default=="0",]$rf_pred,
               curve=T)
plot(pr)

roc <- roc.curve(scores.class0= scores[scores$test_set.default=="1",]$rf_pred,
                 scores.class1= scores[scores$test_set.default=="0",]$rf_pred,
                 curve=T)
plot(roc)

##################### KNN ##################

# Fitting K-NN to the Training set and Predicting the Test set results
knn_pred = knn(train = training_set[, -9], test = test_set[, -9], cl = training_set[, 9],
               k = 7, prob = TRUE)

# Making the Confusion Matrix
cm_knn = table(test_set[, 9], knn_pred)
confusionMatrix(cm_knn, positive = '1', mode = "everything")

# saving predicted and actual values to different data frame to find ROC and PR.
scores <- data.frame(knn_pred, test_set$default)

# evaluating PR and ROC for Random forest classifier

pr <- pr.curve(scores.class0= scores[scores$test_set.default=="1",]$knn_pred,
               scores.class1= scores[scores$test_set.default=="0",]$knn_pred,
               curve=T)
plot(pr)

roc <- roc.curve(scores.class0= scores[scores$test_set.default=="1",]$knn_pred,
                 scores.class1= scores[scores$test_set.default=="0",]$knn_pred,
                 curve=T)
plot(roc)

##################### Decision Tree C5.0 ##################

#Develop Model on training data
C50_classifier = C5.0(default ~., training_set, trials = 100, rules = TRUE)

#Lets predict for test cases
C50_pred = predict(C50_classifier, test_set[,-9])

# Making the Confusion Matrix
cm_C50 = table(test_set[, 9], C50_pred)
confusionMatrix(cm_C50, positive = '1', mode = "everything")

# saving predicted and actual values to different data frame to find ROC and PR.
scores <- data.frame(C50_pred, test_set$default)

# evaluating PR and ROC for Random forest classifier

pr <- pr.curve(scores.class0= scores[scores$test_set.default=="1",]$C50_pred,
               scores.class1= scores[scores$test_set.default=="0",]$C50_pred,
               curve=T)
plot(pr)

roc <- roc.curve(scores.class0= scores[scores$test_set.default=="1",]$C50_pred,
                 scores.class1= scores[scores$test_set.default=="0",]$C50_pred,
                 curve=T)
plot(roc)

##################### Decision Tree ##################

# Fitting Decision Tree Classification to the Training set
dt_classifier = rpart(formula = default ~ .,
                      data = training_set)

# Predicting the Test set results
dt_pred = predict(dt_classifier, newdata = test_set[-9], type = 'class')

# Making the Confusion Matrix
cm_dt = table(test_set[, 9], dt_pred)
confusionMatrix(cm_dt, positive = '1', mode = "everything")

# saving predicted and actual values to different data frame to find ROC and PR.
scores <- data.frame(dt_pred, test_set$default)

# evaluating PR and ROC for Random forest classifier

pr <- pr.curve(scores.class0= scores[scores$test_set.default=="1",]$dt_pred,
               scores.class1= scores[scores$test_set.default=="0",]$dt_pred,
               curve=T)
plot(pr)

roc <- roc.curve(scores.class0= scores[scores$test_set.default=="1",]$dt_pred,
                 scores.class1= scores[scores$test_set.default=="0",]$dt_pred,
                 curve=T)
plot(roc)

##################### SVM ##################

# Fitting SVM to the Training set
svm_classifier = svm(formula = default ~ .,
                     data = training_set,
                     type = 'C-classification',
                     kernel = 'linear')

# Predicting the Test set results
svm_pred = predict(svm_classifier, newdata = test_set[-9])

# Making the Confusion Matrix
cm_svm = table(test_set[, 9], svm_pred)
confusionMatrix(cm_svm, positive = '1', mode = "everything")

# saving predicted and actual values to different data frame to find ROC and PR.
scores <- data.frame(svm_pred, test_set$default)

# evaluating PR and ROC for Random forest classifier

pr <- pr.curve(scores.class0= scores[scores$test_set.default=="1",]$svm_pred,
               scores.class1= scores[scores$test_set.default=="0",]$svm_pred,
               curve=T)
plot(pr)

roc <- roc.curve(scores.class0= scores[scores$test_set.default=="1",]$svm_pred,
                 scores.class1= scores[scores$test_set.default=="0",]$svm_pred,
                 curve=T)
plot(roc)
rm(pr, roc, scores)
rm(list = ls(pattern = "cm"))
rm(list = ls(pattern = "pred"))

######################## Result ##########################

# predicting test value and saving back to file
test_pred = predict(nb_classifier, newdata = test[,-9])
test$default = test_pred

rm(training_set, test_set)
rm(list = ls(pattern = "classifier"))

# Writing a csv (output)
#write.csv(test, "test_output_R.csv", row.names = F)

# Plotting target column of Test Dataset for visualization
ggplot(data = test) + 
  geom_bar(mapping = aes(x = default, fill = default), width = 0.5, alpha = 0.5)

table(test$default)

#merging data back to make one file
loan_data = rbind(train, test)


