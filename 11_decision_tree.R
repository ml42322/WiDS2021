# Load relevant libraries
library(tidyverse)
library(magrittr)
library(rpart)
library(caret)

########################################
# SET UP FOR MODELING
########################################
# Load cleaned data frame for modeling
train_rus <- read.csv("train_rus.csv")
train_ros <- read.csv("train_ros.csv")
test <- read.csv("test_data.csv")
# train_rus <- train_rus[, -which(names(train_rus) %in% c('encounter_id', 'hospital_id', 'icu_id'))]

# Ensure our dependent variable is factored 
train_rus$diabetes_mellitus %<>% 
  as.factor()

train_ros$diabetes_mellitus %<>%
  as.factor()

test$diabetes_mellitus %<>% 
  as.factor()

# Ensure all our categorical variables are factored
cols = colnames(train_rus)[1:58]
train_rus[cols] <- lapply(train_rus[cols], factor)
train_ros[cols] <- lapply(train_ros[cols], factor)
test[cols] <- lapply(test[cols], factor)


##############################
### DECISION TREE - TRAIN RUS
##############################

# Model is trained on the entire data sett
decision_tree <- rpart(diabetes_mellitus ~ .,
                       data = train_rus[, -which(names(train_rus) %in% c('encounter_id', 
                                                                         'hospital_id', 
                                                                         'icu_id', 
                                                                         'readmission_status'))],
                       method = "class",
                       control = rpart.control(c=0.0001))

save(decision_tree, file = "/Users/preeyamody/Desktop/widsdatathon2021/decision_tree.Rdata")

# Print the covariates (in order of importance)
decision_tree$variable.importance

# Print prediction error on the tetst sett
y_hat <- predict(decision_tree, newdata = test[, -which(names(test %in% c('encounter_id', 
                                                                          'hospital_id', 
                                                                          'icu_id',
                                                                          'readmission_status')))], type = c("class"))
table(y_hat, test$diabetes_mellitus)
test_error <- mean(y_hat != test$diabetes_mellitus)
test_error

confusionMatrix(data = y_hat,
                reference = test$diabetes_mellitus)

# False positive = 1 - specificity
# False negatiive = 1 - sensitivity

##############################
### DECISION TREE - TRAIN ROS
##############################

# Model is trained on the entire data sett
decision_tree2 <- rpart(diabetes_mellitus ~ .,
                       data = train_ros[, -which(names(train_ros) %in% c('encounter_id', 
                                                                         'hospital_id', 
                                                                         'icu_id', 
                                                                         'readmission_status'))],
                       method = "class",
                       control = rpart.control(c=0.0001))

save(decision_tree2, file = "/Users/preeyamody/Desktop/widsdatathon2021/decision_tree.Rdata")

# Print the covariates (in order of importance)
decision_tree2$variable.importance

# Print prediction error on the tetst sett
y_hat2 <- predict(decision_tree2, newdata = test[, -which(names(test %in% c('encounter_id', 
                                                                          'hospital_id', 
                                                                          'icu_id',
                                                                          'readmission_status')))], type = c("class"))
table(y_hat2, test$diabetes_mellitus)
test_error2 <- mean(y_hat2 != test$diabetes_mellitus)
test_error2

confusionMatrix(data = y_hat2,
                reference = test$diabetes_mellitus)
