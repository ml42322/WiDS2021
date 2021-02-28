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


### Random forest model was ont able to hand the full training set so it was subset to 100,000 obs
# set.seed(1)
# temp <- train %>% 
#  sample_n(100000)

##############################
### RANDOM FOREST - TRAIN RUS
##############################
# Initialize model without any tuning
set.seed(1)
rf_1 <- randomForest::randomForest(diabetes_mellitus ~ .,
                       data = train_rus[, -which(names(train_rus) %in% c('encounter_id', 
                                                                         'hospital_id', 
                                                                         'icu_id', 
                                                                         'readmission_status'))])

save(rf_1, "/Users/preeyamody/Desktop/widsdatathon2021/random_forest_1.Rdata")

rf_1$importance
y_hat_rf_1 <- predict(rf_1, newdata = test, type = "class")
table(y_hat_rf_1, test$diabetes_mellitus)
rf_1_test_error <- mean(y_hat_rf_1 != test$diabetes_mellitus)
rf_1_test_error
confusionMatrix(data = y_hat_rf_1,
                reference = test$diabetes_mellitus)

# Tune the number of trees to fit the minimum out of bag error
set.seed(1)
rf_2 <- randomForest(diabetes_mellitus ~ .,
                     data = train_rus[, -which(names(train_rus) %in% c('encounter_id', 
                                                                       'hospital_id', 
                                                                       'icu_id', 
                                                                       'readmission_status'))],
                     ntree = which.min(rf_1$err.rate[, "OOB"]))

# save(rf_2, "/Users/preeyamody/Desktop/widsdatathon2021/random_forest_2.Rdata")

rf_2$importance
y_hat_rf_2 <- predict(rf_2, newdata = test, type = "class")
table(y_hat_rf_2, test$diabetes_mellitus)
rf_2_test_error <- mean(y_hat_rf_2 != test$diabetes_mellitus)
rf_2_test_error

confusionMatrix(data = y_hat_rf_2,
                reference = test$diabetes_mellitus)


##############################
### RANDOM FOREST - TRAIN ROS
##############################

# Initialize model without any tuning
set.seed(1)
rf_3 <- randomForest::randomForest(diabetes_mellitus ~ .,
                                   data = train_ros[, -which(names(train_ros) %in% c('encounter_id', 
                                                                                     'hospital_id', 
                                                                                     'icu_id', 
                                                                                     'readmission_status'))])

save(rf_3, "/Users/preeyamody/Desktop/widsdatathon2021/random_forest_3.Rdata")

rf_3$importance
y_hat_rf_3 <- predict(rf_3, newdata = test, type = "class")
table(y_hat_rf_3, test$diabetes_mellitus)
rf_3_test_error <- mean(y_hat_rf_3 != test$diabetes_mellitus)
rf_3_test_error
confusionMatrix(data = y_hat_rf_3,
                reference = test$diabetes_mellitus)

# Tune the number of trees to fit the minimum out of bag error
set.seed(1)
rf_4 <- randomForest(diabetes_mellitus ~ .,
                     data = train_ros[, -which(names(train_ros) %in% c('encounter_id', 
                                                                       'hospital_id', 
                                                                       'icu_id', 
                                                                       'readmission_status'))],
                     ntree = which.min(rf_3$err.rate[, "OOB"]))

# save(rf_4, "/Users/preeyamody/Desktop/widsdatathon2021/random_forest_2.Rdata")

rf_4$importance
y_hat_rf_4 <- predict(rf_4, newdata = test, type = "class")
table(y_hat_rf_4, test$diabetes_mellitus)
rf_4_test_error <- mean(y_hat_rf_4 != test$diabetes_mellitus)
rf_4_test_error

confusionMatrix(data = y_hat_rf_4,
                reference = test$diabetes_mellitus)