library(dplyr)
library(tidyverse)
library(ggplot2)
library(magrittr)
library(glmnet)
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


##################################################
# Logistic regression on training set - TRAIN_RUS
##################################################
# Remove features we don't need for modeling
glm_1 <- glm(diabetes_mellitus ~ .,
             family= binomial, 
             data= train_rus[, -which(names(train_rus) %in% c('encounter_id', 
                                                              'hospital_id', 
                                                              'icu_id', 
                                                              'readmission_status'))])

summary(glm_1)
car::vif(glm_1) 
predicted_1 <- predict(glm_1, test)
predicted_1 <- ifelse(predicted_1 > 0.5, 1, 0)
predicted_1 <- predicted_1 %>% as.factor()
truth_1 <- test$diabetes_mellitus
caret::confusionMatrix(predicted_1, truth_1)

# Let's now explore what cases are being misflagged -- are they the ones with imputed values?
df_1 <- cbind(test, predicted_1)
misclass_1 <- df_1 %>% subset(diabetes_mellitus == 0 & predicted_1 == 1)
# Note that the positive class is 0

# Let's examine the signs of coefficients/interpretability

#####################################################
# Lasso Regression (without Grid Search) - TRAIN_RUS
#####################################################
# Convert data to a matrix format 
X <- model.matrix(diabetes_mellitus ~ ., 
                  train_rus[, -which(names(train_rus) %in% c('encounter_id', 
                                                             'hospital_id', 
                                                             'icu_id',
                                                             'readmission_status'))])
# store dependent variable as y
y <- train_rus$diabetes_mellitus
# model
cv.lasso <- glmnet::cv.glmnet(X, y, alpha=1, family = "binomial", type.measure = "mse", standardize = TRUE)
# check the lasso coeffcients
coef(cv.lasso, cv.lasso$lambda.min)
coef(cv.lasso, cv.lasso$lambda.1se)
# Print the optimal lambda parameters
cv.lasso$lambda.min
cv.lasso$lambda.1se
# Min value of lambda
lambda_min_lasso <- cv.lasso$lambda.min
# Best value of lambda
lambda_1se_lasso <- cv.lasso$lambda.1se
# The coefficients of all other variables have been set to zero by the algorithm!
coef(cv.lasso, s=lambda_1se_lasso)
# Lasso with the best lambda
glmod_lasso <- glmnet::glmnet(X, y, alpha=1, lambda= lambda_1se_lasso, family= "binomial")
# Lasso results (absolute values, sorted)
coefs_lasso = coef(glmod_lasso)[,1]
coefs_l = sort(abs(coefs_lasso), decreasing=T)
coefs_l
# Variable importance:
# arf_apache1, aids1, cirrhosis1, icu_type_CSICU1, gcs_unable_apache1
# Get Test Data
lasso_test <- test[, -which(names(test) %in% c('encounter_id', 'hospital_id', 'icu_id', 'readmission_status'))]
X_test_lasso <- model.matrix(diabetes_mellitus ~ ., lasso_test)
# predict class, type = "class"
lasso_prob <- predict(cv.lasso, newx = X_test_lasso, s=lambda_1se_lasso, type = "response")
# Translate probabilities to predictions
lasso_predict <- rep(0, nrow(lasso_test))
lasso_predict[lasso_prob > .5] <- 1
# Confusion Matrix 
table(pred=lasso_predict, true=lasso_test$diabetes_mellitus)
# Accuracy
mean(lasso_predict==lasso_test$diabetes_mellitus) # 78.57198

####################################################
# Ridge Regression (without Grid Search) - TRAIN_RUS
####################################################
cv.ridge <- glmnet::cv.glmnet(X, y, alpha=0, family="binomial", type.measure = "mse", standardize = TRUE)
# plot result
plot(cv.ridge, main = "Ridge penalty\n\n")
# check the ridge coefficients
coef(cv.ridge, cv.ridge$lambda.min)
coef(cv.ridge, cv.ridge$lambda.1se)
# Print the optimal lambda paramters
cv.ridge$lambda.min
cv.ridge$lambda.1se
# min value of lambda 
lambda_min_ridge <- cv.ridge$lambda.min
# best value of lambda
lambda_1se_ridge <- cv.ridge$lambda.1se
# Let's now look at the ridge with the best lambda
coef(cv.ridge, s=lambda_1se_ridge)
# Ridge with best lambda
glmod_ridge <- glmnet::glmnet(X, y, alpha=0, lambda=lambda_1se_ridge, family="binomial")
# Ridge results (absolute values, sorted)
coefs_ridge = coef(glmod_ridge)[,1]
coefs_r = sort(abs(coefs_ridge), decreasing = T)
coefs_r
# VARIABLE IMPORTANCE:
# hospital_admit_source_Other1, aids1, arf_apache1, hospital_admit_source_Observation1
# Get test data
ridge_test <- test[, -which(names(test) %in% c('encounter_id', 'hospital_id', 'icu_id', 'readmission_status'))]
X_test_ridge <- model.matrix(diabetes_mellitus ~ ., ridge_test)
# Prediction on the testing site
ridge_prob <- predict(cv.ridge, newx = X_test_ridge, s=lambda_1se_ridge, type = "response")
# Translate probabilities to predictions
ridge_predict <- rep(0, nrow(ridge_test))
ridge_predict[ridge_prob > .5] <- 1
# Confusion Matrix 
table(pred=ridge_predict, true=ridge_test$diabetes_mellitus)
# Accuracy
mean(ridge_predict==ridge_test$diabetes_mellitus) # 0.7855917
################################################


### TRAIN ROS

##################################################
# Logistic regression on training set - TRAIN_ROS
##################################################
# Remove features we don't need for modeling
glm_2 <- glm(diabetes_mellitus ~ .,
             family= binomial, 
             data= train_ros[, -which(names(train_ros) %in% c('encounter_id', 
                                                              'hospital_id', 
                                                              'icu_id', 
                                                              'readmission_status'))])

summary(glm_2)
car::vif(glm_2) 
predicted_2 <- predict(glm_2, test)
predicted_2 <- ifelse(predicted_2 > 0.5, 1, 0)
predicted_2 <- predicted_2 %>% as.factor()
truth_2 <- test$diabetes_mellitus
caret::confusionMatrix(predicted_2, truth_2)

# Let's now explore what cases are being misflagged -- are they the ones with imputed values?
df_2 <- cbind(test, predicted_2)
misclass_2 <- df_2 %>% subset(diabetes_mellitus == 0 & predicted_2 == 1)
# Note that the positive class is 0

# Let's examine the signs of coefficients/interpretability

#####################################################
# Lasso Regression (without Grid Search) - TRAIN_ROS
#####################################################
# Convert data to a matrix format 
X2 <- model.matrix(diabetes_mellitus ~ ., 
                  train_ros[, -which(names(train_ros) %in% c('encounter_id', 
                                                             'hospital_id', 
                                                             'icu_id',
                                                             'readmission_status'))])
# store dependent variable as y
y2 <- train_ros$diabetes_mellitus
# model
cv.lasso2 <- glmnet::cv.glmnet(X2, y2, alpha=1, family = "binomial", type.measure = "mse", standardize = TRUE)
# check the lasso coeffcients
coef(cv.lasso2, cv.lasso2$lambda.min)
coef(cv.lasso2, cv.lasso2$lambda.1se)
# Print the optimal lambda parameters
cv.lasso2$lambda.min
cv.lasso2$lambda.1se
# Min value of lambda
lambda_min_lasso2 <- cv.lasso2$lambda.min
# Best value of lambda
lambda_1se_lasso2 <- cv.lasso2$lambda.1se
# The coefficients of all other variables have been set to zero by the algorithm!
coef(cv.lasso2, s=lambda_1se_lasso2)
# Lasso with the best lambda
glmod_lasso2 <- glmnet::glmnet(X2, y2, alpha=1, lambda= lambda_1se_lasso2, family= "binomial")
# Lasso results (absolute values, sorted)
coefs_lasso2 = coef(glmod_lasso2)[,1]
coefs_l2 = sort(abs(coefs_lasso2), decreasing=T)
coefs_l2
# Variable importance:
# arf_apache, aids1, intercept, hospital_admit_source_Chest_Pain, gcs_unable_apache
# Get Test Data
lasso_test2 <- test[, -which(names(test) %in% c('encounter_id', 'hospital_id', 'icu_id', 'readmission_status'))]
X_test_lasso2 <- model.matrix(diabetes_mellitus ~ ., lasso_test2)
# predict class, type = "class"
lasso_prob2 <- predict(cv.lasso2, newx = X_test_lasso2, s=lambda_1se_lasso2, type = "response")
# Translate probabilities to predictions
lasso_predict2 <- rep(0, nrow(lasso_test2))
lasso_predict2[lasso_prob2 > .5] <- 1
# Confusion Matrix 
table(pred=lasso_predict2, true=lasso_test2$diabetes_mellitus)
# Accuracy
mean(lasso_predict2==lasso_test2$diabetes_mellitus) #  0.7856942

####################################################
# Ridge Regression (without Grid Search) - TRAIN_ROS
####################################################
cv.ridge2 <- glmnet::cv.glmnet(X2, y2, alpha=0, family="binomial", type.measure = "mse", standardize = TRUE)
# plot result
plot(cv.ridge2, main = "Ridge penalty\n\n")
# check the ridge coefficients
coef(cv.ridge2, cv.ridge2$lambda.min)
coef(cv.ridge2, cv.ridge2$lambda.1se)
# Print the optimal lambda paramters
cv.ridge2$lambda.min
cv.ridge2$lambda.1se
# min value of lambda 
lambda_min_ridge2 <- cv.ridge2$lambda.min
# best value of lambda
lambda_1se_ridge2 <- cv.ridge2$lambda.1se
# Let's now look at the ridge with the best lambda
coef(cv.ridge2, s=lambda_1se_ridge2)
# Ridge with best lambda
glmod_ridge2 <- glmnet::glmnet(X2, y2, alpha=0, lambda=lambda_1se_ridge2, family="binomial")
# Ridge results (absolute values, sorted)
coefs_ridge2 = coef(glmod_ridge2)[,1]
coefs_r2 = sort(abs(coefs_ridge2), decreasing = T)
coefs_r2
# VARIABLE IMPORTANCE:

# Get test data
ridge_test2 <- test[, -which(names(test) %in% c('encounter_id', 'hospital_id', 'icu_id', 'readmission_status'))]
X_test_ridge2 <- model.matrix(diabetes_mellitus ~ ., ridge_test2)
# Prediction on the testing site
ridge_prob2 <- predict(cv.ridge2, newx = X_test_ridge2, s=lambda_1se_ridge2, type = "response")
# Translate probabilities to predictions
ridge_predict2 <- rep(0, nrow(ridge_test2))
ridge_predict2[ridge_prob2 > .5] <- 1
# Confusion Matrix 
table(pred=ridge_predict2, true=ridge_test2$diabetes_mellitus)
# Accuracy
mean(ridge_predict2==ridge_test2$diabetes_mellitus) # 0.7855661
################################################








# ################################################
# # Lasso and Ridge Regression (with Grid Search)
#####################################################
# # Create features and target matrices
# X_train_new <- train_rus[, -which(names(train_rus) %in% c('encounter_id', 'hospital_id', 'icu_id'))]
# y_train_new <- train_rus$diabetes_mellitus
# 
# X_test_new <- test[, -which(names(test) %in% c('encounter_id', 'hospital_id', 'icu_id'))]
# y_test_new <- test$diabetes_mellitus
# 
# # Create and fit Lasso and Ridge obejcts
# lasso <- train(y = y_train_new,
#                x = X_train_new,
#                method= 'glmnet',
#                tuneGrid = expand.grid(alpha=1, lambda=1))
# 
# ridge <- train(y = y_train, 
#                x = X_train, 
#                mtehod= 'glmnet',
#                tuneGrid = expand.grid(alpha=0, lambda=1))
# 
# # Make the predictions
# predictions_lasso <- lasso %>% predict(X_test)
# predictions_ridge <- ridge %>% predict(X_test)
# 
# # Print R squared scores
# data.frame(
#   Lasso_R2 = R2(predictions_lasso, y_test),
#   Ridge_R2 = RMSE(predictions_ridge, y_test)
# )
# 
# # Print RMSE
# data.frame(
#   Lasso_R2 = RMSE(predictions_lasso, y_test),
#   Ridge_R2 = RMSE(predictions_ridge, y_test)
# )
# 
# # Print coefficients
# data.frame(
#   as.data.frame.matrix(coef(lasso$finalModel, lasso$bestTune$lambda)),
#   as.data.frame.matrix(coef(ridge$finalModel, ridge$bestTune$lambda))
# ) %>% 
#   rename(Lasso_coef = X1, Ridge_coef = X1.1)
# 
# # Now let's choose hte regularization paramter with the help of tuenGrid.
# # The models with the highest R-squared score will give us the best parameters.
# # Splitting training set into two paramters based on the outcome: 75% and 25%.
# 
# # parameters <- c(seq(0.1,2,by=0.1), seq(2,5,0.5), seq(5,25,1))
# paramters <- seq(0.1, 10, length=1000)
# 
# lasso <- train(y= y_train,
#                x = X_train,
#                method = 'glmnet',
#                tuneGrid = expand.grid(alpha = 1, lambda = parameters),
#                metric = "RMSE")
# 
# ridge <- train(y= y_train,
#                x = X_train,
#                method = 'glmnet',
#                tuneGrid = expand.grid(alpha = 0, lambda = parameters),
#                metric = "Rsquared")
# 
# linear <- train(y= y_train,
#                x = X_train,
#                method = 'glm',
#                family = "binomiial",
#                metric = "Rsquared")
# 
# print(paste0('Lasso best parameters: ', lasso$finalModel$lambdaOpt))
# print(paste0('Ridge best parameters: ', ridge$finalModel$lambdaOpt))
# 
# predictions_lasso <- lasso %>% predict(X_test)
# predictions_ridge <- ridge %>% predict(X_test)
# predictions_lin <- linera %>% predict(X_test)
# 
# data.frame(
#   Lasso_R2 = R2(predictions_lasso, y_test),
#   Ridge_R2 = R2(predictions_ridge, y_test),
#   Linear_R2 = R2(predictions_ridge, y_test)
#   )
# 
# data.frame(
#   Lasso_RMSE = RMSE(predictions_lasso, y_test),
#   Ridge_RMSE = RMSE(predictions_ridge, y_test),
#   Linear_RMSE = RMSE(predictions_ridge, y_test)
# )
# 
# print('Best estimator coefficients:')
# data.frame(
#   lasso = as.data.frame.matrix(coef(lasso$finalModel, lasso$finalModel$lambdaOpt)),
#   ridge = as.data.frame.matrix(coef(ridge$finalModel, ridge$finalModel$lambdaOpt)),
#   linear = (linear$finalModel$coefficients) 
# ) %>% rename(rename(lasso= X1, ridge = X1.1))
# 
# # Convert X_train to matrix for usign it with glmnet function
# X_train_m <- as.matrix(X_train)
# 
# # Build Lasso and Ridge for 200 values of lambda
# llasso <- glmnet(
#   x = X_train_m,
#   y= y_trian,
#   alpha = 1, #lasso
#   lambda = parameters
# )
# 
# rridge <- glmnet(
#   x = X_train_m,
#   y = y_train,
#   alpha = 0, # Ridge,
#   lambda = parameters
# )
# ################################################
# 
