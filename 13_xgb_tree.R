# Load relevant libraries
library(tidyverse)
library(magrittr)
library(caret)
library(xgboost)
library(Matrix)

########################################
# SET UP FOR MODELING
########################################
# Load cleaned data frame for modeling
train_rus <- read.csv("train_rus.csv")
test <- read.csv("test_data.csv")
train <- train_rus[, -which(names(train_rus) %in% c('encounter_id', 'hospital_id', 'icu_id'))]

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

data_rus <- rbind(train_rus[, -which(names(train_rus) %in% c('encounter_id', 'hospital_id', 'icu_id', 'readmission_status'))]
                  , test[, -which(names(test) %in% c('encounter_id', 'hospital_id', 'icu_id', 'readmission_status'))])

data_ros <- rbind(train_ros[, -which(names(train_ros) %in% c('encounter_id', 'hospital_id', 'icu_id', 'readmission_status'))]
                  , test[, -which(names(test) %in% c('encounter_id', 'hospital_id', 'icu_id', 'readmission_status'))])

target_rus <- train_rus$diabetes_mellitus
target_ros <- train_ros$diabetes_mellitus

data_sparse <- sparse.model.matrix(~.-1, data = as.data.frame(data_rus))
dtrain <- xgb.DMatrix(data = data_sparse[1:nrow(train_rus), ], label = train_rus$diabetes_mellitus) 
dtest <- xgb.DMatrix(data = data_sparse[(nrow(train_rus)+1):nrow(data_rus), ])

data_sparse2 <- sparse.model.matrix(~.-1, data = as.data.frame(data_ros))
dtrain2 <- xgb.DMatrix(data = data_sparse[1:nrow(train_ros), ], label = train_ros$diabetes_mellitus) 
dtest2 <- xgb.DMatrix(data = data_sparse[(nrow(train_ros)+1):nrow(data_ros), ])

# param <- list(max_depth = 2, eta = 1, verbose = 0, nthread = 2,
#               objective = "binary:logistic", eval_metric = "auc")
# watchlist <- list(train = dtrain, eval = dtest)
# bst <- xgb.train(param, dtrain, nrounds = 2, watchlist)
# bst <- xgb.train(param, dtrain, nrounds = 25, watchlist,
#                  early_stopping_rounds = 3)
# 




gc()

searchGridSubCol <- expand.grid(subsample = c(0.5, 0.6), 
                                colsample_bytree = c(0.5, 0.6),
                                max_depth = c(3, 4),
                                min_child = seq(1), 
                                eta = c(0.1)
)

ntrees <- 100

system.time(
  rmseErrorsHyperparameters <- apply(searchGridSubCol, 1, function(parameterList){
    
    #Extract Parameters to test
    currentSubsampleRate <- parameterList[["subsample"]]
    currentColsampleRate <- parameterList[["colsample_bytree"]]
    currentDepth <- parameterList[["max_depth"]]
    currentEta <- parameterList[["eta"]]
    currentMinChild <- parameterList[["min_child"]]
    xgboostModelCV <- xgb.cv(data =  dtrain, nrounds = ntrees, nfold = 5, showsd = TRUE, 
                             metrics = "rmse", verbose = TRUE, "eval_metric" = "rmse",
                             "objective" = "reg:linear", "max.depth" = currentDepth, "eta" = currentEta,                               
                             "subsample" = currentSubsampleRate, "colsample_bytree" = currentColsampleRate
                             , print_every_n = 10, "min_child_weight" = currentMinChild, booster = "gbtree",
                             early_stopping_rounds = 10)
    
    xvalidationScores <- as.data.frame(xgboostModelCV$evaluation_log)
    rmse <- tail(xvalidationScores$test_rmse_mean, 1)
    trmse <- tail(xvalidationScores$train_rmse_mean,1)
    output <- return(c(rmse, trmse, currentSubsampleRate, currentColsampleRate, currentDepth, currentEta, currentMinChild))}))

output <- as.data.frame(t(rmseErrorsHyperparameters))
varnames <- c("TestRMSE", "TrainRMSE", "SubSampRate", "ColSampRate", "Depth", "eta", "currentMinChild")
names(output) <- varnames
head(output)

write.csv(output, "xgb_gridsearch.csv")

# ### XGB Boost
# # Train extreme gradient boosted decision trees model using caret::train
# # the caret::train package does a grid search and recommends the best hyperparamters
# set.seed(1)
# xgb_tree <- train(diabetes_mellitus ~ .,
#                   data = train,
#                   method = "xgbTree",
#                   trControl = train("cv", number = 10)
#                   )
# 
# # save(xgb_tree, file= "/Users/preeyamody/Desktop/widsdatathon2021/xgb_tree.Rdata")
# 
# # Print the ideal hyperparamter settings determiend by caret::train()
# xgb_tree$bestTune
# 
# # Print the most important covariates
# varImp(xgb_tree)
# 
# y_hat <- xgb_tree %>% 
#   predict(test)
# 
# mean(y_hat == test$diabetes_mellitus)
# 
# # Take a look into sensitivity, specficity, false pos rate, false neg rate
# confusionMatrix(data = y_hat,
#                 reference = test$diabetes_mellitus)
# 
# # Accuracy = ~ (/10 diabetes patietnts are identified correctly)
# 
# # False positive = 1 - specificity
# 
# # False negative = 1 - sensitivity