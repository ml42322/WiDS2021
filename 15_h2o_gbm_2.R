library(magrittr)
library(dplyr)

# Getting started
if(!("h2o" %in% installed.packages()[, "Package"] )) {
  install.packages("h2o")
}


### Start up the H2O Cluster
# Load the H2O library and start up the H2O cluster locally on your machine 
library(h2o)

# Number of threads, nthreads = -1, means use all cores to your machine
# max_mem_size is the maximum memory (in GB) to allocate to H2O
h2o.init(nthreads = -1, max_mem_size = "8G")
h2o.no_progress()

### Data Prep
# Load the data
train_ros <- read.csv("train_ros.csv")
test <- read.csv("test_data.csv")


train <- train_ros[, -which(names(train_ros) %in% c('encounter_id', 'hospital_id', 'icu_id', 'readmission_status'))]
test <- test[, -which(names(test) %in% c('encounter_id', 'hospital_id', 'icu_id', 'readmission_status'))]

### Ensure that all the categorical variables are factor variables
cols = colnames(train)[1:58]
train[cols] <- lapply(train[cols], factor)
test[cols] <- lapply(test[cols], factor)

### Move Data to H2O 
train_ros_h2o <- as.h2o(train)
test_h2o <- as.h2o(test)

# First, let's look at the shape and preview the data to make sure it loaded correctly.
# get the number of rows, cols
dim(train_ros_h2o)
dim(test_h2o)

# preview the data
head(train_ros_h2o)
head(test_h2o)

# Here the cleaning is already done so let's just look at a summary of the data
summary(train_ros_h2o)

# Check the distribution of the target
h2o.table(train_ros_h2o$diabetes_mellitus)
train_ros_h2o[["diabetes_mellitus"]]

### Encode response variable
# if the target is 0/1 and needs recoded, run this line
train_ros_h2o$diabetes_mellitus <- as.factor(train_ros_h2o$diabetes_mellitus) # encode the binary response as a factor

# check levels
h2o.levels(train_ros_h2o$diabetes_mellitus)

### Identify response and predictor variables
y <- "diabetes_mellitus"
x <- setdiff(names(train_ros_h2o), "diabetes_mellitus")

# List of predictor columns
x

### H2O Machine Learning
# Now that we have perpared the data, we can train some models
# We will start by training a GBM model and tehn adjjust its hyperparameters.
# For reference, H2O has several other built in supervised learning algorithms, 
# including: Generalized Linear Model (GLM), Random Forest (RF), Deep Learning (DL)

### Gradient Boosting Machine
# H2O's Gradient Boosting Machine (GBM) offers a Stochastic GBM, which can
# increase performance quite a bit compared to the original GBM implementation.

# In this section, we will builid several GBM models on the Diabetes dataset
# and then comparre their performance.

# We'll explore: 
# 1. Default GBM (categorical_encoding = enum)
# 2. Default GBM with One-Hot Encoding
# 3. GBM with more trees
# 4. Train a GBM with early stoppoing
# 5. Train a GBM with early stopping and smaller learning rate
# 6. Train a GBM with early stoppoing, shallower trees, and smaller learning trees

### 1. Train a default GBM
# First we will train a bsic GBM mdoel with default parameters.
# GBM will infer the rresponse distribution from the response encoding if not 
# specified explicitly thorug hthe 'distribution' argument. 
# A seed is required for reproducibility.

gbm_fit1 <- h2o.gbm(
  x = x,
  y = y,
  training_frame = train_ros_h2o,
  model_id = 'gbm_fit', #if you don't give it a model id then it will give you
  # some random alphanumeric name that's hard to remember
  seed = 1) 

# Print basic model info
print(gbm_fit1)

# Print variable importance
print(h2o.varimp(gbm_fit1))
h2o.varimp_plot(gbm_fit1)
# age, arf_apache, urineoutputt_apache, bmi, d1_glucose_min, h1_resprate_max,
# d1_calcium_max, d1_wbc_min, h1_mbp_noninvasive_max, d1_mbp_max

### 2. Train a default GBM with One-Hot Encoding
# The only difference between this model and the first is the categorical encoding parameter.
# This is using one hot encoding
# Dummy encoding is what you would do in linear regression so say you have k = 5
# but in dummy encoding you only have 4 types of coding and the fifth is your intercept
gbm_fit2 <- h2o.gbm(
  x = x,
  y = y,
  training_frame = train_ros_h2o,
  model_id = "gbm_fit_2",
  categorical_encoding = "OneHotExplicit",
  seed = 1
)

# Print basic model info
print(gbm_fit2)

# Print variable importance
print(h2o.varimp(gbm_fit2))
h2o.varimp_plot(gbm_fit2)

### 3. Train a GBM with more trees
# Next we will incrrease the number of trees used in the GBM by setting ntrees=50.
# The default number of trees in an H2O GBM is 50, so this GBM will be trained using ten times the default.
# Increasing the number of trees in a GBM is one way to increase performance of the model, however, you have 
# to be careful not to overfit your model to the training data by using too many trees.
# To automatically find the optimal number of trees, you must use H2O's early stopping functionality.
# This example will not do that, however, the following example will.

gbm_fit3 <- h2o.gbm(
  x = x,
  y = y,
  training_frame = train_ros_h2o,
  model_id = 'gbm_fit3',
  ntrees = 500,
  seed = 1
)

# Print basic model info
print(gbm_fit3)

# Print variable importance
print(h2o.varimp(gbm_fit3))
h2o.varimp_plot(gbm_fit3)

### 4. Train a GBM with early stopping
# We will again set ntrees = 500, however ,this time we will use early stopping in order to prevent overfitting (from too many trees).
# All of H2O's algorithms have early stopping available, however, with the except of Deep Learning, it is not enabled by default.

# There are several parameters that should be used to control early stopping.
# The three that are generic to all the algorithms are: stopping_rounds, stopping_metric, and stopping_tolerance.
# The stopping metric is the metric by which you'd like to measure performance, and so we will choose AUC here.
# The score_tree_interval is a parameter specific to Random Forest and GBM. 
# Setting score_tree_interval=5 will score the model after every five trees. 
# The parameters we have set below specify that the model will stop training afte there have been 
# three scoring intervals where the AUC has not increased more than 0.0005.
# Since we have specified a validation frme, the stopping tolerance will be computed on validation
# AUC rather than training AUC.

# Now let's use early stopping to find optimal ntrees
gbm_fit4 <- h2o.gbm(
  x = x,
  y = y,
  training_frame = train_ros_h2o,
  model_id = 'gbm_fit4',
  ntrees = 500,
  score_tree_interval = 5, # used for early stopping # just checks the score every 5 trees
  stopping_rounds = 3, # used for early stopping
  stopping_metric = "AUC", # used for early stopping # this is the performance metric thatt is important to me
  stopping_tolerance = 0.0005, # used for early stopping
  seed = 1
)

# Print basic model info
print(gbm_fit4)

# Print variable importance
print(h2o.varimp(gbm_fit4))
h2o.varimp_plot(gbm_fit4)

### 5. Train a GBM with early stopping and smaller learning rate
# This model will use ntrees=500 and early stotpping, but this time, we will introduce a smaller learning rate into the model.
gbm_fit5 <- h2o.gbm(
  x = x,
  y = y,
  training_frame = train_ros_h2o,
  model_id = 'gbm_fit5',
  ntrees = 500, # one of the most importannt parameter,
  score_tree_interval = 5, # used for early stopping # just checks the score every 5 trees
  stopping_rounds = 3, # used for early stopping
  stopping_metric = "AUC", # used for early stopping # this is the performance metric thatt is important to me
  stopping_tolerance = 0.0005, # used for early stopping
  learn_rate = 0.05, # one of the most important paramter
  seed = 1
)

# Print basic model info
print(gbm_fit5)

# Print variable importance
print(h2o.varimp(gbm_fit5))
h2o.varimp_plot(gbm_fit5)

### 6. Train a GBM with early stopping, shallower trees, and small learning rate
# In this model, we'll continue to use early stopping, but this time we'll increase
# 'max_depth' to make deeper trees (default = 5) and lower the learn rate = 0.1
# third most effective tuning parameter is the tree depth
gbm_fit6 <- h2o.gbm(
  x = x,
  y = y,
  training_frame = train_ros_h2o,
  model_id = 'gbm_fit6',
  ntrees = 500, 
  score_tree_interval = 5, # used for early stopping
  stopping_rounds = 3, # used for early stopping
  stopping_metric = "AUC", # used for early stopping 
  stopping_tolerance = 0.0005, # used for early stopping
  learn_rate = 0.05, 
  max_depth = 3,
  seed = 1
)

# Print basic model info
print(gbm_fit6)

# Print variable importance
print(h2o.varimp(gbm_fit6))
h2o.varimp_plot(gbm_fit6)


### Compare model performance
# Let's compare the performance of the six GBMs that were just trained on the 
# test set (data the model has not seen before.)
gbm_perf1 <- h2o.performance(gbm_fit1, newdata = test_h2o)
gbm_perf2 <- h2o.performance(gbm_fit2, newdata = test_h2o)
gbm_perf3 <- h2o.performance(gbm_fit3, newdata = test_h2o)
gbm_perf4 <- h2o.performance(gbm_fit4, newdata = test_h2o)
gbm_perf5 <- h2o.performance(gbm_fit5, newdata = test_h2o)
gbm_perf6 <- h2o.performance(gbm_fit6, newdata = test_h2o)

# Retrieve test set AUC
gbm_perf1@metrics$AUC # 0.5427583
gbm_perf2@metrics$AUC 
gbm_perf3@metrics$AUC # 0.5361977
gbm_perf4@metrics$AUC
gbm_perf5@metrics$AUC
gbm_perf6@metrics$AUC

# Which model performed the best? Think about and discuss the reasons for this.
# Now let's compare the performance of the model trained on 'ntrees=500' (Model 3),
# with the final model with shallower trees and early stopping on the training set:
# gbm_perf3_train <- h2o.performance(gbm_fit3, newdata  = train)
# gbm_perf6_train <- h2o.performance(gbm_fit6, newdata  = train)
# gbm_perf3_train@metrics$AUC
# gbm_perf6_train@metrics$AUC

# the problem with r-squared is you can inflate it with predictors
# adjusted r-squared penalizes you for having too many variables and is a better 
# performance metric than just r squared

# What do you notice about hte relationship between a model's performance on the training set
# and the test set?

## H2O Grid Search for GBMs
# Training models one-by-one, as we just did, can quickly become tedious.
# In this section, rather than training models manually, we will make use of the
# H2O Grid Search functionality train a bunch of models at once.
# H2O offers two types of grid search -- "Cartesian" and "RandomDiscrete".
# Caretsian is the traditional, exhaustive, grid search over all combinations
# of model parameters in the grid.
# Random grid search will sample sets of model parameters randomly for some specified period of time (or maximum number of models).
# We will continue to use GBMs to demosntrate H2O's grid search functionality.

### Caretsian Grid Search
# We first need to define a grid of GBM model hyperparamters. For this particular example, we 
# will grid over the following model parameters:
# - 'learn_rate'
# - 'max_depth'
# - 'sample_rate'
# -'col_sample_rate'

# GBM hyperparamters
gbm_params1 <- list(
  "learn_rate" = c(0.01, 0.1),
  "max_depth" = c(3, 5, 7),
  "sample_rate" = c(0.8, 1.0),
  "col_sample_rate" = c(0.2, 0.5, 1.0)
)

### Train and validate a grid of GBMs 
# If you want to specify non-default model parameters that are not part of your 
# grid, you can pass them along to 'h2o.grid()' via the '...' argument.
# We will enable early stopping here as well.

gbm_grid_1 <- h2o.grid(
  algorithm = "gbm",
  grid_id = "gbm_grd1_r",
  x = x,
  y = y,
  training_frame = train_ros_h2o,
  hyper_params = gbm_params1,
  ntrees = 500,
  score_tree_interval = 5, #used for early stopping
  stopping_rounds = 3, #used for early stopping
  stopping_metric = "AUC", #used for early stopping
  stopping_tolerance = 0.0005, #used for early stoppin
  seed = 1)

### Compare model performance
# To compare the model performance among all models in a grid, sorted by a 
# particular metric (e.g. AUC), you can use the 'get_grid' method.
gbm_grid_perf1 <- h2o.getGrid("gbm_grd1_r", sort_by = "auc", decreasing = TRUE)

# Lastly, let's extract the top model, as determined by validation AUC, from the grid.
# Grab the model_id for the top GBM chosen by validation AUC
best_gbm_model <- h2o.getModel(gbm_grid_perf1@model_ids[[1L]])

# Now let's evaluate the model performance on a test set 
# so we can get an honest estimate of top model performance
gbm_perf <- h2o.performance(best_gbm_model, newdata= test_h2o)

# Check out the performance
print(gbm_perf) # 0.5323208

### Random Grid Search
# This example is set to run fairly quickly -- increase 'max_runtime_secs' or
# 'max_models' to cover more of the hyperparamter space. Also, you can expand
# the hyperparamtter space of each of the algorithms by modifying the hyperparamterr
# list below.
# In addition to the hyperparamtter dictionary, we will specify the 
# 'search_criteria' as 'RandomDiscrete', with a max number of models equal to 36.

# GBM hyperparameters
gbm_params2 = list(
  "learn_rate" = 1L:10L/100L,
  "max_depth" = 2L:10L,
  "sample_rate" = 5L:10L/10L,
  "col_sample_rate" = 1L:10L/10L
)

# Search criteria
search_criteria2 = list(
  "strategy" = "RandomDiscrete",
  "max_models" = 36
)

# Train and validate a random grid of GBMs
gbm_grid2 <- h2o.grid(
  algorithm = "gbm",
  grid_id  = "gbm_grid_2",
  x = x,
  y = y,
  training_frame = train_ros_h2o,
  hyper_params = gbm_params2,
  ntrees = 500,
  score_tree_interval = 5, #used for early stopping
  stopping_rounds = 3, #used for early stopping
  stopping_metric = "AUC", #used for early stopping
  stopping_tolerance = 0.0005, #used for early stopping
  seed = 1
)

# Compare model performance
gbm_grid_perf2 <- h2o.getGrid("gbm_grid_2", sort_by = "auc", decreasing = TRUE)

# Lastly, let's extract the top model, as determined by validation AUC, from the grid.
# Grab the model_id for the top GBM chosen by validation AUC
best_gbm_model <- h2o.getModel(gbm_grid_perf2@model_ids[[1L]])

# Now let's evaluate the model performance on a test set 
# so we can get an honest estimate of top model performance
(gbm_perf <- h2o.performance(best_gbm_model, newdata= test))

# This is slightly higher than the AUC on the validation set of the top model.
# We typically do NOT see this.
# Models generally perform better on validation data than test data, but in this case,
# it's likely due to the small of amount of data in the validation and test sets.

## Cross-Validation
# Early in this script, we split the data into three parts: (1) training, (2) validation
# and (3) test. A different - and generally better way - to split the dat is with 
# cross-validation. Here's what H2O has to say about cross-validation.

# K-fold cross validation is used to validate a model internally, i.e., estimate
# the model performance without having to sacrifice a validation split.
# Also, you avoid statistical issues with your validation split (it might be a 
# "lucky" split especially for imbalanced data). Good values for K are around 5
# to 10. Comparing the K validation metrics is always a good idea, to check
# the stability of the estimation, before "trosting" the main mdoel. 
# You have to make sure, however, that the holdout sets for each of the K models
# are good."

# As H2O mentioned, with cross-validation, we do not have to "sacrifice" training
# data for a validation set. Thus, we can combine the training and validation 
# sets we created earlier for into a larger training set.
# combine validation set back into the training set
# train2 <- h2o.rbind(train, valid)

# verify the dimensions of the new, larger training set
# dim(train) # oldtraining observations
# dim(train2) # new training observations

# Next, we will fit a GBM using the cross-validation argument 'nfolds' and set 
# it equal to 5. We can also use the argument 'balance_classes', which will 
# enforce each fold of the data to have the same target distribution.

gbm_fit7 <- h2o.gbm(
  x = x,
  y = y,
  training_frame = train_ros_h2o,
  model_id = 'gbm_fit7',
  ntrees = 500,
  score_tree_interval = 5, #used for early stopping
  stopping_rounds = 3, #used for early stopping
  stopping_metric = "AUC", #used for early stopping
  stopping_tolerance = 0.0005, #used for early stopping
  learn_rate = 0.05,
  max_depth = 3,
  nfolds = 5,
  balance_classes = TRUE,
  seed = 1
)

# Retrieve test set AUC
gbm_perf7 <- h2o.performance(gbm_fit7, newdata = test_h2o)@metrics$AUC

# Check out the performance
print(gbm_perf7) # 0.5474839

# Cross-validation can be combined with grid search options we explore above,
# but we will leave that as an exercise

# Train and validate a random grid of GBMs
gbm_grid3 <- h2o.grid(
  algorithm = "gbm",
  grid_id  = "gbm_grid3_r",
  x = x,
  y = y,
  training_frame = train_ros_h2o,
  hyper_params = gbm_params2,
  validation_frame = valid,
  ntrees = 500,
  score_tree_interval = 5, #used for early stopping
  stopping_rounds = 3, #used for early stopping
  stopping_metric = "AUC", #used for early stopping
  stopping_tolerance = 0.0005, #used for early stopping
  learn_rate = 0.05,
  max_depth = 3, 
  nfolds = 5,
  balance_classes = TRUE,
  seed = 1
)


## Final thoughts
# In this notebook, we gave a quick tutorial on using and tuning GBMs with H2O.

# remember to shut down your H2O cluster
h2o.shutdown(prompt = FALSE)





