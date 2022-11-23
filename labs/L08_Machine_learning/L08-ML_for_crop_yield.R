##### SEFS508: Plant Modeling #####
# Lab08: Machine learning for crop yield predictions
# Last modified: 22 Nov 2022 by Soo-Hyung Kim

# Set working directory to source file location (in Session menu)

# install.packages("randomForest")
# install.packages("hydroGOF")
# install.packages("tidyverse")
# install.packages("tidymodels")
# install.packages("skimr")

library(randomForest) # random forest CART package
library(hydroGOF) # this library provides a variety of model performance stats
library(dplyr) # data plier 
library(ggplot2) # popular plotting package
library(skimr)
library(tidyverse)
library(tidymodels)

##### DNN Preparation #####################
# Install and load Tensorflow and other packages for using DNN in RStudio 
# install.packages("tensorflow")
# install.packages("keras")
# install.packages("reticulate") # This is likely to have been installed already.

# If Python is not installed in your machine, uncomment and run the next two lines
#path_to_python <- install_python()
#virtualenv_create("r-reticulate", python = path_to_python)

#If Python is already installed, run and create a virtual environment for Tensorflow
library(reticulate)
virtualenv_create("r-reticulate")

# if first time to run a session of this code, run the next four lines.
 # library(tensorflow)
 # library(keras)
 # install_tensorflow(envname = "r-reticulate")
 # install_keras(envname = "r-reticulate")
# If you ran the above 4 lines to install Tensorflow and keras,
# your R session would have restarted. Once it's done comment out the 4 lines

library(tensorflow)
library(keras)
# Test if Tensorflow is installed properly
tf$constant("Hello Tensorflow")


##### Data preparation #####
#Read crop yield data for the Northeast Seaboard Region used in Jeong et al. (2016).
#This data originates from Resop et al. (2014).

crop<-read.table(file="./L08-potato.csv", sep= ",", header = T)

#Data quality check. Use observed yield data that are positve and non-missing. 
crop<-crop[crop$obs_yield >= 0.0,]
tmpdf<-na.omit(crop)

## divide observations randomly into 70% training data and the rest as test data
split <- initial_split(tmpdf, 0.7)
train_dataset <- training(split)
test_dataset <- testing(split)

##### Generalized Linear Model (GLM) ######
# Apply multiple linear regression to predict obs_yield based on input variables
# GLM is a benchmark
# model training
glm<-lm(obs_yield ~ clay + bulk + water + hyd + sat+ precip + maxt + mint + rad +lat+elev, data = train_dataset)
summary(glm)

# Plot training result
layout(matrix(c(1),1,1)) # graph configuration/page 

plot(train_dataset$obs_yield, glm$fitted.values, main = "Multiple Linear Regression: Training",
     xlab = "Observed (tons/ha)", ylab = "Predicted (tons/ha)",
     pch = 1, frame = FALSE)
abline(lm(glm$fitted.values ~ train_dataset$obs_yield, data = train_dataset), col = "blue")
abline(a = 0, b = 1, col = "red")

# model evaluation with independent data and plot the result
glm.test.pred<-predict(glm, newdata = test_dataset)
plot(test_dataset$obs_yield, glm.test.pred, main = "Multiple Linear Regression: Evaluation",
     xlab = "Observed (tons/ha)", ylab = "Predicted (tons/ha)",
     pch = 1, frame = FALSE)
abline(lm(glm.test.pred ~ test_dataset$obs_yield, data = test_dataset), col = "blue")
abline(a = 0, b = 1, col = "red")

##### Random Forests (RF) #####
# RF model training
rf<-randomForest(obs_yield ~., data = train_dataset, ntree = 500, keepforest = T)  
rf
summary(rf)
importance(rf)
varImpPlot(rf)
hist (treesize(rf))

# Plot training result
layout(matrix(c(1),1,1)) # graph configuration/page 
plot(train_dataset$obs_yield, rf$predicted, main = "Random Forests: Training",
     xlab = "Observed (tons/ha)", ylab = "Predicted (tons/ha)",
     pch = 1, frame = FALSE)
abline(lm(rf$predicted ~ train_dataset$obs_yield, data = train_dataset), col = "blue")
abline(a = 0, b = 1, col = "red")

# RF model evaluation with independent data
rf.test.pred <-predict(rf, newdata = test_dataset)
# Plot evaluation result
plot(test_dataset$obs_yield, rf.test.pred, main = "Random Forests: Testing",
     xlab = "Observed (tons/ha)", ylab = "Predicted (tons/ha)",
     pch = 1, frame = FALSE)
abline(lm(rf.test.pred ~ test_dataset$obs_yield, data = test_dataset), col = "blue")
abline(a = 0, b = 1, col = "red")


##### Deep Neural Network (DNN) using TensorFlow #####
# For RStudio installation of TensorFlow: https://tensorflow.rstudio.com/installation/
# Code below is based on the regression tutorial at:
# https://tensorflow.rstudio.com/tutorials/keras/regression

#Skim the data set
skimr::skim(train_dataset)

#Split features from labels
#Separate the target value—the “label”—from the features. 
#This label is the value that you will train the model to predict.

train_features <- train_dataset %>% select(-obs_yield)
test_features <- test_dataset %>% select(-obs_yield)

train_labels <- train_dataset %>% select(obs_yield)
test_labels <- test_dataset %>% select(obs_yield)

# For NN, data for different variables are put into the same scale
# That is, driving variables are normalized to scale their effects
# See: https://tensorflow.rstudio.com/reference/keras/layer_normalization
normalizer <- layer_normalization(axis = -1L)
normalizer %>% adapt(as.matrix(train_features))

# Once normalized, the meanings of variables are no longer applicable
# As we can see here.
print(normalizer$mean)
first <- as.matrix(train_features[1,])
cat('First example:', first)
cat('Normalized:', as.matrix(normalizer(first)))

# Create a DNN model with a few layers of neural network including “hidden” layers. 
# The name “hidden” here just means not directly connected to the inputs or outputs.
# The first layer is a normalization layer we just created.
# Two hidden, non-linear, Dense layers are added with the ReLU (relu) function.
# Another layer of linear Dense single-output is the added.
# Define a DNN model to be built and compiled using MAE (loss) as optimizer.
build_and_compile_model <- function(norm) {
  model <- keras_model_sequential() %>%
    norm() %>%
    layer_dense(64, activation = 'relu') %>%
    layer_dense(64, activation = 'relu') %>%
    layer_dense(1)
  
  model %>% compile(
    loss = 'mean_absolute_error',
    optimizer = optimizer_adam(0.001)
  )
  
  model
}

# Build an instance of the DNN model using normalized data
dnn_model <- build_and_compile_model(normalizer)
summary(dnn_model)

# Display training progress by printing a single dot for each completed epoch.
print_dot_callback <- callback_lambda(
  on_epoch_end = function(epoch, logs) {
    if (epoch %% 80 == 0) cat("\n")
    cat(".")
  }
)    

# Train DNN model (i.e., calibrate)
# Be patient. It takes a while.
# Uncomment 'callbacks' to see the progress but will slows it down further
history <- dnn_model %>% fit(
  as.matrix(train_features),
  as.matrix(train_labels),
  validation_split = 0.2,
  verbose = 0,
  epochs = 200,
#  callbacks = list(print_dot_callback)
)

# Plot the history of training.
# 'epochs' is like iterations of training
plot(history)

# Collect test results with data not used for training
test_results <- list()
test_results[['dnn_model']] <- dnn_model %>% evaluate(
  as.matrix(test_features),
  as.matrix(test_labels),
  verbose = 0
)

# Plot training result
dnn.train.pred <- predict(dnn_model,as.matrix(train_features))
layout(matrix(c(1),1,1)) # graph configuration/page 
plot(train_dataset$obs_yield, dnn.train.pred, main = "Deep Neural Network: Training",
     xlab = "Observed (tons/ha)", ylab = "Predicted (tons/ha)",
     pch = 1, frame = FALSE)
abline(lm(dnn.train.pred ~ train_dataset$obs_yield, data = train_dataset), col = "blue")
abline(a = 0, b = 1, col = "red")

# DNN model testing and plot result
dnn.test.pred <-predict(dnn_model,as.matrix(test_features))
plot(test_dataset$obs_yield, dnn.test.pred, main = "Deep Neural Network: Testing",
     xlab = "Observed (tons/ha)", ylab = "Predicted (tons/ha)",
     pch = 1, frame = FALSE)
abline(lm(dnn.test.pred ~ test_dataset$obs_yield, data = test_dataset), col = "blue")
abline(a = 0, b = 1, col = "red")

##### Model Performance Evaluations #####
# compare model performance
gof.test.glm <- gof(test_dataset$obs_yield, glm.test.pred)
gof.test.rf <- gof(test_dataset$obs_yield, rf.test.pred)
gof.test.dnn <- gof(test_dataset$obs_yield, c(dnn.test.pred))

#pred vs obs
p_o.test.glm <- lm(glm.test.pred ~ test_dataset$obs_yield)
p_o.test.rf <- lm(rf.test.pred ~ test_dataset$obs_yield) # pred vs obs
p_o.test.dnn <- lm(dnn.test.pred ~ test_dataset$obs_yield)

# compare model performance for training data
gof.train.glm <- gof(train_dataset$obs_yield, glm$fitted.values)
gof.train.rf <- gof(train_dataset$obs_yield, rf$predicted)
gof.train.dnn <- gof(train_dataset$obs_yield, c(dnn.train.pred))

#pred vs obs
p_o.train.glm <- lm(glm$fitted.values~train_dataset$obs_yield)
p_o.train.rf <- lm(rf$predicted~train_dataset$obs_yield) # pred vs obs
p_o.train.dnn <- lm(dnn.train.pred ~ train_dataset$obs_yield)

performance <- cbind (gof.train.glm, gof.train.rf, gof.train.dnn, gof.test.glm, 
                      gof.test.rf, gof.test.dnn)
colnames(performance) <- c("train.glm", "train.rf", "train.dnn", "test.glm", "test.rf", "test.dnn")

#exporting output
write.csv(performance, file = "performance_stats.csv")
write.csv(cbind(train_dataset, rf$predicted, glm$fitted.values, dnn.train.pred), file = "crop_train_results.csv")
write.csv(cbind(test_dataset, glm.test.pred, rf.test.pred, dnn.test.pred), file = "crop_test_results.csv")

##### Silage Corn Data for the same region used in Jeong et al. (2016) ##### 
#silage<-read.table(file="./L08-silage.csv", sep= ",", header = T)

