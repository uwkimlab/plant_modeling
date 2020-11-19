##### SEFS508: Plant Modeling #####
# Lab08: Machine learning for crop yield predictions
# Last modified: 17 Nov 2020 by Soo-Hyung Kim

# Set working directory to source file location (in Session menu)

#install.packages("randomForest")
#install.packages("hydroGOF")
#install.packages("tidyverse")

library(randomForest) # random forest CART package
library(hydroGOF) # this library provides a variety of model performance stats
library(dplyr) # data plier 
library(ggplot2) # popular plotting package

##### Data preparation #####
#Read crop yield data for the Northeast Seaboard Region used in Jeong et al. (2016).
#This data originates from Resop et al. (2014).

crop<-read.table(file="./L08-potato.csv", sep= ",", header = T)

#Data quality check. Use observed yield data that are positve and non-missing. 
crop<-crop[crop$obs_yield >= 0.0,]
tmpdf<-na.omit(crop)

## Count the number of remaining valid observations
## divide them randomly into training and test data
n <- nrow(tmpdf)
indextrain <- sample(1:n,round(0.7*n))
crop.traindf<-tmpdf[indextrain,]
crop.testdf <-tmpdf[-indextrain,]

##### Generalized Linear Model (GLM) ######
# Apply multiple linear regression to predict obs_yield based on input variables
# GLM is a benchmark
# model training
crop.glm<-lm(obs_yield ~ clay + bulk + water + hyd + sat+ precip + maxt + mint + rad +lat+elev, data = crop.traindf)
summary(crop.glm)

# Plot training result
layout(matrix(c(1),1,1)) # graph configuration/page 

plot(crop.traindf$obs_yield, crop.glm$fitted.values, main = "Multiple Linear Regression: Training",
     xlab = "Observed (tons/ha)", ylab = "Predicted (tons/ha)",
     pch = 1, frame = FALSE)
abline(lm(crop.glm$fitted.values ~ crop.traindf$obs_yield, data = crop.traindf), col = "blue")
abline(a = 0, b = 1, col = "red")

# model testing and plot test result
crop.glm.test.pred<-predict(crop.glm, newdata = crop.testdf)
plot(crop.testdf$obs_yield, crop.glm.test.pred, main = "Multiple Linear Regression: Testing",
     xlab = "Observed (tons/ha)", ylab = "Predicted (tons/ha)",
     pch = 1, frame = FALSE)
abline(lm(crop.glm.test.pred ~ crop.testdf$obs_yield, data = crop.testdf), col = "blue")
abline(a = 0, b = 1, col = "red")

##### Random Forests (RF) #####
# RF model training
crop.rf<-randomForest(obs_yield ~., data = crop.traindf, ntree = 500, keepforest = T)  
crop.rf
summary(crop.rf)
importance(crop.rf)
varImpPlot(crop.rf)
hist (treesize(crop.rf))

# Plot training result
layout(matrix(c(1),1,1)) # graph configuration/page 
plot(crop.traindf$obs_yield, crop.rf$predicted, main = "Random Forests: Training",
     xlab = "Observed (tons/ha)", ylab = "Predicted (tons/ha)",
     pch = 1, frame = FALSE)
abline(lm(crop.rf$predicted ~ crop.traindf$obs_yield, data = crop.traindf), col = "blue")
abline(a = 0, b = 1, col = "red")

# RF model testing and plot result
crop.rf.test.pred <-predict(crop.rf, newdata = crop.testdf)
plot(crop.testdf$obs_yield, crop.rf.test.pred, main = "Random Forests: Testing",
     xlab = "Observed (tons/ha)", ylab = "Predicted (tons/ha)",
     pch = 1, frame = FALSE)
abline(lm(crop.rf.test.pred ~ crop.testdf$obs_yield, data = crop.testdf), col = "blue")
abline(a = 0, b = 1, col = "red")


##### Deep Neural Network (DNN) using TensorFlow #####
# to use this package, you might have to install Anaconda/Miniconda first.
# See: https://docs.conda.io/en/latest/miniconda.html
# Or install/update TensorFlow package of your Python environment separately.
# see: https://www.tensorflow.org/install 
# Code below is based on the regressio tutorial at:
# https://tensorflow.rstudio.com/tutorials/beginners/basic-ml/

#install.packages("tensorflow")
#install.packages("keras")
#install.packages("tfdatasets")
install_tensorflow()

library(tensorflow)
library(keras)
library(tfdatasets)


# normalize driving variables to scale their effects
spec <- feature_spec(crop.traindf, obs_yield ~ . ) %>% 
  step_numeric_column(all_numeric(), normalizer_fn = scaler_standard()) %>% 
  fit()

spec

# create neural network layers
layer <- layer_dense_features(
  feature_columns = dense_features(spec), 
  dtype = tf$float32)
layer(crop.traindf)

#summary(model)

# create a function to build a DNN model
build_model <- function() {
  input <- layer_input_from_dataset(crop.traindf %>% select(-obs_yield))
  
  output <- input %>% 
    layer_dense_features(dense_features(spec)) %>% 
    layer_dense(units = 64, activation = "relu") %>%
    layer_dense(units = 64, activation = "relu") %>%
    layer_dense(units = 1) 
  
  model <- keras_model(input, output)
  
  model %>% 
    compile(
      loss = "mse",
      optimizer = optimizer_rmsprop(),
      metrics = list("mean_absolute_error")
    )
  
  model
}

# Display training progress by printing a single dot for each completed epoch.
print_dot_callback <- callback_lambda(
  on_epoch_end = function(epoch, logs) {
    if (epoch %% 80 == 0) cat("\n")
    cat(".")
  }
)    

model <- build_model()

history <- model %>% fit(
  x = crop.traindf %>% select(-obs_yield),
  y = crop.traindf$obs_yield,
  epochs = 100,
  validation_split = 0.2,
  verbose = 0,
  callbacks = list(print_dot_callback)
)
plot(history)


# Plot training result
crop.dnn.train.pred <- model %>% predict(crop.traindf %>% select(-obs_yield))
layout(matrix(c(1),1,1)) # graph configuration/page 
plot(crop.traindf$obs_yield, crop.dnn.train.pred, main = "Deep Neural Network: Training",
     xlab = "Observed (tons/ha)", ylab = "Predicted (tons/ha)",
     pch = 1, frame = FALSE)
abline(lm(crop.dnn.train.pred ~ crop.traindf$obs_yield, data = crop.traindf), col = "blue")
abline(a = 0, b = 1, col = "red")

# DNN model testing and plot result
crop.dnn.test.pred <-model %>% predict(crop.testdf %>% select(-obs_yield))
plot(crop.testdf$obs_yield, crop.dnn.test.pred, main = "Deep Neural Network: Testing",
     xlab = "Observed (tons/ha)", ylab = "Predicted (tons/ha)",
     pch = 1, frame = FALSE)
abline(lm(crop.dnn.test.pred ~ crop.testdf$obs_yield, data = crop.testdf), col = "blue")
abline(a = 0, b = 1, col = "red")

##### Model Performance Evaluations #####
# compare model performance
gof.test.glm <- gof(crop.testdf$obs_yield, crop.glm.test.pred)
gof.test.rf <- gof(crop.testdf$obs_yield, crop.rf.test.pred)
gof.test.dnn <- gof(crop.testdf$obs_yield, c(crop.dnn.test.pred))

#pred vs obs
p_o.test.glm <- lm(crop.glm.test.pred ~ crop.testdf$obs_yield)
p_o.test.rf <- lm(crop.rf.test.pred ~ crop.testdf$obs_yield) # pred vs obs
p_o.test.dnn <- lm(crop.dnn.test.pred ~ crop.testdf$obs_yield)

# compare model performance for training data
gof.train.glm <- gof(crop.traindf$obs_yield, crop.glm$fitted.values)
gof.train.rf <- gof(crop.traindf$obs_yield, crop.rf$predicted)
gof.train.dnn <- gof(crop.traindf$obs_yield, c(crop.dnn.train.pred))

#pred vs obs
p_o.train.glm <- lm(crop.glm$fitted.values~crop.traindf$obs_yield)
p_o.train.rf <- lm(crop.rf$predicted~crop.traindf$obs_yield) # pred vs obs
p_o.train.dnn <- lm(crop.dnn.train.pred ~ crop.traindf$obs_yield)

performance <- cbind (gof.train.glm, gof.train.rf, gof.train.dnn, gof.test.glm, 
                      gof.test.rf, gof.test.dnn)
colnames(performance) <- c("train.glm", "train.rf", "train.dnn", "test.glm", "test.rf", "test.dnn")

#exporting output
write.csv(performance, file = "performance_stats.csv")
write.csv(cbind(crop.traindf, crop.rf$predicted, crop.glm$fitted.values, crop.dnn.train.pred), file = "crop_train_results.csv")
write.csv(cbind(crop.testdf, crop.glm.test.pred, crop.rf.test.pred, crop.dnn.test.pred), file = "crop_test_results.csv")

##### Silage Corn Data for the same region used in Jeong et al. (2016) ##### 
#silage<-read.table(file="./L08_silage.csv", sep= ",", header = T)

