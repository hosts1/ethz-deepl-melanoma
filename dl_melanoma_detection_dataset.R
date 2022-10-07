# Project - Deep learning

library(fs)
library(grid)
library(keras)
library(tensorflow)
library(tidyverse)
library(tfdatasets)
library(randomForest)

# Define utility functions ----
load_data <- function(base_path, type = 'train', image_size=64) { 
  
  path <- paste0(base_path, "/", type)
  img_ls<- fs::dir_ls(path, recurse = TRUE, type='file') 
  
  y_type <- rep('default', length(img_ls))
  X_type = array(dim=c(length(img_ls), image_size, image_size ,3L))
  
  loaded = 0
  for (idx in 1:length(img_ls)) {
    
    img_path <- img_ls[idx]
    type <- stringr::str_extract(img_path, "melanoma|nevus|seborrheic_keratosis") # Not really nice
    print(paste0(str(loaded), ": ", type))
    img <- keras::image_load(img_path, target_size = c(image_size,image_size))
    img.arr <- keras::image_to_array(img)
    X_type[idx,,,] = img.arr
    y_type[idx] <- type
    
    # Dummy code output 
    loaded = loaded + 1
    
  }
  return (list(X=X_type, y_label=as.factor(y_type), y_one_hot_encoded=to_categorical(as.integer(as.factor(y_type)) - 1 , num_classes = 3)))
}


# Working on pre-trained VGG Dataset
application_vgg16 = function (include_top = TRUE,
                              weights = "imagenet",
                              input_tensor = NULL,
                              input_shape = NULL,
                              pooling = NULL,
                              classes = 1000){
  
  tf$keras$applications$VGG16(include_top = include_top, weights = weights, 
                              input_tensor = input_tensor, 
                              # MP: comment out next line, include succeeding line
                              #input_shape = normalize_shape(input_shape)
                              input_shape = input_shape, 
                              pooling = pooling, classes = as.integer(classes))
}


# Load datasets ----
base_path = 'Projects/deep-learning-detection/skin-lesions'
image_size = 64L

d_train = load_data(base_path = base_path, type='train',
              image_size = image_size)
X_train = d_train$X 
Y_train = d_train$y_one_hot_encoded # Classes one-hot-encoded
y_train = d_train$y_label

## Validation data
d_val = load_data(base_path = 'Projects/deep-learning-detection/skin-lesions-dummy', type='valid',
              image_size = image_size)
Y_val = d_val$y_one_hot_encoded # Classes one-hot-encoded
y_val = d_val$y_label

## Test data
d_test = load_data(base_path = 'Projects/deep-learning-detection/skin-lesions-dummy', type='test',
              image_size = image_size)
Y_test = d_test$y_one_hot_encoded # Classes one-hot-encoded
y_test = d_test$y_label


# Sanity check ----
idx = 10
grid.raster(X_train[idx,,,]/255) #Check an image 
as.numeric(y_train[idx])
summary(X_train) 

# 1) Working with a pre-trained VGG net ----

## Load base model weights ----
base_model <- application_vgg16(weights = "imagenet",
                                include_top = FALSE,
                                input_shape = c(image_size, image_size, 3L))
summary(base_model)

# i.e. freeze all convolutional VGG16 layers
freeze_weights(base_model)

## 1) Random forest as baseline model

# Define model
features = base_model$output %>% layer_flatten()
model_feature <- keras_model(
  inputs = base_model$input, 
  outputs = features)
summary(model_feature)

# Compute new feature space
X_train_features = model_feature(X_train)
X_val_features = model_feature(X_val)
X_test_features = model_feature(X_test)

# Train random forest classifier
df_train = data.frame(
  y = y_train, # Factor with class definitions
  as.matrix(X_train_features)
)
df_test = data.frame(
  y = y_test, 
  as.matrix(X_test_features)
)

# Train random forest on baseline features
rf = randomForest(as.factor(y) ~ ., df_train)

# Predict class labels
pred_rf = predict(rf, df_test)

## Calculate prediction accuracy
acc_rf = mean(pred_rf == y_test)

# 2) Transfer learned VGG with new head ----

## Train a new head based on melanoma data
predictions <- base_model$output %>% 
  layer_flatten() %>% 
  layer_dense(units = 100, activation = "relu") %>% 
  layer_dropout(rate = 0.2) %>%
  layer_dense(units = 3, activation = "softmax")

## Define model
transfer_model <- keras_model(inputs = base_model$input, outputs = predictions)
summary(transfer_model)

## compile model
transfer_model %>% compile(
  optimizer = "adam",
  loss = "categorical_crossentropy",
  metrics = c("accuracy")
)

## Train 
transfer_history <- transfer_model %>% fit(
  X_train, Y_train,
  epochs = 20,
  batch_size=128,
  verbose=0,
  validation_data = list(X_val, Y_val),
  )

## Show results
transfer_accuracy <- model %>% evaluate(X_test, Y_test, verbose=0)

# 3) Train CNN from scratch ----

batch_size = 128
nb_classes = 3
img_rows = size
img_cols = size
kernel_size = c(3, 3)
input_shape = c(img_rows, img_cols, 3)
pool_size = c(2, 2)

## Define model 
scratch_model <- keras_model_sequential() %>% 
  # convolutional part of the CNN
  layer_conv_2d(filters = 16, kernel_size = kernel_size, padding = "same",
                activation = "relu",input_shape = input_shape) %>% 
  layer_conv_2d(filters = 16, kernel_size = kernel_size, padding = "same",
                activation = "relu") %>% 
  layer_max_pooling_2d(pool_size = pool_size) %>%
  
  layer_conv_2d(filters = 32, kernel_size = kernel_size, padding = "same",
                activation = "relu") %>% 
  layer_conv_2d(filters = 32, kernel_size = kernel_size, padding = "same",
                activation = "relu") %>% 
  layer_max_pooling_2d(pool_size = pool_size) %>%
  
  layer_conv_2d(filters = 64, kernel_size = kernel_size, padding = "same",
                activation = "relu") %>% 
  layer_conv_2d(filters = 46, kernel_size = kernel_size, padding = "same",
                activation = "relu") %>% 
  layer_max_pooling_2d(pool_size = pool_size) %>%
  
  
  layer_flatten() %>% 
  
  #fully connected part of the CNN
  layer_dense(units = 100, activation = "relu") %>% 
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 100, activation = "relu") %>% 
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = nb_classes, activation = "softmax")

summary(scratch_model)

# compile model and intitialize weights
scratch_model %>% compile(
  optimizer = "adam",
  loss = "categorical_crossentropy",
  metrics = c("accuracy")
)

# train the model (can take 2-3 minuntes)
scratch_history <- scratch_model %>% fit(
  X_train, Y_train, 
  epochs = 30, batch_size=batch_size,
  verbose=0, validation_data = list(X_val, Y_val))
plot(scratch_history)
  
## Show model accuracy
scratch_accuracy <- scratch_model %>% evaluate(X_test, Y_test, verbose=0)

# 4) Identify miss-labeled images
predictions <- scratch_model %>%
  predict(X_test)  %>% 
  k_argmax() + 1 # Index shift python vs r

predictions <- as.array(predictions)
  
### Sanity check: Accuracy
(mean(as.integer(y_test) == predictions))

miss_classified <- as.integer(y_test) != predictions

## Get wrongly assigned images
X_test_miss <- X_test[miss_classified,,,]
y_test_miss <- y_test[miss_classified]

## Miss-classified fraction per class
table(y_test_miss) /table(y_test)

## Visualize sample images
idx <- 1
grid.raster(X_test_miss[idx,,,]/255, name=y_test_miss[idx]) #Check an image 
