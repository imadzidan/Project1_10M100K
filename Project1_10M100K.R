---
title: "Project1_10M100K"
author: "I Zidan"
date: "15/04/2020"
---

  
library(dplyr)
library(ggplot2)

knitr::opts_chunk$set(echo = TRUE)

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

# use .RDATA files if they exists to save time. I found that if you run the data data extraction routine through rStudio
# couple of records are missing in the edx dataset. place the .RDATA files in the the below dirctory to save time.
#Please download files using the below links. Place them in the same directory as the script.

# validation file : https://drive.google.com/open?id=1vJDE-26kh5ioWO7QX88Y-xoeRx4i7u8Q
# edx file: https://drive.google.com/open?id=15GqskACjQXbd-xyI-kPnQoSWi1vbZZSF

getwd()

# set the path to where the dataset files exist
setwd(str_c(getwd(),"/"))

edxFile <- "edx.RData"

if (!file.exists(edxFile)) {
  # MovieLens 10M dataset:
  # https://grouplens.org/datasets/movielens/10m/
  # http://files.grouplens.org/datasets/movielens/ml-10m.zip
  
  dl <- tempfile()
  download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)
  
  ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                   col.names = c("userId", "movieId", "rating", "timestamp"))
  
  movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
  colnames(movies) <- c("movieId", "title", "genres")
  movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                             title = as.character(title),
                                             genres = as.character(genres))
  
  movielens <- left_join(ratings, movies, by = "movieId")
  
  # Validation set will be 10% of MovieLens data
  set.seed(1)
  # if using R 3.5 or earlier, use `set.seed(1)` instead
  test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
  edx <- movielens[-test_index,]
  temp <- movielens[test_index,]
  
  # Make sure userId and movieId in validation set are also in edx set
  validation <- temp %>% 
    semi_join(edx, by = "movieId") %>%
    semi_join(edx, by = "userId")
  
  # Add rows removed from validation set back into edx set
  removed <- anti_join(temp, validation)
  edx <- rbind(edx, removed)
  
  rm(dl, ratings, movies, test_index, temp, movielens, removed)
} else {
  
  load(file = "edx.RData")
  load(file = "validation.RData")
}




# analyse contents
# data overview in a snaphot
summary(edx)


# Ploting rating by movie.
edx %>% 
  count(movieId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 30, color = "black") + 
  scale_x_log10() + 
  ggtitle("Movies")

# Ploting rating by users
edx %>%
     count(userId) %>% 
     ggplot(aes(n)) + 
     geom_histogram(bins = 30, color = "black") + 
     scale_x_log10() +
     ggtitle("Users")

# partition dataset. 30% of the data goes to test and the rest for training
rating_test_index <- createDataPartition(y = edx$rating, times = 1,  p = 0.3, list = FALSE)
rating_train_set <- edx[-rating_test_index,]
rating_test_set <- edx[rating_test_index,]



# function to set current dataset to use
WhichDataSet <- function(partition) {
  
  if (partition == "train") {
    rating_train_set
  } else if (partition == "test") {
    rating_test_set
  } else if (partition == "validation") {
    validation
  }
  
}


# Root Mean Square Error
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

# Set the dataset to training
dSet <- WhichDataSet("train")

str_c("there are ",nrow(dSet)," records in the training dataset")

# The mean of data set
mu <- mean(rating_train_set$rating) 

# tuning factors sequence vector
lambdas <- seq(0, 5, 0.1)


# run algorithim and produce RMSE for each lamda and store in a vector
rmses_mu <- sapply(lambdas, function(l){
  
  # Regularisation to estimate movies effect. 
  # Penalising large estimates that come from small samples
  low_ratings <- rating_train_set%>% 
    group_by(movieId)%>%
    summarize(low_ratings = sum(rating - mu)/(n()+l))
  
  
  #Regularisation to estimate user effect.
  #Penalising large estimates that come from small samples
  scaled_mean <- rating_train_set %>% 
    left_join(low_ratings, by="movieId") %>%
    group_by(userId) %>%
    summarize(scaled_mean = sum(rating - low_ratings - mu)/(n()+l))  
  
  
  # extract penalty value lambdas
  predicted_ratings <- 
    dSet %>% 
    left_join(low_ratings, by = "movieId") %>%
    left_join(scaled_mean, by = "userId") %>%
    mutate(pred = mu + low_ratings + scaled_mean) %>%.$pred
  
  return(RMSE(dSet$rating,predicted_ratings))
  
})


# Cross validation of lambda tuning parameter.
plot(lambdas, rmses_mu)


# output the best lambda
min_lambda <- lambdas[which.min(rmses_mu)]
str_c('Smallest RMSE: ', min(rmses_mu), ' by lambda: ', min_lambda)

#store the result in a tibble
rmse_results <- tibble(method = "Result from traing dataset", RMSE = min(rmses_mu))

# use test dataset
dSet <- WhichDataSet("test")

str_c("there are ",nrow(dSet)," records in the test dataset")

# plot cross validation with the test dataset
plot(lambdas, rmses_mu)


# output the best lambda when test dataset is used
min_lambda <- lambdas[which.min(rmses_mu)]
str_c('Smallest RMSE: ', min(rmses_mu), ' by lambda: ', min_lambda)

#store the result in a tibble
rmse_results <- bind_rows(rmse_results,tibble(method = "Result from test dataset", RMSE = min(rmses_mu)))

#assign lambda to the smalles value found
lambdas <- min(rmses_mu)

# Use validation dataset for final prediction
dSet <- WhichDataSet("validation")
str_c("there are ",nrow(dSet)," records in the validation dataset")

# run algorithim against the validation dataset
val_rmse <- rmses_mu

# Store the result in a tibble
rmse_results <- bind_rows(rmse_results,tibble(method = "Result from validation dataset", RMSE = min(val_rmse)))


# Output the tibble of all results
rmse_results
