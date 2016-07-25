library(dplyr)
library(caret)
library(rpart)

set.seed(1)
setwd('~/articles/20160717-decision-trees/') # set the path to your work directory
loans <- read.csv('LoanStats3a.csv', stringsAsFactors = F, skip = 1)
safe_loans  <- loans[loans$loan_status == 'Fully Paid',]
risky_loans <- loans[loans$loan_status == 'Charged Off',]
safe_loans$safe  <- 1
risky_loans$safe <- 0

dim(safe_loans)
dim(risky_loans)

safe_loans <- safe_loans[1:nrow(risky_loans),]

loans <- select(rbind(safe_loans, risky_loans), grade, term, home_ownership, emp_length, safe)
loans.data <- data.frame(safe = loans$safe)
loans.data <- cbind(loans.data, model.matrix(~loans$grade - 1))
loans.data <- cbind(loans.data, model.matrix(~loans$term - 1))
loans.data <- cbind(loans.data, model.matrix(~loans$home_ownership - 1))
loans.data <- cbind(loans.data, model.matrix(~loans$emp_length - 1))

inTrain <- createDataPartition(loans.data$safe, p = 0.5, list = F)
train_data = loans.data[inTrain,] 
test_data  = loans.data[-inTrain,]

calculate_error <- function (labels) {
  # if we use a label of majority as a label of the leaf
  # the minority is the error of the leaf
  min(sum(labels == 1), sum(labels == 0))
}  

splitting_feature <- function (data, features, target) {
  # we calculate error of splitting for every feature
  errors <- sapply(
    features, 
    function (feature) {
      left_labels  <- data[data[feature] == 0,][[target]]
      right_labels <- data[data[feature] == 1,][[target]]
      left_error  <- calculate_error(left_labels)            
      right_error <- calculate_error(right_labels)
      error = (left_error + right_error) / length(data)
    }
  )
  # and take the feature with the lowest error
  names(which.min(errors))
}

create_leaf <- function (labels) {
  # we will use a label of majority as a label of the leaf
  positive = sum(labels == 1)
  negative = sum(labels == 0)
  
  list(
    splitting_feature = NULL,
    left              = NULL,
    right             = NULL,
    is_leaf           = TRUE,
    prediction        = ifelse(positive > negative, 1, 0)
  )
}

mytree_create <- function(data, features, target, current_depth = 0, max_depth = 10){
  remaining_features = features
  
  target_values = data[[target]]
  
  # If all datapoints in current node have the same label
  if (calculate_error(target_values) == 0) {
    return(create_leaf(target_values))
  }
  
  # If we've used all features to consider a split already
  if (length(remaining_features) == 0) {
    return(create_leaf(target_values))    
  }
  
  # If we reached pre-defined maximum depth of the tree
  if (current_depth >= max_depth) {
    return(create_leaf(target_values))
  }
  
  # we will find the feature to make a split
  splitting_feature = splitting_feature(data, remaining_features, target)
  
  # and make a split 
  left_split  = data[data[splitting_feature] == 0,]
  right_split = data[data[splitting_feature] == 1,]
  
  # we remove feature we just used
  remaining_features = remaining_features[-which(remaining_features == splitting_feature)]
  
  # if one of the subsets is the original set, 
  # create a leaf using this set
  if (nrow(left_split) == nrow(data)) {
    return(create_leaf(left_split[[target]]))
  }
  if (nrow(right_split) == nrow(data)) {
    return(create_leaf(right_split[[target]]))
  }
  
  # Keep splitting
  left_tree  = mytree_create(left_split, remaining_features, target, current_depth + 1, max_depth) 
  right_tree = mytree_create(right_split, remaining_features, target, current_depth + 1, max_depth)        
  
  # return the tree 
  list(
    is_leaf = FALSE, 
    prediction = NULL,
    splitting_feature = splitting_feature,
    left = left_tree, 
    right = right_tree
  )
}

mytree_predict <- function (tree, data) {   
  # return prediction if the node is a leaf node.
  if (tree$is_leaf) {
    return(tree$prediction)
  }
  
  # split on feature.
  split_feature_value = data[[tree$splitting_feature]]
  
  # keep splitting down the tree
  ifelse(
    split_feature_value == 0,
    mytree_predict(tree$left, data),
    mytree_predict(tree$right, data)
  )
}

# train rpart tree on training data 
rpart_tree <- rpart(safe ~ ., train_data, method = 'class')

# predict labels for test data using rpart tree
rpart_pred <- predict(rpart_tree, test_data, type = 'class')

# compare labels predicted by rpart to real ones
confusionMatrix(rpart_pred, test_data$safe)

# train my tree on training data
my_decision_tree <- mytree_create(train_data, colnames(train_data[-1]), 'safe', max_depth = 6)

# predict labels for test data using my tree
pred <- mytree_predict(my_decision_tree, test_data)

# compare labels predicted by my tree to real ones
confusionMatrix(pred, test_data$safe)