library(MASS)
library(caret)
library(randomForest)
library(gbm)
library(ggplot2)
library(rpart)
library(reshape2)
library(rpart.plot)
library(tidyverse)

rest_rev <- read.csv("C:/Users/Lenovo/Downloads/restaurant.csv")
View(rest_rev)

###########################################################################
# Data cleanup and Feature engineering
###########################################################################

summary(rest_rev)

#Creating a new feature review per customer
rest_rev$Review_per_customer = rest_rev$Reviews/rest_rev$Number_of_Customers 

#Flooring negative values which were present in target variable
rest_rev[rest_rev<0] <- 0

#setting categorical variables as factors
rest_rev$Promotions=factor(rest_rev$Promotions,levels = c(1,0))
rest_rev$Cuisine_Type=factor(rest_rev$Cuisine_Type,levels = c("Japanese","Italian","American","Mexican"))


pairs(Monthly_Revenue~ Number_of_Customers+Menu_Price+Marketing_Spend+Average_Customer_Spending+Reviews+Review_per_customer, data=rest_rev)

###########################################################################
# Defining Train and Test data sets
###########################################################################

set.seed(18)

# Hold out 20% of the data as a final validation set
train_index = createDataPartition(rest_rev$Monthly_Revenue,p = 0.8)

train_data = rest_rev[train_index$Resample1,]
test_data  = rest_rev[-train_index$Resample1,]

###########################################################################
# Setting up cross-validation
###########################################################################

# Number of folds
kcv <- 10

fit_control <- trainControl(
  method = "cv",
  selectionFunction="oneSE")

###########################################################################
# Linear Regression
###########################################################################

#MLR on train data set and all independent variables
set.seed(18)

result <- lm(Monthly_Revenue~.,data=train_data)
summary(result)

#MLR on train data set and all independent variables and log of dependent variable
#log_train_data <- train_data
#log_train_data$Log_Monthly_Rev <- ifelse(log_train_data$Monthly_Revenue == 0,0,log(log_train_data$Monthly_Revenue))
#result_log <- lm(train_data$Log_Monthly_Rev~.,data=train_data)
#summary(result_log)


#Stepwise regression & feature selection
stepwise_interaction = step(lm(Monthly_Revenue~(.)^2, data=train_data),
                     direction="both",
                     scope = ~.)
summary(stepwise_interaction)

ggplot(aes(y=Monthly_Revenue, x = Number_of_Customers),
       data=train_data) +  geom_point() + geom_smooth(method='lm')
###########################################################################
# Single tree
###########################################################################

set.seed(18)

bigtree = rpart(Monthly_Revenue~.,data=train_data)
plotcp(bigtree)

best_cp_index = which.min(bigtree$cptable[,4]) 
bigtree$cptable[best_cp_index,]

tol_error = bigtree$cptable[best_cp_ix,4] + bigtree$cptable[best_cp_ix,5]
bigtree$cptable[bigtree$cptable[,4]<tol_error,][1,]
best_cp_onesd = bigtree$cptable[bigtree$cptable[,4]<tol,][1,1]
best_cp_onesd

custom_var_names <- c("Number_of_Customers" = "No.of Customers", "Menu_Price" = "Menu Price", "Marketing_Spend" = "Marketing Spend", "Monthly_Revenue" = "Monthly Revenue")

custom_node_fun <- function(x, labs, digits, varlen) {
  for (var in names(custom_var_names)) {
    labs <- gsub(var, custom_var_names[[var]], labs)
  }
  return(labs)
}

cvtree = prune(bigtree, cp=best_cp_onesd)
prp(cvtree, nn = TRUE, faclen = 0, varlen = 0, node.fun = custom_node_fun, cex = 0.7)

###########################################################################
# Bagging
###########################################################################

set.seed(18)
bagging_model <- randomForest(Monthly_Revenue ~., data = train_data, coob = TRUE, mtry = 8)
print(bagging_model)
# Predicted values
oob_predictions <- bagging_model$predicted
# Actual values
oob_actuals <- train_data$Monthly_Revenue
# Calculate OOB error
oob_error <- sqrt(mean((oob_predictions - oob_actuals)^2))
print(paste("OOB Error (RMSE):", round(oob_error, 4)))
plot(bagging_model$mse, type = "l", col = "blue",
     xlab = "Number of Trees", ylab = "OOB Error (MSE)",
     main = "OOB Error Rate vs. Number of Trees")


###########################################################################
# Random forests
###########################################################################

set.seed(18)

rf_grid = data.frame(mtry = c(2,3,4,6,7,8))
rf_fit <- train( Monthly_Revenue ~., data = train_data, 
                 method = "rf", 
                 trControl = fit_control,
                 tuneGrid = rf_grid,
                 ntree = 250)

plot(rf_fit)
rf_fit$results

best = rf_fit$results[which.min(rf_fit$results$RMSE),]
onesd = best$RMSE + best$RMSESD/sqrt(kcv)
onesd

ggplot(rf_fit) + 
  geom_segment(aes(x=mtry, 
                   xend=mtry, 
                   y=RMSE-RMSESD/sqrt(kcv), 
                   yend=RMSE+RMSESD/sqrt(kcv)), 
               data=rf_fit$results) + 
  geom_hline(yintercept = onesd, linetype='dotted')

rf_fit$finalModel

###########################################################################
# Boosting
###########################################################################

set.seed(18)

gbm_grid <-  expand.grid(interaction.depth = c(1,2,3), 
                         n.trees = c(300,500,1000),
                         shrinkage = c(0.001,0.01, 0.02),
                         n.minobsinnode = 10)


gbmfit <- train(Monthly_Revenue~.,data=train_data, method = "gbm", trControl = fit_control, 
                tuneGrid = gbm_grid, verbose = FALSE)

print(gbmfit)


best_ix = which.min(gbmfit$results$RMSE)
best = gbmfit$results[best_ix,]
onese_max_RMSE = best$RMSE + best$RMSESD/sqrt(kcv) 
onese_ixs = gbmfit$results$RMSE<onese_max_RMSE

print(gbmfit$results[onese_ixs,])
gbmfit$bestTune

gbmfit_rmse_min = train(Monthly_Revenue~.,data=train_data, method = "gbm", trControl = trainControl(method="none"),tuneGrid = best[,1:4],verbose = FALSE)
gbmfit_rmse_min$bestTune


gbm_plot_df = gbmfit$results
gbm_plot_df$n.trees = factor(gbm_plot_df$n.trees)
ggplot(aes(x=interaction.depth, y=RMSE, color=n.trees), 
       data=gbm_plot_df) +
  facet_grid(~shrinkage, labeller = label_both) +
  geom_point() + 
  geom_line() + 
  geom_segment(aes(x=interaction.depth, 
                   xend=interaction.depth, 
                   y=RMSE-RMSESD/sqrt(kcv), 
                   yend=RMSE+RMSESD/sqrt(kcv))) + 
  geom_hline(yintercept = onese_max_RMSE, linetype='dotted') +
  xlab("Max Tree Depth") + 
  ylab("RMSE (CV)") + 
  scale_color_discrete(name = "Num Boosting Iter") + 
  theme(legend.position="bottom") 

cor(predict(gbmfit), predict(gbmfit_rmse_min))
plot(predict(gbmfit), predict(gbmfit_rmse_min))


##################################################################
# Comparing Linear Regression, Regression tree, RF & Boosting
##################################################################

lin_yhat = predict(result,newdata=test_data)
lin_yhat_int = predict(stepwise_int2,newdata=test_data)
rt_yhat = predict(cvtree,newdata=test_data)
rf_yhat  = predict(rf_fit,newdata=test_data)
gbm_yhat = predict(gbmfit, newdata=test_data)


# Test RMSE

lin_rmse <- sqrt(mean((test_data$Monthly_Revenue - lin_yhat)^2))
lin_int_rmse <- sqrt(mean((test_data$Monthly_Revenue - lin_yhat_int)^2))
rt_rmse <- sqrt(mean((test_data$Monthly_Revenue - rt_yhat)^2))
rf_rmse <- sqrt(mean((test_data$Monthly_Revenue - rf_yhat)^2))
gbm_rmse <- sqrt(mean((test_data$Monthly_Revenue - gbm_yhat)^2))


lin_imp = varImp(result)
lin_int_imp = varImp(stepwise_int2)
rt_imp  = varImp(cvtree)
rf_imp = varImp(rf_fit)
gbm_imp  = varImp(gbmfit)

combined_df = data.frame(variable=rownames(gbm_imp$importance),
                         gbm = gbm_imp$importance$Overall,
                         rf  = rf_imp$importance$Overall)

View(combined_df)

data <- data.frame(
  Variable = c(
    "Number_of_Customers", 
    "Menu_Price", 
    "Marketing_Spend", 
    "Average_Customer_Spending", 
    "Review_per_customer", 
    "Number_of_Customers:Marketing_Spend", 
    "Average_Customer_Spending:Review_per_customer"
  ),
  Importance = c(
    16.191986, 
    11.416619, 
    4.259008, 
    1.637039, 
    1.634163, 
    1.776017, 
    1.448011
  )
)

# Plot the data
ggplot(data, aes(x = reorder(Variable, Importance), y = Importance)) +
  geom_bar(stat = "identity", fill = "darkorange") +
  coord_flip() +
  labs(
    title = "Variable Importance Scores",
    x = "Variable",
    y = "Importance Score"
  ) +
  theme_minimal()

ggplot(lin_int_imp, aes(x = reorder(variable, Importance), y = Importance)) +
  geom_bar(stat = "identity", fill = "darkorange") +  # Set bar color to dark orange
  theme_minimal() +
  labs(title = "Variable Importance",
       x = "Variable",
       y = "Importance") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))


combined_long <- melt(combined_df, id.vars = "variable", variable.name = "Model", value.name = "Importance")

# Create the horizontal grouped bar plot
ggplot(combined_long, aes(x = Importance, y = variable, fill = Model)) +
  geom_bar(stat = "identity", position = "dodge", width = 0.7) +
  theme_minimal() +
  labs(title = "Variable Importance Comparison",
       x = "Importance",
       y = "Variable") +
  theme(axis.text.y = element_text(angle = 0, hjust = 1)) +
  scale_fill_manual(values = c("gbm" = "steelblue", "rf" = "darkorange"))


#RMSE plot
rmse_values <- data.frame(
  Model = c("Linear Regression", "Regression Tree", "Random Forest", "Gradient Boosting"),
  RMSE = c(lin_rmse, rt_rmse, rf_rmse, gbm_rmse)
)

ggplot(rmse_values, aes(x = Model, y = RMSE, fill = Model)) +
  geom_bar(stat = "identity", width = 0.5) +  # Reduce the width of the bars
  geom_text(aes(label = round(RMSE, 2)), vjust = -0.3, size = 3) +  # Add values on top of the bars
  theme_minimal() +
  labs(title = "RMSE Comparison of Different Models",
       x = "Model",
       y = "RMSE") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
