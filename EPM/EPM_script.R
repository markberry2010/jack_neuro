require(ggplot2)
require(dplyr)
require(tidyr)
require(readr)
require(reshape2)
require(caret)
require(randomForest)

#
NEURON_FILENAME <- "data/D87EPMraw.csv"
BEH_FILENAME <- "data/D87EPMbeh.csv"

# Read data from disk on csv
# Data is of shape T x 69 with each row indicating a timestamp value
colnames <- sprintf("n%d",seq(1:69))
wide_data <- read.csv(NEURON_FILENAME, col.names = colnames)


wide_data_zeroed <- wide_data
wide_data_zeroed[wide_data_zeroed < 1] = 0

# Add Timestamp column
wide_data['timestamp'] <- seq(from=0,
                         to = dim(wide_data)[1] / 10 - .1, #subtract .1 because 0 indexing
                         by = .1) 

# Add Timestamp column
wide_data_zeroed['timestamp'] <- seq(from=0,
                              to = dim(wide_data_zeroed)[1] / 10 - .1, #subtract .1 because 0 indexing
                              by = .1) 



# Convert wide data into Tidy / long datafram
long_data <- tidyr::gather(wide_data,"neuron_id",'zscore', -timestamp)
long_data_zeroed <- tidyr::gather(wide_data_zeroed,"neuron_id",'zscore', -timestamp)



plot_response <- function(data, neuron_ids, title="") {
  # Plots the response of a set of neurons
  #
  # Args:
  #     data:  (df) in long format, with timestamp, neuron, and auc columns
  #     neruon_names: (list) of neuron names
  #     title: (string) chart title
  #
  # Returns: a chart
  
  data %>%
    filter(neuron_id %in% neuron_ids) %>%
    ggplot(aes(x=timestamp, y=zscore, color=neuron_id)) +
    geom_line() + labs(title=title, x="Timestamp (s)",y="Z-Score Calcium Response Curve")
}

# Get lower triangle of the correlation matrix
get_lower_tri<-function(cormat){
  cormat[upper.tri(cormat)] <- NA
  return(cormat)
}
# Get upper triangle of the correlation matrix
get_upper_tri <- function(cormat){
  cormat[lower.tri(cormat)]<- NA
  return(cormat)
}

plot_correlation_matrix <- function(data, title=""){
  # Plots correlation matrix of a dataframe
  #
  # Args: 
  #     data: (df)
  #     title: (str)
  
  cormat <- data %>% select(-timestamp) %>% cor()
  # turn upper triangle to N/A
  upper_tri <- get_upper_tri(cormat)
  # Melt correlation matrix to long format for use in ggplot
  melted_cormat <- melt(upper_tri)
  ggplot(data = melted_cormat, aes(Var2, Var1, fill = value))+
    geom_tile(color = "white")+
    scale_fill_gradient2(low = "blue", high = "red", mid = "white", 
                         midpoint = 0, limit = c(-1,1), space = "Lab", 
                         name="Pearson\nCorrelation", na.value = "white") + 
    theme(panel.grid.major = element_blank(), 
          panel.grid.minor = element_blank(),
            panel.background = element_blank(),
          axis.text.x = element_text(angle = 90, hjust = 1)) + 
  labs(title=title,x="Neuron 1", y="Neuron 1")
  
}

# Add lagged features to dataframe
# Args: 
#     -data: dataframe
#     -columns: vector of colnames
#     -lags: vector of lags to include
#
# Example lagged_df <- add_lags(df, c("n1","n2"), 1:3);
#
# TODO: Vectorize this fxn
add_lags <- function(data, columns, lags, drop.na=TRUE){
  for (col in columns){
    for (lag in lags){
        data[paste(col, "_lag_", lag,sep='')] <- lag(data[[col]], n=lag)
    }
  }
  if (drop.na){
    return(drop_na(data))
  }
  else{
    return(data)
  }
}

wide_data_lagged <- add_lags(wide_data_zeroed, colnames, 1:4)

plot_correlation_matrix(wide_data, "Correlations Non Zeroed")
plot_correlation_matrix(wide_data_zeroed, "Coreelations Zeroed < 1")




# Find most correlated neurons
most_correlated_neurons <- function(data, n=5){
  top_pairs <- data %>% 
            select(-timestamp) %>%
            cor() %>%
            get_upper_tri() %>%
            melt() %>%
            filter(Var1 != Var2) %>%
            top_n(n, value) %>%
            rename(neuron1=Var1, neuron2=Var2) %>% 
            arrange(desc(value))
            
  return(top_pairs)
}
most_correlated <- most_correlated_neurons(wide_data_zeroed)
neuron_ids = c(as.matrix(most_correlated[0:2,1]))
plot_response(long_data,neuron_ids)

# Plots the response curve of various pricinpal components
# Arguments: -wide_data: neural response dataframe
#            -pc_to_plot: vector containing names of which principal
#                         components to plot
plot_pcs <- function(wide_data, pc_to_plot, n=10, title="") {
  pcs <- get_pcs(wide_data,n)
  long_pcs <- pcs %>% data.frame() %>%
              gather("PC","Value", -timestamp) 
  long_pcs %>% filter(PC %in% pc_to_plot) %>% 
        ggplot(aes(x=timestamp, y=Value,color=PC)) + geom_line() +
    labs(title=title,x="Timestamp (s)", y="Neural Activation or Something")
  #return(long_pcs)
}

# Calculates principal components
# 
get_pcs <- function(wide_data, n_components){
  pcs <- wide_data %>% select(-timestamp) %>% scale %>% prcomp()
  pcs_with_time <- cbind(pcs$x[,1:n_components], wide_data$timestamp)
  colnames(pcs_with_time)[length(colnames(pcs_with_time))] <- "timestamp"
  return(data.frame(pcs_with_time))
}

###
###   Cluster Section
###

clusters <- wide_data_zeroed[,1:69] %>% t() %>% kmeans( 10)
cluster_ids <- clusters$cluster  %>% data.frame() 
cluster_ids$neuron_id <- rownames(cluster_ids)
colnames(cluster_ids) <- c("cluster_id", "neuron_id")

long_data_wclusters <- inner_join(long_data, cluster_ids, by ="neuron_id")

cluster_means <- long_data_wclusters %>% group_by(timestamp, cluster_id) %>%
                  summarise(mean_zscore=mean(zscore), std= sd(zscore))
cluster_means %>% filter(cluster_id<4) %>% ggplot(aes(x=timestamp, y=mean_zscore, color=factor(cluster_id))) + geom_line()
cluster_means_wide <- cluster_means %>% select(-std) %>% spread(cluster_id,mean_zscore)

#########

## Joining Beh data
beh_colnames <- c("trial_time",	"recording_time","X_center","Y_center",
                  "area","areachange","elongation","distance_moved","velocity",
                  "arena", "open1","open2","closed1","closed2","open_arms",
                  "closed_arms","result1")


beh_raw <- read.csv(BEH_FILENAME, col.names = beh_colnames)
# Downsampling behav data due to differences in freq
beh <- beh_raw[seq(from=1,
                   to=dim(beh_raw)[1],
                   by=3) # downsample by 3
               ,]

# we use a integer column to allow for a successful join.
beh$timestamp_int<- round(beh$trial_time * 10 )
beh <- beh %>% select(c("timestamp_int",	"X_center","Y_center","velocity",
                            "open1","open2","closed1","closed2","open_arms",
                             "closed_arms")) %>%
            mutate(maze_location = case_when(open_arms==1 ~ 'open',
                                    closed_arms==1~'closed',
                                    closed_arms + open_arms==0 ~ 'center'))

merged <- wide_data_zeroed %>% 
              mutate(timestamp_int = round(timestamp *10))  %>%
              left_join(beh, by="timestamp_int") %>%
              select(-timestamp_int)

merged_lagged <- wide_data_lagged %>% 
                    mutate(timestamp_int = round(timestamp *10))  %>%
                    left_join(beh, by="timestamp_int") %>%
                    select(-timestamp_int)


pca_merged <-   wide_data_lagged %>%
                    get_pcs(n_components=30) %>%
                    mutate(timestamp_int = round(timestamp *10))  %>%
                    left_join(beh, by="timestamp_int") %>%
                    select(-timestamp_int)

#clusters_merged <- inner_join(cluster_means_wide,beh, by="timestamp")
#plot_correlation_matrix(clusters_merged)

###########
#  Model Predictions
###########
predictive_model <- function(data, model_fxn, train_period=60, train_modulo=2,
                             regex="^n\\d|maze_location"){
  
  train_index <- round(data$timestamp  / train_period) %% train_modulo == 0 
  train <- data[train_index,] %>% select(matches(regex))
  test <- data[!train_index,] %>% select(matches(regex))
  set.seed(47)

  
  
  # RandomForest predictions
  model <- model_fxn(factor(maze_location) ~., data=train)
  test$predictions <- predict(model,test)
  confusion <- confusionMatrix(data=test$predictions, test$maze_location)
  return(confusion)
}







# Keep every other 30 second window as a holdout
train_index <- round(merged_lagged$timestamp  / 30) %% 2 == 0 
train <- merged_lagged[train_index,]
test <- merged_lagged[!train_index,]
set.seed(47)

# Select columns for use in predictive models
maze_location.train <- train %>% select(matches("^n\\d|maze_location"))
maze_location.test <- test %>% select(matches("^n\\d|maze_location"))

open_arms.train <- train %>% select(matches("^n\\d|open_arms"))
open_arms.test <- test %>% select(matches("^n\\d|open_arms"))


# RandomForest predictions
rf <- randomForest(factor(maze_location) ~., data=maze_location.train)
rf_preds <- predict(rf,maze_location.test)
confusionMatrix(data=rf_preds, maze_location.test$maze_location)

# Logisitic Regression predictions
#
glm <- glm(factor(open_arms) ~., data=open_arms.train, family='binomial')
glm_preds <- round(predict(glm,open_arms.test, type='response'))
confusionMatrix(data=glm_preds, open_arms.test$open_arms)

#############
# principal components regression
############
pc_confusion <- predictive_model(pca_merged, randomForest, regex="^PC\\d|maze_location")



#####################################################
# Random Forest feature importance ---- speculative...
rf_plots <- function(data, y_variable, cluster_ids) {
  data <- data %>% 
           select(matches(paste("^n\\d|",y_variable, sep="")))
  
  # Fit RandomForest Model
  formula <- paste(y_variable,"~ .",sep='')
  rf <- randomForest(as.formula(formula),
                     data = data)
  # Select Feature Importnances
  rf.importance <- rf$importance %>% data.frame()
  rf.importance$neuron_id <- rownames(rf.importance)
  rf.importance <- inner_join(rf.importance, cluster_ids, by="neuron_id")
  rf.importance %>% group_by(cluster_id) %>% summarise(mean_importance = mean(MeanDecreaseGini)) %>% 
    ggplot(aes(x=cluster_id,y=mean_importance)) + geom_col()
}


rf <- randomForest(open_arms ~., data=merged %>% select(matches("^n\\d|open_arms")) )
rf.importance <- rf$importance %>% data.frame()
rf.importance$neuron_id <- rownames(rf.importance)
rf.importance <- inner_join(rf.importance, cluster_ids, by="neuron_id")
rf.importance %>% group_by(cluster_id) %>% summarise(mean_importance = mean(MeanDecreaseGini)) %>% 
  ggplot(aes(x=cluster_id,y=mean_importance)) + geom_col()



rf.importance %>% arrange(desc(MeanDecreaseGini)) %>%
  ggplot(aes(reorder(neuron_id,desc(MeanDecreaseGini)), MeanDecreaseGini, fill=factor(cluster_id))) +
  geom_col() +
  theme(panel.grid.major = element_blank(), 
       panel.grid.minor = element_blank(),
       panel.background = element_blank(),
       axis.text.x = element_text(angle = 90, hjust = 1))+
  labs(title='Neuron Importance for Open Arms Prediction', x='NeuronId', y='Importance (Mean Decrease Gini)')


#######
long_data_zeroed %>%
  inner_join(beh, by="timestamp") %>%
  filter(neuron_id =='n31') %>%
  ggplot(aes(timestamp, zscore)) + geom_line() +

##################
#### Bar Plot 
#################
long_data_zeroed %>%
  mutate(timestamp_int = round(timestamp*10)) %>%
  inner_join(beh, by="timestamp_int") %>%
  group_by(maze_location) %>%
  summarise(mean_zscore=mean(zscore), se = sd(zscore)/sqrt(n())) %>%
  ggplot(aes(maze_location,mean_zscore,fill=maze_location)) +
  geom_bar(stat='identity')  +
  # Error Bars 
  geom_errorbar(aes(ymin=mean_zscore-se,ymax=mean_zscore + se), width=.2) +
  labs(title='Activation by Arm Status',y='Mean Activation Z Score', x='Arm Status') +
  theme(panel.grid.major = element_blank(), 
        panel.grid.minor = element_blank(),
        panel.background = element_blank(),
        legend.position="none") 


