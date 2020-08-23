#######################################################
###       data load
#######################################################

###download required libraries
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(dplyr)) install.packages("dplyr", repos = "http://cran.us.r-project.org")
if(!require(readr)) install.packages("readr", repos = "http://cran.us.r-project.org")

###enabling the libraries
library(tidyverse)
library(caret)
library(dplyr)
library(readr)

###download the dataset
#data originally found here : https://www.kaggle.com/rush4ratio/video-game-sales-with-ratings
#uploaded to my github repository
url <- "https://raw.githubusercontent.com/AxelComparetto/RdataSciennce_VideoGameRating/master/Video_Games_Sales_as_at_22_Dec_2016.csv"
VideoGames <- read_csv(url)
head(VideoGames)

###data cleaning
VideoGames <- na.omit(VideoGames) #clear the rows with missing values
VideoGames <- VideoGames %>% mutate(gameId = which(Name == Name)) #create a gameId column
vec_reorder <- c(17, 1:16) #a vector to place gameId as first column
VideoGames <- VideoGames %>% select(all_of(vec_reorder)) #places gameId as first column

###changing scales
#putting User_score (/10) and Critic_score(/100) on the same scale (/100)
VideoGames <- VideoGames %>% mutate(User_Score = 10*as.numeric(User_Score))
#the original dataset is in million copies, I will use the number of units instead
VideoGames <- VideoGames %>% mutate(NA_Sales=NA_Sales*10^6,
                                    EU_Sales=EU_Sales*10^6,
                                    JP_Sales=JP_Sales*10^6,
                                    Other_Sales=Other_Sales*10^6,
                                    Global_Sales=Global_Sales*10^6)

###generate training and test set
#evaluate the number of unique predictors
length(unique(VideoGames$Platform))
length(unique(VideoGames$Genre))
length(unique(VideoGames$Year_of_Release))
length(unique(VideoGames$Publisher))
length(unique(VideoGames$Developer)) #too many developers to split sets equally
length(unique(VideoGames$Rating))
#create the data partition
set.seed(1, sample.kind = "Rounding")
indexes <- createDataPartition(y=VideoGames$gameId, times=1, p=0.2, list=FALSE)
#80% training set, 20% test set
training <- VideoGames[-indexes,]
temp <- VideoGames[indexes,]
#make sure every platform/genre/YoR/Publisher/Rating in the test set is in the training set
test <- temp %>%
  semi_join(training, by = "Platform") %>%
  semi_join(training, by = "Genre") %>%
  semi_join(training, by = "Year_of_Release") %>%
  semi_join(training, by = "Publisher") %>%
  semi_join(training, by = "Rating") %>%
  semi_join(training, by = "Developer")
removed <- anti_join(temp, test)
training <- rbind(training, removed)

###free memory
rm(url, vec_reorder, temp, removed, indexes, VideoGames)



#######################################################
###       data processing
#######################################################

###looking for number of sales' scales
hist(training$Global_Sales)
hist(log10(training$Global_Sales) )

#qqplot of the sales
p <- seq(0.01, 0.99, 0.01) 
sample_quantiles <- quantile(log10(training$Global_Sales), p)
theoretical_quantiles <- qnorm(p, mean(log10(training$Global_Sales),
                                       sd(log10(training$Global_Sales))))
qplot(theoretical_quantiles, sample_quantiles) + geom_abline()
rm(p, sample_quantiles, theoretical_quantiles)

#making new column for the log transform
training <- training %>% mutate(logGlobal_Sales=log10(Global_Sales))



#######################################################
###       worldwide: naive approach
#######################################################

#defining how to compute the RMSE
fun_RMSE <- function(actual_values, predictions)
{
  percent_error = (actual_values-predictions)/actual_values
  sqrt(mean(percent_error^2))
}

log10_mu <- mean(training$logGlobal_Sales) #overall mean of the log10 sales worldwide

#making naive predictions: every game sold the average number of copies
pred = 10^log10_mu
RMSE <- fun_RMSE(test$Global_Sales, pred)

#creating a matrix to keep track of successive RMSEs
mat_RMSE <- matrix(data = c("Worldwide naive approach: average", RMSE), nrow = 1, ncol = 2)
colnames(mat_RMSE) <- c("Method", "RMSE")
mat_RMSE


#######################################################
### worldwide: bias evaluation: publisher, genre, platform, year of release and rating
#######################################################

###publisher bias
bias_pub <- training %>%
  group_by(Publisher) %>%
  summarise(b_pu  = mean(logGlobal_Sales - log10_mu),
            count_pu = n(),
            sum_bpulog = sum(logGlobal_Sales - log10_mu),
            .groups="drop")
bias_pub %>%
  ggplot(aes(Publisher, b_pu)) +
  geom_col() +
  xlab("Publisher") +
  ylab("Publisher bias (log scale)") +
  ggtitle("Effect of the publisher on global sales")
mean(bias_pub$b_pu)
sd(bias_pub$b_pu)

###platform bias
bias_plat <- training %>%
  left_join(bias_pub, by="Publisher") %>%
  group_by(Platform) %>%
  summarise(b_pla  = mean(logGlobal_Sales -log10_mu - b_pu),
            count_pla = n(),
            sum_bplalog = sum(logGlobal_Sales - log10_mu - b_pu),
            .groups="drop") 
bias_plat %>%
  ggplot(aes(Platform, b_pla)) +
  geom_col() +
  xlab("Platform") +
  ylab("Platform bias (log scale)") +
  ggtitle("Effect of the platform on global sales")
mean(bias_plat$b_pla)
sd(bias_plat$b_pla)

###genre bias
bias_genre <- training %>%
  left_join(bias_pub, by="Publisher") %>%
  left_join(bias_plat, by="Platform") %>%
  group_by(Genre) %>%
  summarise(b_ge  = mean(logGlobal_Sales - log10_mu - b_pu - b_pla),
            count_ge = n(),
            sum_bgelog = sum(logGlobal_Sales - log10_mu - b_pu - b_pla),
            .groups="drop")
bias_genre %>%
  ggplot(aes(Genre, b_ge)) +
  geom_col() +
  xlab("Genre") +
  ylab("Genre bias (log scale)") +
  ggtitle("Effect of the genre on global sales")
mean(bias_genre$b_ge)
sd(bias_genre$b_ge)

###year of release bias
bias_yor <- training %>%
  left_join(bias_pub, by="Publisher") %>%
  left_join(bias_plat, by="Platform") %>%
  left_join(bias_genre, by="Genre") %>%
  group_by(Year_of_Release) %>%
  summarise(b_yor  = mean(logGlobal_Sales - log10_mu - b_pu - b_pla - b_ge),
            count_yor = n(),
            sum_byorlog = sum(logGlobal_Sales - log10_mu - b_pu - b_pla - b_ge),
            .groups="drop")
bias_yor %>%
  ggplot(aes(Year_of_Release, b_yor)) +
  geom_col() +
  xlab("Release year") +
  ylab("Release year bias (log scale)") +
  ggtitle("Effect of the year of release on global sales")
mean(bias_yor$b_yor)
sd(bias_yor$b_yor)

###rating bias
bias_rat <- training %>%
  left_join(bias_pub, by="Publisher") %>%
  left_join(bias_plat, by="Platform") %>%
  left_join(bias_genre, by="Genre") %>%
  left_join(bias_yor, by="Year_of_Release") %>%
  group_by(Rating) %>%
  summarise(b_rat  = mean(logGlobal_Sales - log10_mu - b_pu - b_pla - b_ge - b_yor),
            count_rat = n(),
            sum_bratlog = sum(logGlobal_Sales - log10_mu - b_pu - b_pla - b_ge - b_yor),
            .groups="drop")
bias_rat %>%
  ggplot(aes(Rating, b_rat)) +
  geom_col() +
  xlab("Rating") +
  ylab("Rating bias (log scale)") +
  ggtitle("Effect of rating on global sales")
mean(bias_rat$b_rat)
sd(bias_rat$b_rat)

###developer bias
bias_dev <- training %>%
  left_join(bias_pub, by="Publisher") %>%
  left_join(bias_plat, by="Platform") %>%
  left_join(bias_genre, by="Genre") %>%
  left_join(bias_yor, by="Year_of_Release") %>%
  left_join(bias_rat, by="Rating") %>%
  group_by(Developer) %>%
  summarise(b_dev  = mean(logGlobal_Sales - log10_mu - b_pu - b_pla - b_ge - b_yor - b_rat),
            count_dev = n(),
            sum_bdevlog = sum(logGlobal_Sales - log10_mu - b_pu - b_pla - b_ge - b_yor - b_rat),
            .groups="drop")
bias_dev %>%
  ggplot(aes(Developer, b_dev)) +
  geom_col() +
  xlab("Developer") +
  ylab("Developer bias (log scale)") +
  ggtitle("Effect of developer on global sales")
mean(bias_dev$b_dev)
sd(bias_dev$b_dev)

###computes the bias fro each row of the test set
vec_bias_pub <- as.numeric(sapply(test$Publisher, function(x)
  bias_pub$b_pu[which(bias_pub$Publisher == x)]))
vec_bias_pla <- as.numeric(sapply(test$Platform, function(x)
  bias_plat$b_pla[which(bias_plat$Platform == x)]))
vec_bias_ge <- as.numeric(sapply(test$Genre, function(x)
  bias_genre$b_ge[which(bias_genre$Genre == x)]))
vec_bias_yor <- as.numeric(sapply(test$Year_of_Release, function(x)
  bias_yor$b_yor[which(bias_yor$Year_of_Release == x)]))
vec_bias_rat <- as.numeric(sapply(test$Rating, function(x)
  bias_rat$b_rat[which(bias_rat$Rating == x)]))
vec_bias_dev <- as.numeric(sapply(test$Developer, function(x)
  bias_dev$b_dev[which(bias_dev$Developer == x)]))

#using the four bias to make predictions
pred = 10^(log10_mu + vec_bias_pub + vec_bias_pla + vec_bias_ge +
             vec_bias_yor + vec_bias_rat + vec_bias_dev)
RMSE <- fun_RMSE(test$Global_Sales, pred)
#keep track of RMSE
mat_RMSE <- rbind(mat_RMSE,
  c("LM: Worldwide with environmental bias (publisher, platform, genre, YoR, rating and dev)", RMSE))
mat_RMSE



#######################################################
###   worldwide: regularizing environmental bias
#######################################################

###computes the bias for the rows of the training set before regularizing
t_v_b_pub <- as.numeric(sapply(training$Publisher, function(x)
  bias_pub$b_pu[which(bias_pub$Publisher == x)]))
t_v_b_pla <- as.numeric(sapply(training$Platform, function(x)
  bias_plat$b_pla[which(bias_plat$Platform == x)]))
t_v_b_ge <- as.numeric(sapply(training$Genre, function(x)
  bias_genre$b_ge[which(bias_genre$Genre == x)]))
t_v_b_yor <- as.numeric(sapply(training$Year_of_Release, function(x)
  bias_yor$b_yor[which(bias_yor$Year_of_Release == x)]))
t_v_b_rat <- as.numeric(sapply(training$Rating, function(x)
  bias_rat$b_rat[which(bias_rat$Rating == x)]))
t_v_b_dev <- as.numeric(sapply(training$Developer, function(x)
  bias_dev$b_dev[which(bias_dev$Developer == x)]))

###regularization of publisher bias
lambdas <- seq(0, 100, 1)
#vector of RMSE for each lambda
RMSEs_train <- sapply(lambdas, function(lam){
  reg_pub_bias <- bias_pub$sum_bpulog/(lam + bias_pub$count_pu) #regularized publisher bias
  t_v_b_pub_reg <- as.numeric(sapply(training$Publisher, function(x)
    reg_pub_bias[which(bias_pub$Publisher == x)]))
  pred <- 10^(log10_mu + t_v_b_pub_reg +
                t_v_b_pla + t_v_b_ge +
                t_v_b_yor + t_v_b_rat + t_v_b_dev)
  fun_RMSE(training$Global_Sales, pred)
})
#working out the lambda that minimizes training RMSE
plot(lambdas, RMSEs_train)
lambda_pub <- lambdas[which.min(RMSEs_train)]
#creating a new column for regularized publisher bias
bias_pub <- bias_pub %>% mutate(reg_b_pub  = sum_bpulog/(lambda_pub + count_pu))
#re compute the publisher bias for training set row with regularized data
t_v_b_pub <- as.numeric(sapply(training$Publisher, function(x)
  bias_pub$reg_b_pub[which(bias_pub$Publisher == x)]))

###regularization of platform bias
lambdas <- seq(0, 100, 1)
#vector of RMSE for each lambda
RMSEs_train <- sapply(lambdas, function(lam){
  reg_pla_bias <- bias_plat$sum_bplalog/(lam + bias_plat$count_pla) #regularized platform bias
  t_v_b_pla_reg <- as.numeric(sapply(training$Platform, function(x)
    reg_pla_bias[which(bias_plat$Platform == x)]))
  pred <- 10^(log10_mu + t_v_b_pub +
                t_v_b_pla_reg + t_v_b_ge +
                t_v_b_yor + t_v_b_rat + t_v_b_dev)
  fun_RMSE(training$Global_Sales, pred)
})
#working out the lambda that minimizes training RMSE
plot(lambdas, RMSEs_train)
lambda_plat <- lambdas[which.min(RMSEs_train)]
#creating a new column for regularized platform bias
bias_plat <- bias_plat %>% mutate(reg_b_pla  = sum_bplalog/(lambda_plat + count_pla))
#re compute the platform bias for training set row with regularized data
t_v_b_pla <- as.numeric(sapply(training$Platform, function(x)
  bias_plat$reg_b_pla[which(bias_plat$Platform == x)]))

###regularization of genre bias
lambdas <- seq(0, 100, 1)
#vector of RMSE for each lambda
RMSEs_train <- sapply(lambdas, function(lam){
  reg_ge_bias <- bias_genre$sum_bgelog/(lam + bias_genre$count_ge) #regularized genre bias
  t_v_b_ge_reg <- as.numeric(sapply(training$Genre, function(x)
    reg_ge_bias[which(bias_genre$Genre == x)]))
  pred <- 10^(log10_mu + t_v_b_pub +
                t_v_b_pla + t_v_b_ge_reg +
                t_v_b_yor + t_v_b_rat + t_v_b_dev)
  fun_RMSE(training$Global_Sales, pred)
})
#working out the lambda that minimizes training RMSE
plot(lambdas, RMSEs_train)
lambda_ge <- lambdas[which.min(RMSEs_train)]
#creating a new column for regularized genre bias
bias_genre <- bias_genre %>% mutate(reg_b_ge  = sum_bgelog/(lambda_ge + count_ge))
#re compute the genre bias for training set row with regularized data
t_v_b_ge <- as.numeric(sapply(training$Genre, function(x)
  bias_genre$reg_b_ge[which(bias_genre$Genre == x)]))

###regularization of year of release bias
lambdas <- seq(20, 30, 1)
#vector of RMSE for each lambda
RMSEs_train <- sapply(lambdas, function(lam){
  reg_yor_bias <- bias_yor$sum_byorlog/(lam + bias_yor$count_yor) #regularized year of release bias
  t_v_b_yor_reg <- as.numeric(sapply(training$Year_of_Release, function(x)
    reg_yor_bias[which(bias_yor$Year_of_Release == x)]))
  pred <- 10^(log10_mu + t_v_b_pub +
                t_v_b_pla + t_v_b_ge +
                t_v_b_yor_reg + t_v_b_rat + t_v_b_dev)
  fun_RMSE(training$Global_Sales, pred)
})
#working out the lambda that minimizes training RMSE
plot(lambdas, RMSEs_train)
lambda_yor <- lambdas[which.min(RMSEs_train)]
#creating a new column for regularized release year bias
bias_yor <- bias_yor %>% mutate(reg_b_yor  = sum_byorlog/(lambda_yor + count_yor))
#re compute the release year bias for training set row with regularized data
t_v_b_yor <- as.numeric(sapply(training$Year_of_Release, function(x)
  bias_yor$reg_b_yor[which(bias_yor$Year_of_Release  == x)]))

###regularization of rating bias
lambdas <- seq(5, 100, 1)
#vector of RMSE for each lambda
RMSEs_train <- sapply(lambdas, function(lam){
  reg_rat_bias <- bias_rat$sum_bratlog/(lam + bias_rat$count_rat) #regularized rating bias
  t_v_b_rat_reg <- as.numeric(sapply(training$Rating, function(x)
    reg_rat_bias[which(bias_rat$Rating == x)]))
  pred <- 10^(log10_mu + t_v_b_pub +
                t_v_b_pla + t_v_b_ge +
                t_v_b_yor + t_v_b_rat_reg + t_v_b_dev)
  fun_RMSE(training$Global_Sales, pred)
})
#working out the lambda that minimizes training RMSE
plot(lambdas, RMSEs_train)
lambda_rat <- lambdas[which.min(RMSEs_train)]
#creating a new column for regularized rating bias
bias_rat <- bias_rat %>% mutate(reg_b_rat  = sum_bratlog/(lambda_rat + count_rat))
#re compute the rating bias for training set row with regularized data
t_v_b_rat <- as.numeric(sapply(training$Rating, function(x)
  bias_rat$reg_b_rat[which(bias_rat$Rating  == x)]))

###regularization of developer bias
lambdas <- seq(5, 100, 1)
#vector of RMSE for each lambda
RMSEs_train <- sapply(lambdas, function(lam){
  reg_dev_bias <- bias_dev$sum_bdevlog/(lam + bias_dev$count_dev) #regularized developer bias
  t_v_b_dev_reg <- as.numeric(sapply(training$Developer, function(x)
    reg_dev_bias[which(bias_dev$Developer == x)]))
  pred <- 10^(log10_mu + t_v_b_pub +
                t_v_b_pla + t_v_b_ge +
                t_v_b_yor + t_v_b_rat + t_v_b_dev_reg)
  fun_RMSE(training$Global_Sales, pred)
})
#working out the lambda that minimizes training RMSE
plot(lambdas, RMSEs_train)
lambda_dev <- lambdas[which.min(RMSEs_train)]
#creating a new column for regularized rating bias
bias_dev <- bias_dev %>% mutate(reg_b_dev  = sum_bdevlog/(lambda_dev + count_dev))
#re compute the rating bias for training set row with regularized data
t_v_b_dev <- as.numeric(sapply(training$Developer, function(x)
  bias_dev$reg_b_dev[which(bias_dev$Developer  == x)]))

###computes the regularized bias fro each row of the test set
vec_bias_pub <- as.numeric(sapply(test$Publisher, function(x)
  bias_pub$reg_b_pub[which(bias_pub$Publisher == x)]))
vec_bias_pla <- as.numeric(sapply(test$Platform, function(x)
  bias_plat$reg_b_pla[which(bias_plat$Platform == x)]))
vec_bias_ge <- as.numeric(sapply(test$Genre, function(x)
  bias_genre$reg_b_ge[which(bias_genre$Genre == x)]))
vec_bias_yor <- as.numeric(sapply(test$Year_of_Release, function(x)
  bias_yor$reg_b_yor[which(bias_yor$Year_of_Release == x)]))
vec_bias_rat <- as.numeric(sapply(test$Rating, function(x)
  bias_rat$reg_b_rat[which(bias_rat$Rating == x)]))
vec_bias_dev <- as.numeric(sapply(test$Developer, function(x)
  bias_dev$reg_b_dev[which(bias_dev$Developer == x)]))

#using the four bias to make predictions
pred = 10^(log10_mu + vec_bias_pub + vec_bias_pla + vec_bias_ge +
             vec_bias_yor + vec_bias_rat + vec_bias_dev)
RMSE <- fun_RMSE(test$Global_Sales, pred)
#keep track of RMSE
mat_RMSE <- rbind(mat_RMSE, c("LM: Worldwide with regularized environmental bias", RMSE))
mat_RMSE

###free memory
rm(lambdas, lambda_ge, lambda_plat, lambda_pub, lambda_rat,
   lambda_yor, lambda_dev, RMSEs_train, pred)

#######################################################
###      worldwide: taking reviews into account
#######################################################

###exploratory data analysis
training %>%
  ggplot(aes(Critic_Count, Critic_Score)) +
  geom_point() +
  xlab("Number of critic rating") +
  ylab("Critic mean score") +
  ggtitle("Critic score plotted against number of critic")
training %>%
  ggplot(aes(Critic_Score, Global_Sales)) +
  geom_point() +
  xlab("Mean Critic score") +
  ylab("Sales worldwide") +
  ggtitle("Sales plotted against critic score")
training %>%
  ggplot(aes(User_Count, User_Score)) +
  geom_point() +
  xlab("Number of user rating") +
  ylab("User mean score") +
  ggtitle("User score plotted against number of number of user reviews")
training %>%
  ggplot(aes(User_Score, Global_Sales)) +
  geom_point() +
  xlab("Mean User score") +
  ylab("Sales worldwide") +
  ggtitle("Sales plotted against user score")

###creation of a subset with scores and new relevant columns
scores_set <- training %>%
  select(gameId, logGlobal_Sales, Critic_Score, Critic_Count, User_Score, User_Count) %>%
  mutate(sumCritScore = Critic_Count*Critic_Score, sumUserScore = User_Count*User_Score)
#the two columns added will be useful for regularization

###fits a linear model for critic and user score
fitFormula <- logGlobal_Sales -
  (log10_mu + t_v_b_pub + t_v_b_pla + t_v_b_ge +
    t_v_b_yor + t_v_b_rat + t_v_b_dev) ~ Critic_Score + User_Score
fit <- lm(fitFormula, data = scores_set )
summary(fit)
intercept <- fit$coefficients[1]
alpha <- fit$coefficients[2]
beta <- fit$coefficients[3]
rm(fit)

###using the environmental bias and scores to make predictions
pred = 10^(log10_mu + vec_bias_pub + vec_bias_pla + vec_bias_ge +
             vec_bias_yor + vec_bias_rat + vec_bias_dev +
             intercept + alpha*test$Critic_Score + beta*test$User_Score)
RMSE <- fun_RMSE(test$Global_Sales, pred)
#keep track of RMSE
mat_RMSE <- rbind(mat_RMSE, c("LM: Worldwide with regularized environmental bias and score linear model", RMSE))
mat_RMSE



#######################################################
###      worldwide: regularizing reviews
#######################################################

###regularization of critic score
lambdas <- seq(20, 150, 1)
#vector of RMSE for each lambda
RMSEs_train <- sapply(lambdas, function(lam){
  reg_crit_score <- scores_set$sumCritScore/(lam + scores_set$Critic_Count) #regularized critic score
  pred <- 10^(log10_mu + t_v_b_pub + t_v_b_pla + t_v_b_ge +
          t_v_b_yor + t_v_b_rat + t_v_b_dev + intercept +
          alpha*reg_crit_score + beta*scores_set$User_Score)
  fun_RMSE(training$Global_Sales, pred)
})
#working out the lambda that minimizes training RMSE
plot(lambdas, RMSEs_train)
lambda_crit <- lambdas[which.min(RMSEs_train)]
#adding regularized column to test and train set
training <- training %>% mutate(Critic_Score_Reg = scores_set$sumCritScore/(lambda_crit + scores_set$Critic_Count))
test <- test %>% mutate(Critic_Score_Reg = (Critic_Score*Critic_Count)/(lambda_crit + Critic_Count))

###regularization of user score
lambdas <- seq(-3, 2, 0.1)
#vector of RMSE for each lambda
RMSEs_train <- sapply(lambdas, function(lam){
  reg_user_score <- scores_set$sumUserScore/(lam + scores_set$User_Count) #regularized critic score
  pred <- 10^(log10_mu + t_v_b_pub + t_v_b_pla + t_v_b_ge +
                t_v_b_yor + t_v_b_rat + t_v_b_dev + intercept +
                alpha*training$Critic_Score_Reg + beta*reg_user_score)
  fun_RMSE(training$Global_Sales, pred)
})
#working out the lambda that minimizes training RMSE
plot(lambdas, RMSEs_train)
lambda_user <- lambdas[which.min(RMSEs_train)]
#adding regularized column to test and train set
training <- training %>% mutate(User_Score_Reg = scores_set$sumUserScore/(lambda_user + scores_set$User_Count))
test <- test %>% mutate(User_Score_Reg = (User_Score*User_Count)/(lambda_user + User_Count))

###computes the new RMSE on test set for regularized scores
pred = 10^(log10_mu + vec_bias_pub + vec_bias_pla + vec_bias_ge +
             vec_bias_yor + vec_bias_rat + vec_bias_dev + intercept +
             alpha*test$Critic_Score_Reg + beta*test$User_Score_Reg)
RMSE <- fun_RMSE(test$Global_Sales, pred)
#keep track of RMSE
mat_RMSE <- rbind(mat_RMSE, c("LM: Worldwide with environmental bias and score linear model both regularized", RMSE))
mat_RMSE



#######################################################
###      R environment cleaning
#######################################################

###remove added columns to both datasets
test <- test %>% select(-User_Score_Reg, -Critic_Score_Reg)
training <- training %>% select(-logGlobal_Sales, -User_Score_Reg, -Critic_Score_Reg)
###free memory
rm(bias_genre, bias_plat, bias_pub, bias_rat, bias_yor, bias_dev, scores_set)
rm(alpha, beta, fitFormula, intercept, lambdas, lambda_crit, lambda_user, log10_mu)
rm(pred, RMSE, RMSEs_train, t_v_b_ge, t_v_b_pla, t_v_b_pub, t_v_b_rat, t_v_b_yor, t_v_b_dev)
rm(vec_bias_ge, vec_bias_pla, vec_bias_pub, vec_bias_rat, vec_bias_yor, vec_bias_dev)
###data adjustment
#for games that weren't sold in a given region, the log gives -infinity,
#and of course I can't compute the mean or anything, so I decided
#to transform every 0 to 1 so that the log is 0. This is totally arbitrary
for(j in 7:10) #sales column indexes in both sets
{
  for(i in 1:length(training$gameId)) #row indexes for training
  {
    if(training[i,j] == 0)      training[i,j] <- 1
  }
}
rm(i, j)



#######################################################
###      Region-wise predictions
#######################################################

#the fun_Region_Pred function does the entire process of prediction
#for a specific region. Note that we can use it to predict the
#global sales and end up with the (almost) same result as before.

fun_Region_Pred_LM <- function(Region_Sales_train)
{
  #making new columns for the log transform
  training <- training %>% mutate(Region_Sales_train, logRegion_Sales_train=log10(Region_Sales_train))
  
  log10_mu <- mean(training$logRegion_Sales_train) #overall mean of the log10 sales
  
  ###publisher bias
  bias_pub <- training %>%
    group_by(Publisher) %>%
    summarise(b_pu = mean(logRegion_Sales_train - log10_mu),
              count_pu = n(),
              sum_bpulog = sum(logRegion_Sales_train - log10_mu),
              .groups="drop")
  
  ###platform bias
  bias_plat <- training %>%
    left_join(bias_pub, by="Publisher") %>%
    group_by(Platform) %>%
    summarise(b_pla = mean(logRegion_Sales_train -log10_mu - b_pu),
              count_pla = n(),
              sum_bplalog = sum(logRegion_Sales_train - log10_mu - b_pu),
              .groups="drop")
  
  ###genre bias
  bias_genre <- training %>%
    left_join(bias_pub, by="Publisher") %>%
    left_join(bias_plat, by="Platform") %>%
    group_by(Genre) %>%
    summarise(b_ge = mean(logRegion_Sales_train - log10_mu - b_pu - b_pla),
              count_ge = n(),
              sum_bgelog = sum(logRegion_Sales_train - log10_mu - b_pu - b_pla),
              .groups="drop")
  
  ###year of release bias
  bias_yor <- training %>%
    left_join(bias_pub, by="Publisher") %>%
    left_join(bias_plat, by="Platform") %>%
    left_join(bias_genre, by="Genre") %>%
    group_by(Year_of_Release) %>%
    summarise(b_yor = mean(logRegion_Sales_train - log10_mu - b_pu - b_pla - b_ge),
              count_yor = n(),
              sum_byorlog = sum(logRegion_Sales_train - log10_mu - b_pu - b_pla - b_ge),
              .groups="drop")
  
  ###rating bias
  bias_rat <- training %>%
    left_join(bias_pub, by="Publisher") %>%
    left_join(bias_plat, by="Platform") %>%
    left_join(bias_genre, by="Genre") %>%
    left_join(bias_yor, by="Year_of_Release") %>%
    group_by(Rating) %>%
    summarise(b_rat  = mean(logRegion_Sales_train - log10_mu - b_pu - b_pla - b_ge - b_yor),
              count_rat = n(),
              sum_bratlog = sum(logRegion_Sales_train - log10_mu - b_pu - b_pla - b_ge - b_yor),
              .groups="drop")
  
  ###developer bias
  bias_dev <- training %>%
    left_join(bias_pub, by="Publisher") %>%
    left_join(bias_plat, by="Platform") %>%
    left_join(bias_genre, by="Genre") %>%
    left_join(bias_yor, by="Year_of_Release") %>%
    left_join(bias_rat, by="Rating") %>%
    group_by(Developer) %>%
    summarise(b_dev = mean(logRegion_Sales_train - log10_mu - b_pu - b_pla - b_ge - b_yor - b_rat),
              count_dev = n(),
              sum_bdevlog = sum(logRegion_Sales_train - log10_mu - b_pu - b_pla - b_ge - b_yor - b_rat),
              .groups="drop")
  
  ###computes the bias for the rows of the training set before regularizing
  t_v_b_pub <- as.numeric(sapply(training$Publisher, function(x)
    bias_pub$b_pu[which(bias_pub$Publisher == x)]))
  t_v_b_pla <- as.numeric(sapply(training$Platform, function(x)
    bias_plat$b_pla[which(bias_plat$Platform == x)]))
  t_v_b_ge <- as.numeric(sapply(training$Genre, function(x)
    bias_genre$b_ge[which(bias_genre$Genre == x)]))
  t_v_b_yor <- as.numeric(sapply(training$Year_of_Release, function(x)
    bias_yor$b_yor[which(bias_yor$Year_of_Release == x)]))
  t_v_b_rat <- as.numeric(sapply(training$Rating, function(x)
    bias_rat$b_rat[which(bias_rat$Rating == x)]))
  t_v_b_dev <- as.numeric(sapply(training$Developer, function(x)
    bias_dev$b_dev[which(bias_dev$Developer == x)]))
  
  ###regularization of publisher bias
  lambdas <- seq(0, 100, 1)
  #vector of RMSE for each lambda
  RMSEs_train <- sapply(lambdas, function(lam){
    reg_pub_bias <- bias_pub$sum_bpulog/(lam + bias_pub$count_pu) #regularized publisher bias
    t_v_b_pub_reg <- as.numeric(sapply(training$Publisher, function(x)
      reg_pub_bias[which(bias_pub$Publisher == x)]))
    pred <- 10^(log10_mu + t_v_b_pub_reg + t_v_b_pla + t_v_b_ge +
                  t_v_b_yor + t_v_b_rat + t_v_b_dev)
    fun_RMSE(Region_Sales_train, pred)
  })
  #working out the lambda that minimizes training RMSE
  lambda_pub <- lambdas[which.min(RMSEs_train)]
  #creating a new column for regularized publisher bias
  bias_pub <- bias_pub %>% mutate(reg_b_pub  = sum_bpulog/(lambda_pub + count_pu))
  #re compute the publisher bias for training set row with regularized data
  t_v_b_pub <- as.numeric(sapply(training$Publisher, function(x)
    bias_pub$reg_b_pub [which(bias_pub$Publisher == x)]))
  
  ###regularization of platform bias
  lambdas <- seq(0, 100, 1)
  #vector of RMSE for each lambda
  RMSEs_train <- sapply(lambdas, function(lam){
    reg_pla_bias <- bias_plat$sum_bplalog/(lam + bias_plat$count_pla) #regularized platform bias
    t_v_b_pla_reg <- as.numeric(sapply(training$Platform, function(x)
      reg_pla_bias[which(bias_plat$Platform == x)]))
    pred <- 10^(log10_mu + t_v_b_pub + t_v_b_pla_reg + t_v_b_ge +
                  t_v_b_yor + t_v_b_rat + t_v_b_dev)
    fun_RMSE(Region_Sales_train, pred)
  })
  #working out the lambda that minimizes training RMSE
  lambda_plat <- lambdas[which.min(RMSEs_train)]
  #creating a new column for regularized platform bias
  bias_plat <- bias_plat %>% mutate(reg_b_pla  = sum_bplalog/(lambda_plat + count_pla))
  #re compute the platform bias for training set row with regularized data
  t_v_b_pla <- as.numeric(sapply(training$Platform, function(x)
    bias_plat$reg_b_pla [which(bias_plat$Platform == x)]))
  
  ###regularization of genre bias
  lambdas <- seq(0, 100, 1)
  #vector of RMSE for each lambda
  RMSEs_train <- sapply(lambdas, function(lam){
    reg_ge_bias <- bias_genre$sum_bgelog/(lam + bias_genre$count_ge) #regularized genre bias
    t_v_b_ge_reg <- as.numeric(sapply(training$Genre, function(x)
      reg_ge_bias[which(bias_genre$Genre == x)]))
    pred <- 10^(log10_mu + t_v_b_pub + t_v_b_pla + t_v_b_ge_reg +
                  t_v_b_yor + t_v_b_rat + t_v_b_dev)
    fun_RMSE(Region_Sales_train, pred)
  })
  #working out the lambda that minimizes training RMSE
  lambda_ge <- lambdas[which.min(RMSEs_train)]
  #creating a new column for regularized genre bias
  bias_genre <- bias_genre %>% mutate(reg_b_ge  = sum_bgelog/(lambda_ge + count_ge))
  #re compute the genre bias for training set row with regularized data
  t_v_b_ge <- as.numeric(sapply(training$Genre, function(x)
    bias_genre$reg_b_ge [which(bias_genre$Genre == x)]))
  
  ###regularization of year of release bias
  lambdas <- seq(0, 100, 1)
  #vector of RMSE for each lambda
  RMSEs_train <- sapply(lambdas, function(lam){
    reg_yor_bias <- bias_yor$sum_byorlog/(lam + bias_yor$count_yor) #regularized year of release bias
    t_v_b_yor_reg <- as.numeric(sapply(training$Year_of_Release, function(x)
      reg_yor_bias[which(bias_yor$Year_of_Release == x)]))
    pred <- 10^(log10_mu + t_v_b_pub + t_v_b_pla + t_v_b_ge +
                  t_v_b_yor_reg + t_v_b_rat + t_v_b_dev)
    fun_RMSE(Region_Sales_train, pred)
  })
  #working out the lambda that minimizes training RMSE
  lambda_yor <- lambdas[which.min(RMSEs_train)]
  #creating a new column for regularized release year bias
  bias_yor <- bias_yor %>% mutate(reg_b_yor  = sum_byorlog/(lambda_yor + count_yor))
  #re compute the release year bias for training set row with regularized data
  t_v_b_yor <- as.numeric(sapply(training$Year_of_Release, function(x)
    bias_yor$reg_b_yor [which(bias_yor$Year_of_Release  == x)]))
  
  ###regularization of rating bias
  lambdas <- seq(0, 100, 1)
  #vector of RMSE for each lambda
  RMSEs_train <- sapply(lambdas, function(lam){
    reg_rat_bias <- bias_rat$sum_bratlog/(lam + bias_rat$count_rat) #regularized rating bias
    t_v_b_rat_reg <- as.numeric(sapply(training$Rating, function(x)
      reg_rat_bias[which(bias_rat$Rating == x)]))
    pred <- 10^(log10_mu + t_v_b_pub + t_v_b_pla + t_v_b_ge +
                  t_v_b_yor + t_v_b_rat_reg + t_v_b_dev)
    fun_RMSE(Region_Sales_train, pred)
  })
  #working out the lambda that minimizes training RMSE
  lambda_rat <- lambdas[which.min(RMSEs_train)]
  #creating a new column for regularized rating bias
  bias_rat <- bias_rat %>% mutate(reg_b_rat  = sum_bratlog/(lambda_rat + count_rat))
  #re compute the rating bias for training set row with regularized data
  t_v_b_rat <- as.numeric(sapply(training$Rating, function(x)
    bias_rat$reg_b_rat[which(bias_rat$Rating  == x)]))
  
  ###regularization of developer bias
  lambdas <- seq(0, 100, 1)
  #vector of RMSE for each lambda
  RMSEs_train <- sapply(lambdas, function(lam){
    reg_dev_bias <- bias_dev$sum_bdevlog/(lam + bias_dev$count_dev) #regularized developer bias
    t_v_b_dev_reg <- as.numeric(sapply(training$Developer, function(x)
      reg_dev_bias[which(bias_dev$Developer == x)]))
    pred <- 10^(log10_mu + t_v_b_pub + t_v_b_pla + t_v_b_ge +
                  t_v_b_yor + t_v_b_rat + t_v_b_dev_reg)
    fun_RMSE(Region_Sales_train, pred)
  })
  #working out the lambda that minimizes training RMSE
  lambda_dev <- lambdas[which.min(RMSEs_train)]
  #creating a new column for regularized rating bias
  bias_dev <- bias_dev %>% mutate(reg_b_dev  = sum_bdevlog/(lambda_dev + count_dev))
  #re compute the rating bias for training set row with regularized data
  t_v_b_dev <- as.numeric(sapply(training$Developer, function(x)
    bias_dev$reg_b_dev[which(bias_dev$Developer  == x)]))
  
  ###computes the regularized bias fro each row of the test set
  vec_bias_pub <- as.numeric(sapply(test$Publisher, function(x)
    bias_pub$reg_b_pub[which(bias_pub$Publisher == x)]))
  vec_bias_pla <- as.numeric(sapply(test$Platform, function(x)
    bias_plat$reg_b_pla[which(bias_plat$Platform == x)]))
  vec_bias_ge <- as.numeric(sapply(test$Genre, function(x)
    bias_genre$reg_b_ge[which(bias_genre$Genre == x)]))
  vec_bias_yor <- as.numeric(sapply(test$Year_of_Release, function(x)
    bias_yor$reg_b_yor[which(bias_yor$Year_of_Release == x)]))
  vec_bias_rat <- as.numeric(sapply(test$Rating, function(x)
    bias_rat$reg_b_rat[which(bias_rat$Rating == x)]))
  vec_bias_dev <- as.numeric(sapply(test$Developer, function(x)
    bias_dev$reg_b_dev[which(bias_dev$Developer == x)]))
  
  ###computes the bias for the rows of the training set before fitting a linear model
  t_v_b_pub <- as.numeric(sapply(training$Publisher, function(x)
    bias_pub$reg_b_pub [which(bias_pub$Publisher == x)]))
  t_v_b_pla <- as.numeric(sapply(training$Platform, function(x)
    bias_plat$reg_b_pla [which(bias_plat$Platform == x)]))
  t_v_b_ge <- as.numeric(sapply(training$Genre, function(x)
    bias_genre$reg_b_ge [which(bias_genre$Genre == x)]))
  t_v_b_yor <- as.numeric(sapply(training$Year_of_Release, function(x)
    bias_yor$reg_b_yor [which(bias_yor$Year_of_Release == x)]))
  t_v_b_rat <- as.numeric(sapply(training$Rating, function(x)
    bias_rat$reg_b_rat [which(bias_rat$Rating == x)]))
  t_v_b_dev <- as.numeric(sapply(training$Developer, function(x)
    bias_dev$reg_b_dev[which(bias_dev$Developer == x)]))
  
  ###fits a linear model for critic and user score
  fitFormula <- logRegion_Sales_train -
    (log10_mu + t_v_b_pub + t_v_b_pla + t_v_b_ge +
       t_v_b_yor + t_v_b_rat + t_v_b_dev) ~ Critic_Score + User_Score
  fit <- lm(fitFormula, data = training )
  intercept <- fit$coefficients[1]
  alpha <- fit$coefficients[2]
  beta <- fit$coefficients[3]
  rm(fit)
  
  ###creation of a subset with scores and new relevant columns
  scores_set <- training %>%
    select(gameId, logRegion_Sales_train, Critic_Score, Critic_Count, User_Score, User_Count) %>%
    mutate(sumCritScore = Critic_Count*Critic_Score, sumUserScore = User_Count*User_Score)
  #the two columns added will be useful for regularization
  
  ###regularization of critic score
  lambdas <- seq(20, 150, 1)
  #vector of RMSE for each lambda
  RMSEs_train <- sapply(lambdas, function(lam){
    reg_crit_score <- scores_set$sumCritScore/(lam + scores_set$Critic_Count) #regularized critic score
    pred <- 10^(log10_mu + t_v_b_pub + t_v_b_pla + t_v_b_ge +
                  t_v_b_yor + t_v_b_rat + t_v_b_dev + intercept +
                  alpha*reg_crit_score + beta*scores_set$User_Score)
    fun_RMSE(Region_Sales_train, pred)
  })
  #working out the lambda that minimizes training RMSE
  lambda_crit <- lambdas[which.min(RMSEs_train)]
  #adding regularized column to test and train set
  training <- training %>% mutate(Critic_Score_Reg = scores_set$sumCritScore/(lambda_crit + scores_set$Critic_Count))
  test <- test %>% mutate(Critic_Score_Reg = (Critic_Score*Critic_Count)/(lambda_crit + Critic_Count))
  
  ###regularization of user score
  lambdas <- seq(-3, 5, 0.1)
  #vector of RMSE for each lambda
  RMSEs_train <- sapply(lambdas, function(lam){
    reg_user_score <- scores_set$sumUserScore/(lam + scores_set$User_Count) #regularized critic score
    pred <- 10^(log10_mu + t_v_b_pub + t_v_b_pla + t_v_b_ge +
                  t_v_b_yor + t_v_b_rat + t_v_b_dev + intercept +
                  alpha*training$Critic_Score_Reg + beta*reg_user_score)
    fun_RMSE(Region_Sales_train, pred)
  })
  #working out the lambda that minimizes training RMSE
  lambda_user <- lambdas[which.min(RMSEs_train)]
  #adding regularized column to test and train set
  training <- training %>% mutate(User_Score_Reg = scores_set$sumUserScore/(lambda_user + scores_set$User_Count))
  test <- test %>% mutate(User_Score_Reg = (User_Score*User_Count)/(lambda_user + User_Count))
  
  ###computes the predictions with test set
  pred_Region = 10^(log10_mu + vec_bias_pub + vec_bias_pla + vec_bias_ge +
                    vec_bias_yor + vec_bias_rat + vec_bias_dev + intercept +
                    alpha*test$Critic_Score_Reg + beta*test$User_Score_Reg)
  
  #remove the added columns
  test <- test %>% select(-User_Score_Reg, -Critic_Score_Reg)
  training <- training %>% select(-Region_Sales_train, -logRegion_Sales_train, -User_Score_Reg, -Critic_Score_Reg)
  
  return(pred_Region)
}

#use the fun_Region_Pred_LM function to predict the sales in each region
pred_NA <- fun_Region_Pred_LM(training$NA_Sales)
pred_EU <- fun_Region_Pred_LM(training$EU_Sales)
pred_JP <- fun_Region_Pred_LM(training$JP_Sales)
pred_Other <- fun_Region_Pred_LM(training$Other_Sales)



#######################################################
###      Region-wise: Final results
#######################################################

###computing worldwide final predictions
pred <- pred_NA + pred_EU + pred_JP + pred_Other
rm(pred_EU, pred_JP, pred_NA, pred_Other)
RMSE <- fun_RMSE(test$Global_Sales, pred)
###keeping track of the RMSE
mat_RMSE <- rbind(mat_RMSE, c("LM: Region-wise with environmental bias and score linear model both regularized", RMSE))
mat_RMSE


#######################################################
###                   KNN with reviews
#######################################################

###first estimation using only User and Critic score
train_knn <- train(Global_Sales ~ User_Score + Critic_Score,
                   data = training,
                   method = "knn",
                   tuneGrid = data.frame(k = seq(3,15,2)))
train_knn$bestTune
pred <- predict(train_knn, test)
RMSE <- fun_RMSE(test$Global_Sales, pred)
mat_RMSE <- rbind(mat_RMSE, c("KNN: Worldwide with score review", RMSE))
mat_RMSE


#######################################################
###                   KNN with sum of sales
#######################################################

###creating a distance using sum of sales for each environmental variable
#compute the sum (and other data) for each publisher/platform/etc
pub <- training %>% group_by(Publisher) %>%
  summarise(sum = sum(Global_Sales),
            mean = mean(Global_Sales),
            count = n(),
            .groups="drop")
pla <- training %>% group_by(Platform) %>%
  summarise(sum = sum(Global_Sales),
            mean = mean(Global_Sales),
            count = n(),
            .groups="drop")
gen <- training %>% group_by(Genre) %>%
  summarise(sum = sum(Global_Sales),
            mean = mean(Global_Sales),
            count = n(),
            .groups="drop")
yor <- training %>% group_by(Year_of_Release) %>%
  summarise(sum = sum(Global_Sales),
            mean = mean(Global_Sales),
            count = n(),
            .groups="drop")
rat <- training %>% group_by(Rating) %>%
  summarise(sum = sum(Global_Sales),
            mean = mean(Global_Sales),
            count = n(),
            .groups="drop")
dev <- training %>% group_by(Developer) %>%
  summarise(sum = sum(Global_Sales),
            mean = mean(Global_Sales),
            count = n(),
            .groups="drop")

###creating the matching values to columns in training and test set
#each column added matches the value of the sum of sales for
#each possible value of publisher/platform/developer/etc
training$sum_Pub <- sapply(training$Publisher, function(x)
  pub$sum[which(pub$Publisher == x)])
training$sum_Pla <- sapply(training$Platform, function(x)
  pla$sum[which(pla$Platform == x)])
training$sum_Gen <- sapply(training$Genre, function(x)
  gen$sum[which(gen$Genre == x)])
training$sum_Yor <- sapply(training$Year_of_Release, function(x)
  yor$sum[which(yor$Year_of_Release == x)])
training$sum_Rat <- sapply(training$Publisher, function(x)
  pub$sum[which(pub$Publisher == x)])
training$sum_Dev <- sapply(training$Developer, function(x)
  dev$sum[which(dev$Developer == x)])
test$sum_Pub <- sapply(test$Publisher, function(x)
  pub$sum[which(pub$Publisher == x)])
test$sum_Pla <- sapply(test$Platform, function(x)
  pla$sum[which(pla$Platform == x)])
test$sum_Gen <- sapply(test$Genre, function(x)
  gen$sum[which(gen$Genre == x)])
test$sum_Yor <- sapply(test$Year_of_Release, function(x)
  yor$sum[which(yor$Year_of_Release == x)])
test$sum_Rat <- sapply(test$Publisher, function(x)
  pub$sum[which(pub$Publisher == x)])
test$sum_Dev <- sapply(test$Developer, function(x)
  dev$sum[which(dev$Developer == x)])

###using cross validation to manually select the best K
set.seed(1, sample.kind = "Rounding")
Ks <- seq(3,15,2)
RMSEs_train <- sapply(Ks, function(x){
  #creates cross-validation sets
  indexes <- createDataPartition(training$gameId, times = 1, p = 0.2, list=FALSE) 
  validation <- training[indexes,]
  inner_training <- training[-indexes,]
  train_knn <- train(Global_Sales ~ sum_Pub+sum_Pla+sum_Gen+sum_Yor+sum_Rat+sum_Dev,
                     data = inner_training,
                     method = "knn",
                     tuneGrid = data.frame(k = x))
  pred <- predict(train_knn, validation)
  fun_RMSE(validation$Global_Sales, pred)
})
optiK <- Ks[which.min(RMSEs_train)]
rm(RMSEs_train, Ks)


###knn using the sum for each environmental variable
train_knn <- train(Global_Sales ~ sum_Pub+sum_Pla+sum_Gen+sum_Yor+sum_Rat+sum_Dev,
                   data = training,
                   method = "knn",
                   tuneGrid = data.frame(k = optiK))
pred <- predict(train_knn, test)
RMSE <- fun_RMSE(test$Global_Sales, pred)
mat_RMSE <- rbind(mat_RMSE, c("KNN: Worldwide with sum of sales for environmental variables, K from cross validation", RMSE))
mat_RMSE

#######################################################
###                   KNN with mean of sales
#######################################################

###removing columns for sum of sales
training <- training %>% select(-sum_Pub, -sum_Pla, -sum_Gen, -sum_Yor, -sum_Rat, -sum_Dev)
test <- test %>% select(-sum_Pub, -sum_Pla, -sum_Gen, -sum_Yor, -sum_Rat, -sum_Dev)

###creating the matching values to columns in training and test set
#each column added matches the value of the mean of sales for
#each possible value of publisher/platform/developer/etc
training$mean_Pub <- sapply(training$Publisher, function(x)
  pub$mean[which(pub$Publisher == x)])
training$mean_Pla <- sapply(training$Platform, function(x)
  pla$mean[which(pla$Platform == x)])
training$mean_Gen <- sapply(training$Genre, function(x)
  gen$mean[which(gen$Genre == x)])
training$mean_Yor <- sapply(training$Year_of_Release, function(x)
  yor$mean[which(yor$Year_of_Release == x)])
training$mean_Rat <- sapply(training$Publisher, function(x)
  pub$mean[which(pub$Publisher == x)])
training$mean_Dev <- sapply(training$Developer, function(x)
  dev$mean[which(dev$Developer == x)])
test$mean_Pub <- sapply(test$Publisher, function(x)
  pub$mean[which(pub$Publisher == x)])
test$mean_Pla <- sapply(test$Platform, function(x)
  pla$mean[which(pla$Platform == x)])
test$mean_Gen <- sapply(test$Genre, function(x)
  gen$mean[which(gen$Genre == x)])
test$mean_Yor <- sapply(test$Year_of_Release, function(x)
  yor$mean[which(yor$Year_of_Release == x)])
test$mean_Rat <- sapply(test$Publisher, function(x)
  pub$mean[which(pub$Publisher == x)])
test$mean_Dev <- sapply(test$Developer, function(x)
  dev$mean[which(dev$Developer == x)])

###using cross validation to manually select the best K
set.seed(42, sample.kind = "Rounding")
Ks <- seq(3,15,2)
RMSEs_train <- sapply(Ks, function(x){
  #creates cross-validation sets
  indexes <- createDataPartition(training$gameId, times = 1, p = 0.2, list=FALSE) 
  validation <- training[indexes,]
  inner_training <- training[-indexes,]
  train_knn <- train(Global_Sales ~ mean_Pub+mean_Pla+mean_Gen+mean_Yor+mean_Rat+mean_Dev,
                     data = inner_training,
                     method = "knn",
                     tuneGrid = data.frame(k = x))
  pred <- predict(train_knn, validation)
  fun_RMSE(validation$Global_Sales, pred)
})
optiK <- Ks[which.min(RMSEs_train)]
rm(RMSEs_train, Ks)

###using the mean for each environmental variable
train_knn <- train(Global_Sales ~ mean_Pub+mean_Pla+mean_Gen+mean_Yor+mean_Rat+mean_Dev,
                   data = training,
                   method = "knn",
                   tuneGrid = data.frame(k = optiK))
pred<- predict(train_knn, test)
RMSE <- fun_RMSE(test$Global_Sales, pred)
mat_RMSE <- rbind(mat_RMSE, c("KNN: Worldwide with mean of sales for environmental variables, K from cross validation", RMSE))
mat_RMSE

###free memory
#delete variables
rm(dev, gen, pla, pub, rat, yor, train_knn, optiK)
#removing columns for mean of sales
training <- training %>% select(-mean_Pub, -mean_Pla, -mean_Gen, -mean_Yor, -mean_Rat, -mean_Dev)
test <- test %>% select(-mean_Pub, -mean_Pla, -mean_Gen, -mean_Yor, -mean_Rat, -mean_Dev)

#######################################################
###                   Region-wise KNN
#######################################################

fun_Region_Pred_KNN <- function(Region_Sales_train)
{
  training <- training %>% mutate(Region_Sales_train)
  ###computing the sum of sales in a given region for each environmental variable
  pub <- training %>% group_by(Publisher) %>%
    summarise(sum = sum(Region_Sales_train),
              mean = mean(Region_Sales_train),
              count = n(),
              .groups="drop")
  pla <- training %>% group_by(Platform) %>%
    summarise(sum = sum(Region_Sales_train),
              mean = mean(Region_Sales_train),
              count = n(),
              .groups="drop")
  gen <- training %>% group_by(Genre) %>%
    summarise(sum = sum(Region_Sales_train),
              mean = mean(Region_Sales_train),
              count = n(),
              .groups="drop")
  yor <- training %>% group_by(Year_of_Release) %>%
    summarise(sum = sum(Region_Sales_train),
              mean = mean(Region_Sales_train),
              count = n(),
              .groups="drop")
  rat <- training %>% group_by(Rating) %>%
    summarise(sum = sum(Region_Sales_train),
              mean = mean(Region_Sales_train),
              count = n(),
              .groups="drop")
  dev <- training %>% group_by(Developer) %>%
    summarise(sum = sum(Region_Sales_train),
              mean = mean(Region_Sales_train),
              count = n(),
              .groups="drop")
  
  ###adding matching columns to both training and test set
  training$mean_Pub <- as.numeric(sapply(training$Publisher, function(x)
    pub$mean[which(pub$Publisher == x)]))
  training$mean_Pla <- as.numeric(sapply(training$Platform, function(x)
    pla$mean[which(pla$Platform == x)]))
  training$mean_Gen <- as.numeric(sapply(training$Genre, function(x)
    gen$mean[which(gen$Genre == x)]))
  training$mean_Yor <- as.numeric(sapply(training$Year_of_Release, function(x)
    yor$mean[which(yor$Year_of_Release == x)]))
  training$mean_Rat <- as.numeric(sapply(training$Publisher, function(x)
    pub$mean[which(pub$Publisher == x)]))
  training$mean_Dev <- as.numeric(sapply(training$Developer, function(x)
    dev$mean[which(dev$Developer == x)]))
  test$mean_Pub <- as.numeric(sapply(test$Publisher, function(x)
    pub$mean[which(pub$Publisher == x)]))
  test$mean_Pla <- as.numeric(sapply(test$Platform, function(x)
    pla$mean[which(pla$Platform == x)]))
  test$mean_Gen <- as.numeric(sapply(test$Genre, function(x)
    gen$mean[which(gen$Genre == x)]))
  test$mean_Yor <- as.numeric(sapply(test$Year_of_Release, function(x)
    yor$mean[which(yor$Year_of_Release == x)]))
  test$mean_Rat <- as.numeric(sapply(test$Publisher, function(x)
    pub$mean[which(pub$Publisher == x)]))
  test$mean_Dev <- as.numeric(sapply(test$Developer, function(x)
    dev$mean[which(dev$Developer == x)]))
  
  ###using cross validation to manually select the best K
  set.seed(Region_Sales_train[1], sample.kind = "Rounding")
  Ks <- seq(3,15,2)
  RMSEs_train <- sapply(Ks, function(x){
    #creates cross-validation sets
    indexes <- createDataPartition(training$gameId, times = 1, p = 0.2, list=FALSE) 
    validation <- training[indexes,]
    inner_training <- training[-indexes,]
    train_knn <- train(Global_Sales ~ mean_Pub+mean_Pla+mean_Gen+mean_Yor+mean_Rat+mean_Dev,
                       data = inner_training,
                       method = "knn",
                       tuneGrid = data.frame(k = x))
    pred <- predict(train_knn, validation)
    fun_RMSE(validation$Global_Sales, pred)
  })
  optiK <- Ks[which.min(RMSEs_train)]
  rm(RMSEs_train, Ks)
  
  ###knn using the mean for each environmental variable
  train_knn <- train(Global_Sales ~ mean_Pub+mean_Pla+mean_Gen+mean_Yor+mean_Rat+mean_Dev,
                     data = training,
                     method = "knn",
                     tuneGrid = data.frame(k = optiK))
  pred <- predict(train_knn, test)
  
  ###removing columns added by this function
  training <- training %>% select(-Region_Sales_train, -mean_Pub, -mean_Pla, -mean_Gen, -mean_Yor, -mean_Rat, -mean_Dev)
  test <- test %>% select(-mean_Pub, -mean_Pla, -mean_Gen, -mean_Yor, -mean_Rat, -mean_Dev)
  
  return(pred)
}

###use the fun_Region_Pred_KNN function to predict the sales in each region
pred_NA <- fun_Region_Pred_KNN(training$NA_Sales)
pred_EU <- fun_Region_Pred_KNN(training$EU_Sales)
pred_JP <- fun_Region_Pred_KNN(training$JP_Sales)
pred_Other <- fun_Region_Pred_KNN(training$Other_Sales)

###computing worldwide final predictions
pred <- pred_NA + pred_EU + pred_JP + pred_Other
rm(pred_EU, pred_JP, pred_NA, pred_Other)
RMSE <- fun_RMSE(test$Global_Sales, pred)
###keeping track of the RMSE
mat_RMSE <- rbind(mat_RMSE, c("KNN: Region-wise with mean of sales for environmental variables, K from cross validation", RMSE))
rm(pred, RMSE)
mat_RMSE




#######################################################
###   Data exploratory analysis: LM versus KNN
#######################################################

###creates sets for cross validation
set.seed(1234, sample.kind = "Rounding")
#20% of validation, 80% of "inner training"
indexes <- createDataPartition(training$gameId, times = 1, p = 0.2, list=FALSE) 
sub_training <- training[-indexes,]
temp <- training[indexes,]
validation <- temp %>%
  semi_join(sub_training, by = "Platform") %>%
  semi_join(sub_training, by = "Genre") %>%
  semi_join(sub_training, by = "Year_of_Release") %>%
  semi_join(sub_training, by = "Publisher") %>%
  semi_join(sub_training, by = "Rating") %>%
  semi_join(sub_training, by = "Developer")
removed <- anti_join(temp, validation)
sub_training <- rbind(sub_training, removed)
rm(temp, removed)

###making predictions using both KNN and LM
#cant use functions above, because sets are not training and test
#predictions linear model
  sub_training <- sub_training %>% mutate(Global_Sales, logGlobal_Sales=log10(Global_Sales))
  log10_mu <- mean(sub_training$logGlobal_Sales) #overall mean of the log10 sales
  ###publisher bias
  bias_pub <- sub_training %>%
    group_by(Publisher) %>%
    summarise(b_pu = mean(logGlobal_Sales - log10_mu),
              count_pu = n(),
              sum_bpulog = sum(logGlobal_Sales - log10_mu),
              .groups="drop")
  ###platform bias
  bias_plat <- sub_training %>%
    left_join(bias_pub, by="Publisher") %>%
    group_by(Platform) %>%
    summarise(b_pla = mean(logGlobal_Sales -log10_mu - b_pu),
              count_pla = n(),
              sum_bplalog = sum(logGlobal_Sales - log10_mu - b_pu),
              .groups="drop")
  ###genre bias
  bias_genre <- sub_training %>%
    left_join(bias_pub, by="Publisher") %>%
    left_join(bias_plat, by="Platform") %>%
    group_by(Genre) %>%
    summarise(b_ge = mean(logGlobal_Sales - log10_mu - b_pu - b_pla),
              count_ge = n(),
              sum_bgelog = sum(logGlobal_Sales - log10_mu - b_pu - b_pla),
              .groups="drop")
  ###year of release bias
  bias_yor <- sub_training %>%
    left_join(bias_pub, by="Publisher") %>%
    left_join(bias_plat, by="Platform") %>%
    left_join(bias_genre, by="Genre") %>%
    group_by(Year_of_Release) %>%
    summarise(b_yor = mean(logGlobal_Sales - log10_mu - b_pu - b_pla - b_ge),
              count_yor = n(),
              sum_byorlog = sum(logGlobal_Sales - log10_mu - b_pu - b_pla - b_ge),
              .groups="drop")
  ###rating bias
  bias_rat <- sub_training %>%
    left_join(bias_pub, by="Publisher") %>%
    left_join(bias_plat, by="Platform") %>%
    left_join(bias_genre, by="Genre") %>%
    left_join(bias_yor, by="Year_of_Release") %>%
    group_by(Rating) %>%
    summarise(b_rat  = mean(logGlobal_Sales - log10_mu - b_pu - b_pla - b_ge - b_yor),
              count_rat = n(),
              sum_bratlog = sum(logGlobal_Sales - log10_mu - b_pu - b_pla - b_ge - b_yor),
              .groups="drop")
  ###developer bias
  bias_dev <- sub_training %>%
    left_join(bias_pub, by="Publisher") %>%
    left_join(bias_plat, by="Platform") %>%
    left_join(bias_genre, by="Genre") %>%
    left_join(bias_yor, by="Year_of_Release") %>%
    left_join(bias_rat, by="Rating") %>%
    group_by(Developer) %>%
    summarise(b_dev = mean(logGlobal_Sales - log10_mu - b_pu - b_pla - b_ge - b_yor - b_rat),
              count_dev = n(),
              sum_bdevlog = sum(logGlobal_Sales - log10_mu - b_pu - b_pla - b_ge - b_yor - b_rat),
              .groups="drop")
  ###computes the bias for the rows of the training set before regularizing
  t_v_b_pub <- as.numeric(sapply(sub_training$Publisher, function(x)
    bias_pub$b_pu[which(bias_pub$Publisher == x)]))
  t_v_b_pla <- as.numeric(sapply(sub_training$Platform, function(x)
    bias_plat$b_pla[which(bias_plat$Platform == x)]))
  t_v_b_ge <- as.numeric(sapply(sub_training$Genre, function(x)
    bias_genre$b_ge[which(bias_genre$Genre == x)]))
  t_v_b_yor <- as.numeric(sapply(sub_training$Year_of_Release, function(x)
    bias_yor$b_yor[which(bias_yor$Year_of_Release == x)]))
  t_v_b_rat <- as.numeric(sapply(sub_training$Rating, function(x)
    bias_rat$b_rat[which(bias_rat$Rating == x)]))
  t_v_b_dev <- as.numeric(sapply(sub_training$Developer, function(x)
    bias_dev$b_dev[which(bias_dev$Developer == x)]))
  ###regularization of publisher bias
  lambdas <- seq(-1, 100, 1)
  #vector of RMSE for each lambda
  RMSEs_train <- sapply(lambdas, function(lam){
    reg_pub_bias <- bias_pub$sum_bpulog/(lam + bias_pub$count_pu) #regularized publisher bias
    t_v_b_pub_reg <- as.numeric(sapply(sub_training$Publisher, function(x)
      reg_pub_bias[which(bias_pub$Publisher == x)]))
    pred <- 10^(log10_mu + t_v_b_pub_reg + t_v_b_pla + t_v_b_ge +
                  t_v_b_yor + t_v_b_rat + t_v_b_dev)
    fun_RMSE(sub_training$Global_Sales, pred)
  })
  #working out the lambda that minimizes training RMSE
  lambda_pub <- lambdas[which.min(RMSEs_train)]
  #creating a new column for regularized publisher bias
  bias_pub <- bias_pub %>% mutate(reg_b_pub  = sum_bpulog/(lambda_pub + count_pu))
  #re compute the publisher bias for training set row with regularized data
  t_v_b_pub <- as.numeric(sapply(sub_training$Publisher, function(x)
    bias_pub$reg_b_pub [which(bias_pub$Publisher == x)]))
  ###regularization of platform bias
  lambdas <- seq(-1, 100, 1)
  #vector of RMSE for each lambda
  RMSEs_train <- sapply(lambdas, function(lam){
    reg_pla_bias <- bias_plat$sum_bplalog/(lam + bias_plat$count_pla) #regularized platform bias
    t_v_b_pla_reg <- as.numeric(sapply(sub_training$Platform, function(x)
      reg_pla_bias[which(bias_plat$Platform == x)]))
    pred <- 10^(log10_mu + t_v_b_pub + t_v_b_pla_reg + t_v_b_ge +
                  t_v_b_yor + t_v_b_rat + t_v_b_dev)
    fun_RMSE(sub_training$Global_Sales, pred)
  })
  #working out the lambda that minimizes training RMSE
  lambda_plat <- lambdas[which.min(RMSEs_train)]
  #creating a new column for regularized platform bias
  bias_plat <- bias_plat %>% mutate(reg_b_pla  = sum_bplalog/(lambda_plat + count_pla))
  #re compute the platform bias for training set row with regularized data
  t_v_b_pla <- as.numeric(sapply(sub_training$Platform, function(x)
    bias_plat$reg_b_pla [which(bias_plat$Platform == x)]))
  ###regularization of genre bias
  lambdas <- seq(-1, 100, 1)
  #vector of RMSE for each lambda
  RMSEs_train <- sapply(lambdas, function(lam){
    reg_ge_bias <- bias_genre$sum_bgelog/(lam + bias_genre$count_ge) #regularized genre bias
    t_v_b_ge_reg <- as.numeric(sapply(sub_training$Genre, function(x)
      reg_ge_bias[which(bias_genre$Genre == x)]))
    pred <- 10^(log10_mu + t_v_b_pub + t_v_b_pla + t_v_b_ge_reg +
                  t_v_b_yor + t_v_b_rat + t_v_b_dev)
    fun_RMSE(sub_training$Global_Sales, pred)
  })
  #working out the lambda that minimizes training RMSE
  lambda_ge <- lambdas[which.min(RMSEs_train)]
  #creating a new column for regularized genre bias
  bias_genre <- bias_genre %>% mutate(reg_b_ge  = sum_bgelog/(lambda_ge + count_ge))
  #re compute the genre bias for training set row with regularized data
  t_v_b_ge <- as.numeric(sapply(sub_training$Genre, function(x)
    bias_genre$reg_b_ge [which(bias_genre$Genre == x)]))
  ###regularization of year of release bias
  lambdas <- seq(-1, 100, 1)
  #vector of RMSE for each lambda
  RMSEs_train <- sapply(lambdas, function(lam){
    reg_yor_bias <- bias_yor$sum_byorlog/(lam + bias_yor$count_yor) #regularized year of release bias
    t_v_b_yor_reg <- as.numeric(sapply(sub_training$Year_of_Release, function(x)
      reg_yor_bias[which(bias_yor$Year_of_Release == x)]))
    pred <- 10^(log10_mu + t_v_b_pub + t_v_b_pla + t_v_b_ge +
                  t_v_b_yor_reg + t_v_b_rat + t_v_b_dev)
    fun_RMSE(sub_training$Global_Sales, pred)
  })
  #working out the lambda that minimizes training RMSE
  lambda_yor <- lambdas[which.min(RMSEs_train)]
  #creating a new column for regularized release year bias
  bias_yor <- bias_yor %>% mutate(reg_b_yor  = sum_byorlog/(lambda_yor + count_yor))
  #re compute the release year bias for training set row with regularized data
  t_v_b_yor <- as.numeric(sapply(sub_training$Year_of_Release, function(x)
    bias_yor$reg_b_yor [which(bias_yor$Year_of_Release  == x)]))
  ###regularization of rating bias
  lambdas <- seq(-1, 100, 1)
  #vector of RMSE for each lambda
  RMSEs_train <- sapply(lambdas, function(lam){
    reg_rat_bias <- bias_rat$sum_bratlog/(lam + bias_rat$count_rat) #regularized rating bias
    t_v_b_rat_reg <- as.numeric(sapply(sub_training$Rating, function(x)
      reg_rat_bias[which(bias_rat$Rating == x)]))
    pred <- 10^(log10_mu + t_v_b_pub + t_v_b_pla + t_v_b_ge +
                  t_v_b_yor + t_v_b_rat_reg + t_v_b_dev)
    fun_RMSE(sub_training$Global_Sales, pred)
  })
  #working out the lambda that minimizes training RMSE
  lambda_rat <- lambdas[which.min(RMSEs_train)]
  #creating a new column for regularized rating bias
  bias_rat <- bias_rat %>% mutate(reg_b_rat  = sum_bratlog/(lambda_rat + count_rat))
  #re compute the rating bias for training set row with regularized data
  t_v_b_rat <- as.numeric(sapply(sub_training$Rating, function(x)
    bias_rat$reg_b_rat[which(bias_rat$Rating  == x)]))
  ###regularization of developer bias
  lambdas <- seq(0, 100, 1)
  #vector of RMSE for each lambda
  RMSEs_train <- sapply(lambdas, function(lam){
    reg_dev_bias <- bias_dev$sum_bdevlog/(lam + bias_dev$count_dev) #regularized developer bias
    t_v_b_dev_reg <- as.numeric(sapply(sub_training$Developer, function(x)
      reg_dev_bias[which(bias_dev$Developer == x)]))
    pred <- 10^(log10_mu + t_v_b_pub + t_v_b_pla + t_v_b_ge +
                  t_v_b_yor + t_v_b_rat + t_v_b_dev_reg)
    fun_RMSE(sub_training$Global_Sales, pred)
  })
  #working out the lambda that minimizes training RMSE
  lambda_dev <- lambdas[which.min(RMSEs_train)]
  #creating a new column for regularized rating bias
  bias_dev <- bias_dev %>% mutate(reg_b_dev  = sum_bdevlog/(lambda_dev + count_dev))
  #re compute the rating bias for training set row with regularized data
  t_v_b_dev <- as.numeric(sapply(sub_training$Developer, function(x)
    bias_dev$reg_b_dev[which(bias_dev$Developer  == x)]))
  ###computes the regularized bias fro each row of the test set
  vec_bias_pub <- as.numeric(sapply(validation$Publisher, function(x)
    bias_pub$reg_b_pub[which(bias_pub$Publisher == x)]))
  vec_bias_pla <- as.numeric(sapply(validation$Platform, function(x)
    bias_plat$reg_b_pla[which(bias_plat$Platform == x)]))
  vec_bias_ge <- as.numeric(sapply(validation$Genre, function(x)
    bias_genre$reg_b_ge[which(bias_genre$Genre == x)]))
  vec_bias_yor <- as.numeric(sapply(validation$Year_of_Release, function(x)
    bias_yor$reg_b_yor[which(bias_yor$Year_of_Release == x)]))
  vec_bias_rat <- as.numeric(sapply(validation$Rating, function(x)
    bias_rat$reg_b_rat[which(bias_rat$Rating == x)]))
  vec_bias_dev <- as.numeric(sapply(validation$Developer, function(x)
    bias_dev$reg_b_dev[which(bias_dev$Developer == x)]))
  ###computes the bias for the rows of the training set before fitting a linear model
  t_v_b_pub <- as.numeric(sapply(sub_training$Publisher, function(x)
    bias_pub$reg_b_pub [which(bias_pub$Publisher == x)]))
  t_v_b_pla <- as.numeric(sapply(sub_training$Platform, function(x)
    bias_plat$reg_b_pla [which(bias_plat$Platform == x)]))
  t_v_b_ge <- as.numeric(sapply(sub_training$Genre, function(x)
    bias_genre$reg_b_ge [which(bias_genre$Genre == x)]))
  t_v_b_yor <- as.numeric(sapply(sub_training$Year_of_Release, function(x)
    bias_yor$reg_b_yor [which(bias_yor$Year_of_Release == x)]))
  t_v_b_rat <- as.numeric(sapply(sub_training$Rating, function(x)
    bias_rat$reg_b_rat [which(bias_rat$Rating == x)]))
  t_v_b_dev <- as.numeric(sapply(sub_training$Developer, function(x)
    bias_dev$reg_b_dev[which(bias_dev$Developer == x)]))
  ###fits a linear model for critic and user score
  fitFormula <- logGlobal_Sales -
    (log10_mu + t_v_b_pub + t_v_b_pla + t_v_b_ge +
       t_v_b_yor + t_v_b_rat + t_v_b_dev) ~ Critic_Score + User_Score
  fit <- lm(fitFormula, data = sub_training )
  intercept <- fit$coefficients[1]
  alpha <- fit$coefficients[2]
  beta <- fit$coefficients[3]
  rm(fit)
  ###creation of a subset with scores and new relevant columns
  scores_set <- sub_training %>%
    select(gameId, logGlobal_Sales, Critic_Score, Critic_Count, User_Score, User_Count) %>%
    mutate(sumCritScore = Critic_Count*Critic_Score, sumUserScore = User_Count*User_Score)
  #the two columns added will be useful for regularization
  ###regularization of critic score
  lambdas <- seq(-1, 100, 1)
  #vector of RMSE for each lambda
  RMSEs_train <- sapply(lambdas, function(lam){
    reg_crit_score <- scores_set$sumCritScore/(lam + scores_set$Critic_Count) #regularized critic score
    pred <- 10^(log10_mu + t_v_b_pub + t_v_b_pla + t_v_b_ge +
                  t_v_b_yor + t_v_b_rat + t_v_b_dev + intercept +
                  alpha*reg_crit_score + beta*scores_set$User_Score)
    fun_RMSE(sub_training$Global_Sales, pred)
  })
  #working out the lambda that minimizes training RMSE
  lambda_crit <- lambdas[which.min(RMSEs_train)]
  #adding regularized column to test and train set
  sub_training <- sub_training %>% mutate(Critic_Score_Reg = scores_set$sumCritScore/(lambda_crit + scores_set$Critic_Count))
  validation <- validation %>% mutate(Critic_Score_Reg = (Critic_Score*Critic_Count)/(lambda_crit + Critic_Count))
  ###regularization of user score
  lambdas <- seq(-1, 100, 1)
  #vector of RMSE for each lambda
  RMSEs_train <- sapply(lambdas, function(lam){
    reg_user_score <- scores_set$sumUserScore/(lam + scores_set$User_Count) #regularized critic score
    pred <- 10^(log10_mu + t_v_b_pub + t_v_b_pla + t_v_b_ge +
                  t_v_b_yor + t_v_b_rat + t_v_b_dev + intercept +
                  alpha*sub_training$Critic_Score_Reg + beta*reg_user_score)
    fun_RMSE(sub_training$Global_Sales, pred)
  })
  #working out the lambda that minimizes training RMSE
  lambda_user <- lambdas[which.min(RMSEs_train)]
  #adding regularized column to test and train set
  sub_training <- sub_training %>% mutate(User_Score_Reg = scores_set$sumUserScore/(lambda_user + scores_set$User_Count))
  validation <- validation %>% mutate(User_Score_Reg = (User_Score*User_Count)/(lambda_user + User_Count))
  ###computes the predictions with test set
pred_LM = 10^(log10_mu + vec_bias_pub + vec_bias_pla + vec_bias_ge +
              vec_bias_yor + vec_bias_rat + vec_bias_dev + intercept +
              alpha*validation$Critic_Score_Reg + beta*validation$User_Score_Reg)
  #remove the added columns, free memory
  validation <- validation %>% select(-User_Score_Reg, -Critic_Score_Reg)
  sub_training <- sub_training %>% select(-logGlobal_Sales, -User_Score_Reg, -Critic_Score_Reg)
  rm(bias_dev, bias_genre, bias_plat, bias_pub, bias_rat, bias_yor, scores_set)
  rm(alpha, beta, fitFormula, intercept, lambda_crit, lambda_dev, lambda_ge)
  rm(lambda_plat, lambda_pub, lambda_rat, lambda_user, lambda_yor, lambdas, log10_mu)
  rm(RMSEs_train, t_v_b_dev, t_v_b_ge, t_v_b_pla, t_v_b_pub, t_v_b_rat, t_v_b_yor)
  rm(vec_bias_dev, vec_bias_ge, vec_bias_pla, vec_bias_pub, vec_bias_rat, vec_bias_yor)
  
  
  
#predictions for KNN
  sub_training <- sub_training %>% mutate(Global_Sales)
  ###computing the sum of sales in a given region for each environmental variable
  pub <- sub_training %>% group_by(Publisher) %>%
    summarise(sum = sum(Global_Sales),
              mean = mean(Global_Sales),
              count = n(),
              .groups="drop")
  pla <- sub_training %>% group_by(Platform) %>%
    summarise(sum = sum(Global_Sales),
              mean = mean(Global_Sales),
              count = n(),
              .groups="drop")
  gen <- sub_training %>% group_by(Genre) %>%
    summarise(sum = sum(Global_Sales),
              mean = mean(Global_Sales),
              count = n(),
              .groups="drop")
  yor <- sub_training %>% group_by(Year_of_Release) %>%
    summarise(sum = sum(Global_Sales),
              mean = mean(Global_Sales),
              count = n(),
              .groups="drop")
  rat <- sub_training %>% group_by(Rating) %>%
    summarise(sum = sum(Global_Sales),
              mean = mean(Global_Sales),
              count = n(),
              .groups="drop")
  dev <- sub_training %>% group_by(Developer) %>%
    summarise(sum = sum(Global_Sales),
              mean = mean(Global_Sales),
              count = n(),
              .groups="drop")
  ###adding matching columns to both training and test set
  sub_training$mean_Pub <- as.numeric(sapply(sub_training$Publisher, function(x)
    pub$mean[which(pub$Publisher == x)]))
  sub_training$mean_Pla <- as.numeric(sapply(sub_training$Platform, function(x)
    pla$mean[which(pla$Platform == x)]))
  sub_training$mean_Gen <- as.numeric(sapply(sub_training$Genre, function(x)
    gen$mean[which(gen$Genre == x)]))
  sub_training$mean_Yor <- as.numeric(sapply(sub_training$Year_of_Release, function(x)
    yor$mean[which(yor$Year_of_Release == x)]))
  sub_training$mean_Rat <- as.numeric(sapply(sub_training$Publisher, function(x)
    pub$mean[which(pub$Publisher == x)]))
  sub_training$mean_Dev <- as.numeric(sapply(sub_training$Developer, function(x)
    dev$mean[which(dev$Developer == x)]))
  validation$mean_Pub <- as.numeric(sapply(validation$Publisher, function(x)
    pub$mean[which(pub$Publisher == x)]))
  validation$mean_Pla <- as.numeric(sapply(validation$Platform, function(x)
    pla$mean[which(pla$Platform == x)]))
  validation$mean_Gen <- as.numeric(sapply(validation$Genre, function(x)
    gen$mean[which(gen$Genre == x)]))
  validation$mean_Yor <- as.numeric(sapply(validation$Year_of_Release, function(x)
    yor$mean[which(yor$Year_of_Release == x)]))
  validation$mean_Rat <- as.numeric(sapply(validation$Publisher, function(x)
    pub$mean[which(pub$Publisher == x)]))
  validation$mean_Dev <- as.numeric(sapply(validation$Developer, function(x)
    dev$mean[which(dev$Developer == x)]))
  ###using cross validation to manually select the best K
  set.seed(sub_training$Global_Sales[1], sample.kind = "Rounding")
  Ks <- seq(3,15,2)
  RMSEs_train <- sapply(Ks, function(x){
    #creates cross-validation sets
    indexes <- createDataPartition(sub_training$gameId, times = 1, p = 0.2, list=FALSE) 
    validation <- sub_training[indexes,]
    inner_training <- sub_training[-indexes,]
    train_knn <- train(Global_Sales ~ mean_Pub+mean_Pla+mean_Gen+mean_Yor+mean_Rat+mean_Dev,
                       data = inner_training,
                       method = "knn",
                       tuneGrid = data.frame(k = x))
    pred <- predict(train_knn, validation)
    fun_RMSE(validation$Global_Sales, pred)
  })
  optiK <- Ks[which.min(RMSEs_train)]
  rm(RMSEs_train, Ks)
  ###knn using the mean for each environmental variable
  train_knn <- train(Global_Sales ~ mean_Pub+mean_Pla+mean_Gen+mean_Yor+mean_Rat+mean_Dev,
                     data = sub_training,
                     method = "knn",
                     tuneGrid = data.frame(k = optiK))
pred_KNN <- predict(train_knn, validation)
  ###removing columns added by this function and free memory
  sub_training <- sub_training %>% select(-mean_Pub, -mean_Pla, -mean_Gen, -mean_Yor, -mean_Rat, -mean_Dev)
  validation <- validation %>% select(-mean_Pub, -mean_Pla, -mean_Gen, -mean_Yor, -mean_Rat, -mean_Dev)
  rm(dev, gen, pla, pub, rat, train_knn, yor, optiK)
  
  
###compute the squared error and the RMSE for both KNN and LM
fun_RMSE(validation$Global_Sales, pred_LM)
fun_RMSE(validation$Global_Sales, pred_KNN)
SE_rel_LM <- ( (validation$Global_Sales - pred_LM)/validation$Global_Sales )^2
SE_rel_KNN <- ( (validation$Global_Sales - pred_KNN)/validation$Global_Sales )^2

###EDA
mean(SE_rel_KNN < SE_rel_LM)
df <- data.frame(validation$Global_Sales, SE_rel_LM, SE_rel_KNN)
colors <- c("LM" = "blue", "KNN" = "red")
df %>% ggplot() +
  geom_point(aes(x=validation.Global_Sales, y=SE_rel_KNN, color="KNN")) +
  geom_point(aes(x=validation.Global_Sales, y=SE_rel_LM, color="LM")) +
  xlab("Global Sales") +
  ylab("Relative error with KNN or LM") +
  ggtitle("Accuracy of KNN and LM model versus number of copies sold")
mean(SE_rel_KNN < 5)
mean(SE_rel_LM < 5)
df %>% ggplot() +
  geom_point(aes(x=validation.Global_Sales, y=SE_rel_KNN, color="KNN")) +
  geom_point(aes(x=validation.Global_Sales, y=SE_rel_LM, color="LM")) +
  ylim(0,5) +
  xlab("Global Sales") +
  ylab("Relative error with KNN or LM") +
  ggtitle("Accuracy of KNN and LM model versus number of copies sold")
mean(SE_rel_KNN < 1)
mean(SE_rel_LM < 1)
df %>% ggplot() +
  geom_point(aes(x=validation.Global_Sales, y=SE_rel_KNN, color="KNN")) +
  geom_point(aes(x=validation.Global_Sales, y=SE_rel_LM, color="LM")) +
  ylim(0,5) +
  xlim(0, 10^7) +
  xlab("Global Sales") +
  ylab("Relative error with KNN or LM") +
  ggtitle("Accuracy of KNN and LM model versus number of copies sold") +
  geom_vline(xintercept =  0.09*10^7)
threshold <- 0.09*10^7
mean(SE_rel_KNN[which(validation$Global_Sales > threshold)] < 
       SE_rel_LM[which(validation$Global_Sales > threshold)])

threshs <- seq(0.05*10^7, 0.375*10^7, 0.025*10^7)
means <- sapply(threshs, function(x) 
                  mean(SE_rel_KNN[which(validation$Global_Sales > x)] < 
                        SE_rel_LM[which(validation$Global_Sales > x)]))
ggplot(aes(x=threshs, y=means), data=NULL) +
  geom_point() +
  xlab("Thresholds") +
  ylab("Percentage of KNN errors being less than LM error") +
  ggtitle("Percentage of KNN errors being less than LM error for sales above given a threshold")
max(means)
threshold <- threshs[which.max(means)]
threshold

###free memory
rm(indexes, sub_training, validation, df, colors)
rm(pred_KNN, pred_LM, SE_rel_KNN, SE_rel_LM, means, threshs)


#######################################################
###               Hybrid theory: Worldwide
#######################################################

###making prediction using hybrid theory
pred_LM <- fun_Region_Pred_LM(training$Global_Sales)
pred_KNN <- fun_Region_Pred_KNN(training$Global_Sales)

###combining the predictions
pred <- ifelse( pred_LM>threshold, pred_KNN, pred_LM)
fun_RMSE(test$Global_Sales, pred_KNN)
fun_RMSE(test$Global_Sales, pred_LM)
RMSE <- fun_RMSE(test$Global_Sales, pred)


SE_rel_LM <- ( (test$Global_Sales - pred_LM)/test$Global_Sales )^2
SE_rel_KNN <- ( (test$Global_Sales - pred_KNN)/test$Global_Sales )^2
colors <- c("LM" = "blue", "KNN" = "red")
ggplot(data=test) +
  geom_point(aes(x=Global_Sales, y=SE_rel_KNN, color="KNN")) +
  geom_point(aes(x=Global_Sales, y=SE_rel_LM, color="LM")) +
  ylim(0,5) +
  geom_vline(xintercept =  threshold)

###keeping track of the RMSE
mat_RMSE <- rbind(mat_RMSE, c("Hybrid theory (LM under a given threshold, KNN above said threshold): Worldwide", RMSE))
rm(pred, RMSE, threshold, pred_LM, pred_KNN, colors, SE_rel_KNN, SE_rel_LM)

#######################################################
###               Conclusion
#######################################################

mat_RMSE
"Best model:"
mat_RMSE[which.min(mat_RMSE[,2]), ]
