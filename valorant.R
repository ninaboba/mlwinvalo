# Connect R to database
library(RSQLite)

# 1. Connect DB
conn <- dbConnect(SQLite(), "data/valorant.sqlite")
# 2. List Table
dbListTables(conn)

# 3. List Fields
dbListFields(conn, "Games")

# 4. Query data
Games <- dbGetQuery(conn, "SELECT * 
                    FROM Games LIMIT 1001")
dbDisconnect(conn)

#  load tidyverse library and caret for data transformation and machine learning in advance
library(tidyverse)
library(caret)
library(ranger)
library(caTools) #for logistic regression

# find total NA values
summary(Games)
sum(is.na(Games))
Games <- na.omit(Games)

# data transformation
names(Games)
  ## since I want to see Team 1 winning conditions so I cut other non Team 1 related column
tGames <- Games %>%
  select(Map, Team1, Team2, Winner, starts_with("Team1_")) %>%
  mutate(Team1_Win = if_else(Team1 == Winner, 1, 0),
         Team1_Attack = if_else(Team1_SideFirstHalf == "attack", 1, 0))
         # create 2 new columns

  ## select the working data
tGames2 <- tGames %>%
  select(starts_with("Team1_"), -Team1_SideFirstHalf)

# visulize data to see more correlated value with correlation matrix
library(corrplot) #correlation plot matrix
library(RColorBrewer)
M <- cor(tGames2)
corrplot(M, type="upper", order="hclust",
         col=brewer.pal(n=8, name="RdYlBu"))

# explore data
library(explore) # library for exploring
explore(tGames2) # interactive data 

## change column type from numeric to factor
tGames2 <- tGames2 %>%
  select(starts_with("Team1_")) %>%
  mutate(Team1_Win = as.factor(Team1_Win),
         Team1_Attack = as.factor(Team1_Attack))
    # create 2 new columns and change their type to factor

# split data for training and testing
set.seed(11)
id <- createDataPartition(tGames2$Team1_Win, p = 0.8, list = F)

train_data <- tGames2[id, ] #train80%
test_data <- tGames2[-id, ] #test20%

#set train control to k-fold cv
set.seed(11)
ctrl <- trainControl(method = "cv",
                     number = 5,
                     verboseIter = TRUE)

# build models
set.seed(11)
logit_model <- train(Team1_Win~.,
                   data = train_data,
                   method = "LogitBoost",
                   metric = "Accuracy",
                   trControl = ctrl)
    ##CART
set.seed(11)
cart_model <- train(Team1_Win~.,
                     data = train_data,
                     method = "rpart",
                     metric = "Accuracy",
                     trControl = ctrl)
    ##knn
set.seed(11)
knn_model <- train(Team1_Win~.,
                     data = train_data,
                     method = "knn",
                     metric = "Accuracy",
                     trControl = ctrl)
    ## Randomforest
set.seed(11)
ranger_model <- train(Team1_Win~.,
                     data = train_data,
                     method = "ranger",
                     metric = "Accuracy",
                     trControl = ctrl)
# select best model
  ## summarise accuracy of models
results <- resamples(list(lda = lda_model,
                          logit = logit_model,
                          cart = cart_model,
                          knn = knn_model,
                          ranger = ranger_model))
summary(results)
  ## compare accuracy of models
dotplot(results)
  ## summarise bets model
print(ranger_model)

# score and evaluate
p <- predict(ranger_model, test_data)
cmatrix <- confusionMatrix(p, test_data$Team1_Win)

print(cmatrix)
