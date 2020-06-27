library(dplyr) 
library(plyr) 
library(read_csv) 
library(readr) 
library(MASS) 
library(caret) # Machine Learning 
library(car) # VIF and QQ plot 
library(tidyr) 
library(tseries) # Unit root test for time series 
library(forecast) # ARIMA 
### LOGISTIC MODEL 
# Supress scientific notation 
options(scipen = 4) 
# Import main data and summoner spell data 
file <- "filecast/challenger.csv" 
data <- read_csv(file) 
length(which(!complete.cases(data)))

data <- data %>% drop_na() # Remove NA rows 
# Convert binary variables to factor 
View(data)

data$win <- as.factor(data$win) 
data$firstBlood <- as.factor(data$firstBlood) 
data$firstTower <- as.factor(data$firstBlood) 
# Partition data into two sets, training and validation 
set.seed(61) 
trainSamples <- createDataPartition(data$win, p = .8, list = F) 
trainData <- data[trainSamples,] 
validationData <- data[-trainSamples,] 
# Create Model 
model <- glm(win~., family = binomial(link = 'logit'), data = trainData) %>% 
  stepAIC(trace = F) 
summary(model) 
plot(rstandard(model)) # plot the standardized residuals 
# Check model accuracy 
validationData$probability <- predict(model, validationData, type = "response") 
# If calculated blue team win % > 50, set to 1 
validationData <- validationData %>% mutate(prediction = 1*(probability > 0.5), actual = 1*(win == "1")) 
validationData <- validationData %>% mutate(accuracy = 1*(prediction == actual)) 
sum(validationData$accuracy)/nrow(validationData) # Calculate accuracy 
View(validationData) 
vif(model) 
binnedplot(fitted(model), 
           residuals(model, type = "response"), 
           nclass = NULL, 
           xlab = "Expected Values", 
           ylab = "Average residual", 
           main = "Binned residual plot", 
           cex.pts = 0.8, 
           col.pts = 1, 
           col.int = "gray") 
# Machine learning models 
# Run algorithms using 10-fold cross validation 
control <- trainControl(method = "cv", number = 10) 
#metric <- "Accuracy"
fit.knn <- train(win~., data = trainData, method = "knn", metric = accuracy(), trControl = control) 
fit.lda <- train(win~., data = trainData, method = "lda", metric = accuracy(), trControl = control) 
fit.cart <- train(win~., data = trainData, method = "rpart", metric = accuracy(), trControl = control) 
fit.rf <- train(win~., data = trainData, method = "rf", metric = accuracy(), trcontrol = control ) 
results <- resamples(list(knn = fit.knn, lda = fit.lda, cart = fit.cart, rf = fit.rf)) 
summary(results) 
predictions <- predict(fit.lda, validationData) 
confusionMatrix(predictions, validationData$win)


set.seed(48743)
cv.folds <- createMultiFolds(data$firstBlood, k = 10, times = 3)

cv.cntrl <- trainControl(method = "repeatedcv", number = 10,
                         repeats = 3, index = cv.folds)
install.packages("dplyr")
install.packages("doSNOW")
library(doSNOW)

start.time <- Sys.time()

cl <- makeCluster(3, type = "SOCK")
registerDoSNOW(cl)

rpart.cv.1 <- train(firstBlood ~ ., data = trainData, method = "rpart", 
                    trControl = cv.cntrl, tuneLength = 7)

stopCluster(cl)

total.time <- Sys.time() - start.time
total.time

rpart.cv.1






# TIME SERIES MODEL 
file2 <- "C:/Users/aaron/Documents/LeagueData/RedditSubscribers.csv"
file2 <- "filecast/challenger.csv"
subscribers <- read_csv(file2) 
subscribers <- subscribers[,3] # Use log-normalized data 
# Transform data into time series data 
subscribers <- ts(subscribers, start = c(2012,11), frequency = 12) 
# Use additive since data is log-normalized 
components <- decompose(subscribers, type = 'additive') 
plot(components) 
# Unit root test for stationarity 
# Null hypothesis is stationarity, small p-value rejects null 
kpss.test(subscribers, null = "Level") # Data appears to be non-stationary 
# Auti-ARIMA model with lowest AIC 
arimaModel <- auto.arima(subscribers) 
arimaModel 
qqPlot(arimaModel$residuals) 
predict(arimaModel, n.ahead = 15) 
acf(arimaModel$residuals) 
pacf(arimaModel$residuals) 
forecast <- forecast(arimaModel, h = 15, level = c(99.5)) # add axis 
plot(forecast, ylab = "Log-transformed value")
