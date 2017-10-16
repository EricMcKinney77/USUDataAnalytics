#This is the Titanic Kaggle Competition Script for USU Data Analytics 2017

# install.packages("randomForest") #Uncomment if you haven't installed the Random Forest Package
library(randomForest)

getwd()

# Import the data
train = read.csv("~/Titanic2017/train.csv")
test = read.csv("~/Titanic2017/test.csv")

# Exploratory data analysis
head(train)
tail(train)
names(train)
names(test)
View(train)
View(test)

class(train) #data frame
summary(train)

# Create a prediction variable for the training and test data
train$Prediction = 0
head(train)
test$Prediction = 0
head(test)

# machine learning/data mining
table(train$Sex, train$Survived)
prop.table(table(train$Sex, train$Survived))





#Random Forest benchmark model
rfmodel1 <- randomForest(as.factor(Survived) ~ Sex + Pclass + SibSp + Parch + , data = train)
rfmodel1

test$rf1prediction <- predict(rfmodel1, test)
test$rf1prediction

# Create a prediction data frame to submit
rfpred = NULL
rfpred$PassengerId = test$PassengerId #the csv only needs PassengerID and my Survived
rfpred$Survived = test$rf1prediction #create my guess at who survived
rfpred = as.data.frame(rfpred)
rfpred

# Save a csv file of test to submit to Kaggle
write.csv(rfpred, file = "~/Titanic2017/rfpred.csv", row.names = FALSE) #Change file path as appropriate




# Guess at characteristics of those who survived
train$Prediction[train$Sex == "female"] = 1 #all women
train$Prediction

# See how well this would work in the test set.
table(train$Prediction, train$Survived)
prop.table(table(train$Prediction, train$Survived))

# Apply the same methods to the test set to prepare for submission
test$Prediction[test$Sex == "female"] = 1 #all women
test

# Create a prediction data frame to submit
pred = NULL
pred$PassengerId = test$PassengerId #the csv only needs PassengerID and my Survived
pred$Survived = test$Prediction #create my guess at who survived
pred = as.data.frame(pred)
pred

# Save a csv file of test to submit to Kaggle
write.csv(pred, file = "~/Titanic2017/pred.csv", row.names = FALSE) #Change file path as appropriate
