# This is the Titanic Kaggle Competition Script for USU Data Analytics 2017
# Many of the data preparation techniques used (and improved upon) came from 
# Megan Risdal's Kaggle Kernel https://www.kaggle.com/mrisdal/exploring-survival-on-the-titanic


library(ggplot2) # visualization
library(ggthemes) # visualization
library(scales) # visualization
library(dplyr) # Data manipulation
library(mice) # Missing value imputation
library(randomForest) # Classification algorithm
library(verification)
library(ada)
library(gbm)
library(rpart)
library(rpart.plot)
library(MASS)
library(caret)
library(e1071)
library(utils)
source("kappa and classsum.R") # A function borrowed from Dr. Richard Cutler
library(corrplot)

getwd()

# Imported the data
train <- read.csv("train.csv") # Change the file path if necessary
test <- read.csv("test.csv")

full <- bind_rows(train, test)

# Exploratory data analysis
class(full) # Data frame
summary(full)
str(full) # Similar view in R studio's global environment window
head(full)
tail(full)
names(full)
View(full)

table(train$Sex, train$Survived)
prop.table(table(train$Sex, train$Survived))

#Random Forest benchmark model
rfbmmodel <- randomForest(as.factor(Survived) ~ Sex + Pclass + SibSp + Parch, data = train)

rfbmpred <- predict(rfbmmodel, test)

# Save the solution to a dataframe with two columns: PassengerId and Survived (prediction)
rfbmpred <- data.frame(PassengerID = test$PassengerId, Survived = rfbmpred)

# Write the solution to file
write.csv(rfbmpred, file = 'rfbmpred.csv', row.names = FALSE) # Kaggle PCC 0.77511


####################### Extracting More Variables From Names ##################################

# Get titles form passenger names gsub is a command to match regular expressions
# Each comma groups things together
full$Title <- gsub('(.*, )|(\\..*)', '', full$Name) # Replaced everything surrounding the titles with nothing.
table(full$Sex, full$Title)

# Put all titles that are rare (have low ocurrences) into a single variable
rare_title <- c('Dona', 'Lady', 'the Countess','Capt', 'Col', 'Don', 
                'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer')

# Reassign titles to what we think they should be
full$Title[full$Title == 'Mlle']        <- 'Miss' # Mlle was the french Medemoiselle
full$Title[full$Title == 'Ms']          <- 'Miss' # Ms was often used for unmarried older women
full$Title[full$Title == 'Mme']         <- 'Mrs'  # Mme was the french Madame
full$Title[full$Title %in% rare_title]  <- 'Rare Title'

# This made our titles list a lot smaller
table(full$Sex, full$Title)

full$Surname <- sapply(full$Name,  
                       function(x) strsplit(x, split = '[,.]')[[1]][1])

# Splitting the names gave us more variables that we can play with
str(full)

# We can assume that 66% of passengers didn't come with family onboard
total.surnames = length(full$Surname)
unique.surnames = length(unique(full$Surname))

perentage.of.passengers.with.unique.surname = unique.surnames/total.surnames * 100

########################## Do Families Sink Or Swim Together? ####################################

# Make a new variable that gives the family size of each passenger (1 is a passenger alone)
full$Fsize <- full$SibSp + full$Parch + 1

# Create a family variable that shows the surname along with how many members are onboard
full$Family <- paste(full$Surname, full$Fsize, sep='_')

# Use ggplot2 to visualize the relationship between family size & survival
# From this plot we can see that if you are single there was almost twice as much people who didn't survive

ggplot(full[1:891,], aes(x = Fsize, fill = factor(Survived))) +
  geom_bar(stat='count', position='dodge') +
  scale_x_continuous(breaks=c(1:11)) +
  labs(x = 'Family Size') +
  theme_few()

# Collapse family size into 3 variables single, small (1,5), and large [5,infinity)
full$FsizeD[full$Fsize == 1] <- 'singleton'
full$FsizeD[full$Fsize < 5 & full$Fsize > 1] <- 'small'
full$FsizeD[full$Fsize > 4] <- 'large'

# Mosaic Plot shows that if you have a large family or are a singleton you are more likely to perish
mosaicplot(table(full$FsizeD, full$Survived), main='Family Size by Survival', shade=TRUE)

################################ Treat more Variables ##########################################
# There are many missing values for the Cabin number
full$Cabin[1:28]

# Get the passenger Deck, which is the first character of the string 
full$Deck <- factor(sapply(full$Cabin, function(x) strsplit(x, NULL)[[1]][1]))

####################################### Missingness ###############################################
# Since the data set is small it isn't a good idea to delete entire rows with missing values
# We can instead infer what those values should have been

# These two passengers have a missing value for "Embarked"
full[c(62, 830), 'Embarked']

# Get rid of our missing passenger IDs
embark_fare <- full %>%
  filter(PassengerId != 62 & PassengerId != 830)

# Use ggplot2 to visualize embarkment, passenger class, & median fare to replace
# missing values with a variable
# This shows that the median value of people in first class that departed from Charbourg
# payed a median value of 80 dollars. 
ggplot(embark_fare, aes(x = Embarked, y = Fare, fill = factor(Pclass))) +
  geom_boxplot() +
  geom_hline(aes(yintercept=80), 
             colour='red', linetype='dashed', lwd=2) +
  scale_y_continuous(labels=dollar_format()) +
  theme_few()

# We can then safely replace the embarked value from persons 62 and 830 with 'C' 
# Since their fare was $80 for 1st class, they most likely embarked from 'C'
full$Embarked[c(62, 830)] <- 'C'

# We can assume that person 830 also embarked from Charbourg since they had the same ticket number
full$Ticket[62] == full$Ticket[830]

# We will infer their values for embarkment based on present data that we can imagine may be relevant: 
# passenger class and fare. We see that they paid $ 80 and $ NA respectively and their classes are 1 
# and NA . So from where did they embark?


# Passenger 1044 had a missing Fare value
full$Fare[1044]

# To find a value to replace this passengers fare with we can again find the median value of
# people who shared class and embarkment and hopefully find a value to replace this NA value with
ggplot(full[full$Pclass == '3' & full$Embarked == 'S', ], 
       aes(x = Fare)) +
  geom_density(fill = '#99d6ff', alpha=0.4) + 
  geom_vline(aes(xintercept=median(Fare, na.rm=T)),
             colour='red', linetype='dashed', lwd=1) +
  scale_x_continuous(labels=dollar_format()) +
  theme_few()

# The plot showed the median value was right where the majority of passengers paid, so we can
# assume our passenger paid the same amount and replace the NA value

full$Fare[1044] <- median(full[full$Pclass == '3' & full$Embarked == 'S', ]$Fare, na.rm = TRUE)


############################# Predictive imputation ############################################

# There are a lot of missing age values in our dataset
sum(is.na(full$Age))

# Uses the mice package to to predict ages in the model. She also suggested using recursive partitioning
# for regression

# First turn variables into factors
factor_vars <- c('PassengerId','Pclass','Sex','Embarked',
                 'Title','Surname','Family','FsizeD')
full[factor_vars] <- lapply(full[factor_vars], function(x) as.factor(x))

# Set a random seed
set.seed(129)

# Random forest intside mice to predict ages
mice_mod <- mice(full[, !names(full) %in% c('PassengerId','Name','Ticket','Cabin','Family','Surname','Survived')], method='rf') 

# save output from mice_mod
mice_output <- complete(mice_mod)

# To check that our new predicted ages aren't completely different from the old ages we can 
# look at the distribution of our old set and new set to see if they aren't similar

par(mfrow=c(1,2))
hist(full$Age, freq=F, main='Age: Original Data', 
     col='darkgreen', ylim=c(0,0.04))
hist(mice_output$Age, freq=F, main='Age: MICE Output', 
     col='lightgreen', ylim=c(0,0.04))

# The histograms look almost identical so it is okay for us to replace the ages in the dataset
# with our new predicted ages

full$Age <- mice_output$Age

# There are no longer any missing age values
sum(is.na(full$Age))


######################### Create some new variables Child and Mother ###############################

# A child is someone under the age of 18 and a mother is female, has more than 0 children, and 
# does not have the title "Miss"

# Looks at the relationship between age and survival between men and woman. It looks 
# pretty much normal for people sho survived. Possibly a bit right skewed
ggplot(full[1:891,], aes(Age, fill = factor(Survived))) + 
  geom_histogram() + 
  # I include Sex since we know (a priori) it's a significant predictor
  facet_grid(.~Sex) + 
  theme_few()

# Create a variable for child and adult
full$Child[full$Age < 18] <- 'Child'
full$Child[full$Age >= 18] <- 'Adult'

# it looks like being a child could has an impact on a passengers chance at survival, but not that much
table(full$Child, full$Survived)

# Adding mother variable
full$Mother <- 'Not Mother'
full$Mother[full$Sex == 'female' & full$Parch > 0 & full$Age > 18 & full$Title != 'Miss'] <- 'Mother'

# It looks like this might have helped as well, but there are so few mothers on board
table(full$Mother, full$Survived)

# Factorize two new factor variables
# Finish by factorizing our two new factor variables
full$Child  <- factor(full$Child)
full$Mother <- factor(full$Mother)

# It looks like we replaced the relevant NA values
md.pattern(full)

############################ Finally let's do some prediction #####################################

# Split data back into testing and training datasets
train <- full[1:891,]
test <- full[892:1309,]

# Use random forest to build our model with some of the variables we created and some old variables
rf_model <- randomForest(factor(Survived) ~ Pclass + Sex + Age + SibSp + Parch + 
                           Fare + Embarked + Title + 
                           FsizeD + Child + Mother,
                         data = train)

# Shows the error rate of the model we made. Black line is overall error, green error for
# survived and red is error for deceased
par(mfrow=c(1,1))
plot(rf_model, ylim=c(0,0.36))
legend('topright', colnames(rf_model$err.rate), col=1:3, fill=1:3)


#################################### Variable Importance ############################################

# Get importance from MeanDecreasedGini 
importance    <- importance(rf_model)
varImportance <- data.frame(Variables = row.names(importance), 
                            Importance = round(importance[ ,'MeanDecreaseGini'],2))

# Create a rank variable based on importance
rankImportance <- varImportance %>%
  mutate(Rank = paste0('#',dense_rank(desc(Importance))))

# Use ggplot2 to visualize the relative importance of variables
ggplot(rankImportance, aes(x = reorder(Variables, Importance), 
                           y = Importance, fill = Importance)) +
  geom_bar(stat='identity') + 
  geom_text(aes(x = Variables, y = 0.5, label = Rank),
            hjust=0, vjust=0.55, size = 4, colour = 'red') +
  labs(x = 'Variables') +
  coord_flip() + 
  theme_few()

# Pretty cool that the highest relative importance of all the predictor variables was the title variable
# we created

############################ Making a solution file ############################
# Predict using the test set
prediction <- predict(rf_model, test)

# Save the solution to a dataframe with two columns: PassengerId and Survived (prediction)
rfpred <- data.frame(PassengerID = test$PassengerId, Survived = prediction)

# Write the solution to file
write.csv(rfpred, file = 'rfpred.csv', row.names = FALSE) # Kaggle PCC 0.79425






########################### Model Building ####################################

# Logistic Regression
Survived.lr = glm(factor(Survived) ~ Pclass + Sex + Age + SibSp + Parch + 
                       Fare + Embarked + Title + 
                       FsizeD + Child + Mother, family = binomial, data = train)

table(train$Survived, round(predict(Survived.lr, type = "response") + 0.0000001))
class.sum(train$Survived, predict(Survived.lr, type = "response"))
lrrsb = class.sum(train$Survived, predict(Survived.lr, type = "response"))[1, 2]

Survived.lr.xval = rep(0, length = nrow(train))
xvs = rep(1:10, length = nrow(train))
xvs = sample(xvs)
for (i in 1:10) {
  xval.train = train[xvs != i, ]
  xval.test = train[xvs == i, ]
  mymodel = glm(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + Title + FsizeD + Child + Mother, family = binomial, data = xval.train)
  Survived.lr.xval[xvs == i] = predict(mymodel, test, type="response")
}

table(train$Survived, round(Survived.lr.xval))
class.sum(train$Survived, Survived.lr.xval)
lrfld = class.sum(train$Survived, Survived.lr.xval)[1, 2]

table(test$Survived, round(predict(Survived.lr, test, type = "response") + 0.0000001))
class.sum(test$Survived, predict(Survived.lr, test, type = "response"))
lrtst = class.sum(test$Survived, predict(Survived.lr, test, type = "response"))[1, 2]


# k-nearest neighbors found best of k = 2, 3, 4, 5, 6, 7, 8
nNeighbors = 5
Survived.knn = knn3(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + Title + FsizeD + Child + Mother, data = train, k = nNeighbors)

# Resubstitution confusion matrix and accuracies.
table(train$Survived, round(predict(Survived.knn, train, type = "prob")[, 2]))
class.sum(train$Survived, round(predict(Survived.knn, train, type = "prob")[, 2]))
knnrsb = class.sum(train$Survived, round(predict(Survived.knn, train, type = "prob")[, 2]))[1, 2]

Survived.knn.xval = rep(0, length = nrow(train))
xvs = rep(c(1:10), length = nrow(train))
xvs = sample(xvs)
for (i in 1:10) {
  xval.train = train[xvs != i, ]
  xval.test = train[xvs == i, ]
  mymodel = knn3(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + Title + FsizeD + Child + Mother, data = train, k = nNeighbors)
  Survived.knn.xval[xvs == i] = predict(mymodel, test, type = "prob")[, 2]
}

table(train$Survived, round(Survived.knn.xval))
class.sum(train$Survived, round(Survived.knn.xval))
knnfld = class.sum(train$Survived, round(Survived.knn.xval))[1, 2]

table(test$Survived, round(predict(Survived.knn, test, type = "prob")[, 2]))
class.sum(test$Survived, round(predict(Survived.knn, test, type = "prob")[, 2]))
knntst = class.sum(test$Survived, round(predict(Survived.knn, test, type = "prob")[, 2]))[1, 2]


# Single Classification Tree
Survived.rpartfull = rpart(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + Title + FsizeD + Child + Mother, method = "class", control = rpart.control(cp = 0.0, minsplit = 2), data = train)
par(mfcol = c(2, 1))
plot(Survived.rpartfull, main = "Fully Grown Tree")
plotcp(Survived.rpartfull)

Survived.rpartCP024 = rpart(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + Title + FsizeD + Child + Mother, method = "class", control=rpart.control(cp=0.024), data = train)
prp(Survived.rpartCP024, Margin = 0.1, varlen = 0, extra = 1, tweak = .9)
# Survived.rpartCP024

table(train$Survived, predict(Survived.rpartCP024, type = "class"))
class.sum(train$Survived, predict(Survived.rpartCP024, type = "prob")[, 2])
crtrsb = class.sum(train$Survived, predict(Survived.rpartCP024, type = "prob")[, 2])[1, 2]

Survived.rpartCP024.xval = rep(0, length(nrow(train)))
xvs = rep(c(1:10), length = nrow(train))
xvs = sample(xvs)
for(i in 1:10) {
  xval.train = train[xvs != i, ]
  xval.test = train[xvs == i, ]
  rp = rpart(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + Title + FsizeD + Child + Mother, method = "class", data = train, control = rpart.control(cp=0.024))
  Survived.rpartCP024.xval[xvs == i] = predict(rp, test, type = "prob")[, 2]
}

table(train$Survived, round(Survived.rpartCP024.xval))
class.sum(train$Survived, Survived.rpartCP024.xval)
crtfld = class.sum(train$Survived, Survived.rpartCP024.xval)[1, 2]

table(test$Survived, predict(Survived.rpartCP024, test, type = "class"))
class.sum(test$Survived, predict(Survived.rpartCP024, test, type = "prob")[, 2])
crttst = class.sum(test$Survived, predict(Survived.rpartCP024, test, type = "prob")[, 2])[1, 2]


# ada boost
Survived.ada = ada(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + Title + FsizeD + Child + Mother, data = train, loss = "exponential")

Survived.ada$confusion
class.sum(train$Survived, predict(Survived.ada, train, type = "prob")[, 2])
adarsb = class.sum(train$Survived, predict(Survived.ada, train, type = "prob")[, 2])[1, 2]

Survived.ada.xval.class = rep(0, length = nrow(train))
Survived.ada.xval.prob = rep(0, length = nrow(train))
xvs = rep(1:10, length = nrow(train))
xvs = sample(xvs)
for (i in 1:10) {
  xval.train = train[xvs != i, ]
  xval.test = train[xvs == i, ]
  mymodel = ada(as.factor(Survived) ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + Title + FsizeD + Child + Mother, data = train, loss = "exponential")
  Survived.ada.xval.class[xvs == i] = predict(mymodel, test, type = "vector")
  Survived.ada.xval.prob[xvs == i] = predict(mymodel, test, type = "probs")[, 2]
}

table(train$Survived, Survived.ada.xval.class)
class.sum(train$Survived, Survived.ada.xval.prob)
adafld = as.numeric(class.sum(train$Survived, Survived.ada.xval.prob)[1, 2])

table(test$Survived, round(predict(Survived.ada, test, type = "prob")[, 2]))
class.sum(test$Survived, predict(Survived.ada, test, type = "prob")[, 2])
adatst = class.sum(test$Survived, predict(Survived.ada, test, type = "prob")[, 2])[1, 2]


# Support Vector Machines
Survived.svm = svm(as.factor(Survived) ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + Title + FsizeD + Child + Mother, probability = TRUE, data = train)

Survived.svm.resubpred = predict(Survived.svm, train, probability = TRUE)
table(train$Survived, round(attr(Survived.svm.resubpred, "probabilities")[, 2]))
class.sum(train$Survived, attr(Survived.svm.resubpred, "probabilities")[, 2])
svmrsb = class.sum(train$Survived, attr(Survived.svm.resubpred, "probabilities")[, 2])[1, 2]

Survived.svm.xvalpred = rep(0, nrow(train))
xvs = rep(1:10, length = nrow(train))
xvs = sample(xvs)
for (i in 1:10) {
  xval.train = train[xvs != i, ]
  xval.test = train[xvs == i, ]
  mymodel = svm(as.factor(Survived) ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + Title + FsizeD + Child + Mother, probability = TRUE, data = train)
  Survived.svm.xvalpred[xvs == i] = attr(predict(mymodel, test, probability = TRUE), "probabilities")[, 2]
}

table(train$Survived, round(Survived.svm.xvalpred))
class.sum(train$Survived, Survived.svm.xvalpred)
svmfld = class.sum(train$Survived, Survived.svm.xvalpred)[1, 2]

table(test$Survived, round(attr(predict(Survived.svm, test, probability = TRUE), "probabilities")[, 2]))
class.sum(test$Survived, attr(predict(Survived.svm, test, probability = TRUE), "probabilities")[, 2])
svmtst = class.sum(test$Survived, attr(predict(Survived.svm, test, probability = TRUE), "probabilities")[, 2])[1, 2]


# Untuned GBM
Survived.gbm = gbm(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + Title + FsizeD + Child + Mother, distribution = "bernoulli", n.trees = 100, interaction.depth = 1, shrinkage = 0.01, data = train)

table(train$Survived, round(predict(Survived.gbm, type = "response", n.trees = 100, interaction.depth = 1, shrinkage = 0.01) + 0.0000001))
class.sum(train$Survived, predict(Survived.gbm, type = "response", n.trees = 100, interaction.depth = 1, shrinkage = 0.01))
gbmrsb = class.sum(train$Survived, predict(Survived.gbm, type = "response", n.trees = 100, interaction.depth = 1, shrinkage = 0.01))[1, 2]

Survived.gbm.xvalpr = rep(0, nrow(train))
xvs = rep(1:10, length = nrow(train))
xvs = sample(xvs)
for (i in 1:10) {
  xval.train = train[xvs != i, ]
  xval.test = train[xvs == i, ]
  mymodel = gbm(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + Title + FsizeD + Child + Mother, distribution = "bernoulli", n.trees = 100, interaction.depth = 1, shrinkage = 0.01, data = train)
  Survived.gbm.xvalpr[xvs == i] = predict(mymodel, newdata = test, type = "response", n.trees = 100, interaction.depth = 1, shrinkage = 0.01)
}

table(train$Survived, round(Survived.gbm.xvalpr + 0.0000001))
class.sum(train$Survived, Survived.gbm.xvalpr)
gbmfld = class.sum(train$Survived, Survived.gbm.xvalpr)[1, 2]

table(test$Survived, round(predict(Survived.gbm, test, type = "response", n.trees = 100, interaction.depth = 1, shrinkage = 0.01) + 0.0000001))
class.sum(test$Survived, predict(Survived.gbm, test, type = "response", n.trees = 100, interaction.depth = 1, shrinkage = 0.01))
gbmtst = class.sum(test$Survived, predict(Survived.gbm, test, type = "response", n.trees = 100, interaction.depth = 1, shrinkage = 0.01))[1, 2]


# Tuning GBM
fitControl = trainControl(method = "cv", number = 10)
gbmGridNotes = expand.grid(interaction.depth = c(12, 14, 16, 18, 20), n.trees = c(25, 50, 75, 100), shrinkage = c(0.01, 0.05, 0.1, 0.2), n.minobsinnode = 10)
gbmFitNotes = train(as.factor(Survived) ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + Title + FsizeD + Child + Mother, method = "gbm", tuneGrid = gbmGridNotes, trControl = fitControl, data = train)
gbmFitNotes

gbmGridNotes = expand.grid(interaction.depth = c(19, 20, 21, 24), n.trees = c(40, 50, 60), shrinkage = c(0.1, 0.2, 0.3), n.minobsinnode = 10)
gbmFitNotes = train(as.factor(Survived) ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + Title + FsizeD + Child + Mother, method = "gbm", tuneGrid = gbmGridNotes, trControl = fitControl, data = train)
gbmFitNotes

gbmGridNotes = expand.grid(interaction.depth = c(20), n.trees = c(55, 60, 70, 80), shrinkage = c(0.3, 0.4, 0.5, 0.6), n.minobsinnode = 10)
gbmFitNotes = train(as.factor(Survived) ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + Title + FsizeD + Child + Mother, method = "gbm", tuneGrid = gbmGridNotes, trControl = fitControl, data = train)
gbmFitNotes

gbmGridNotes = expand.grid(interaction.depth = c(16), n.trees = c(65, 70, 75), shrinkage = c(0.6, 0.7, 0.8, 0.9), n.minobsinnode = 10)
gbmFitNotes = train(as.factor(Survived) ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + Title + FsizeD + Child + Mother, method = "gbm", tuneGrid = gbmGridNotes, trControl = fitControl, data = train)
gbmFitNotes

gbmGridNotes = expand.grid(interaction.depth = c(16), n.trees = c(65), shrinkage = c(0.75, 0.8, 0.85), n.minobsinnode = 10)
gbmFitNotes = train(as.factor(Survived) ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + Title + FsizeD + Child + Mother, method = "gbm", tuneGrid = gbmGridNotes, trControl = fitControl, data = train)
gbmFitNotes


# Tuned GBM
Survived.tgbm = gbm(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + Title + FsizeD + Child + Mother, distribution = "bernoulli", n.trees = 65, interaction.depth = 16, shrinkage = 0.85, data = train)

table(train$Survived, round(predict(Survived.tgbm, type = "response", n.trees = 65, interaction.depth = 16, shrinkage = 0.85) + 0.0000001))
class.sum(train$Survived, predict(Survived.tgbm, type = "response", n.trees = 65, interaction.depth = 16, shrinkage = 0.85))
tgbmrsb = class.sum(train$Survived, predict(Survived.tgbm, type = "response", n.trees = 65, interaction.depth = 16, shrinkage = 0.85))[1, 2]

Survived.tgbm.xvalpr = rep(0, nrow(train))
xvs = rep(1:10, length = nrow(train))
xvs = sample(xvs)
for (i in 1:10) {
  xval.train = train[xvs != i, ]
  xval.test = train[xvs == i, ]
  mymodel = gbm(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + Title + FsizeD + Child + Mother, distribution = "bernoulli", n.trees = 65, interaction.depth = 16, shrinkage = 0.85, data = train)
  Survived.tgbm.xvalpr[xvs == i] = predict(mymodel, newdata = test, type = "response", n.trees = 65, interaction.depth = 16, shrinkage = 0.85)
}

table(train$Survived, round(Survived.tgbm.xvalpr + 0.0000001))
class.sum(train$Survived, Survived.tgbm.xvalpr)
tgbmfld = class.sum(train$Survived, Survived.tgbm.xvalpr)[1, 2]

table(test$Survived, round(predict(Survived.tgbm, test, type = "response", n.trees = 65, interaction.depth = 16, shrinkage = 0.85) + 0.0000001))
class.sum(test$Survived, predict(Survived.tgbm, test, type = "response", n.trees = 65, interaction.depth = 16, shrinkage = 0.85))
tgbmtst = class.sum(test$Survived, predict(Survived.tgbm, test, type = "response", n.trees = 65, interaction.depth = 16, shrinkage = 0.85))[1, 2]


# xvalTabl <- matrix(c(ldarsb, ldafld, ldatst, qdarsb, qdafld, qdatst, lrrsb, lrfld, lrtst, knnrsb, knnfld, knntst, crtrsb, crtfld, crttst, rfrsb, rffld, rftst, adarsb, adafld, adatst, svmrsb, svmfld, svmtst, gbmrsb, gbmfld, gbmtst, tgbmrsb, tgbmfld, tgbmtst), ncol = 3, byrow = TRUE)
# colnames(xvalTabl) <- c("Resubstitution", "10-fold", "Test data")
# rownames(xvalTabl) <- c("LDA", "QDA", "Logistic Regression", "Nearest Neighbor", "Classification Tree", "Random Forests", "Adaboosted Trees", "Support Vector Machines", "Gradient Boosted Trees", "Tuned Gradient Boosted Trees")
# xvalTabl <- as.table(xvalTabl)
# kable(xvalTabl, digits = 2, caption = "Cross Validated PCC's")





