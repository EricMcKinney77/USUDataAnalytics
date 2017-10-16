# Goes over the tutorial at https://www.kaggle.com/mrisdal/exploring-survival-on-the-titanic
# Load Libaries
library('ggplot2') # visualization
library('ggthemes') # visualization
library('scales') # visualization
library('dplyr') # data manipulation
library('mice') # imputation
library('randomForest') # classification algorithm

# Load the Titanic Testing and Training Datasets
train <- read.csv('train.csv', stringsAsFactors = F)
test <- read.csv('test.csv', stringsAsFactors = F)

# Combine training and testing Dataset to have a full dataset
full <- bind_rows(train,test)

# look what's in the data as coimpact strings
str(full)

# What do these variables mean?

# Survived: 	Survived (1) or died (0)
# Pclass:	    Passenger’s class
# Name"	      Passenger’s name
# Sex:	      Passenger’s sex
# Age:	      Passenger’s age
# SibSp:	    Number of siblings/spouses aboard
# Parch:	    Number of parents/children aboard
# Ticket:	    Ticket number
# Fare:	      Fare
# Cabin:	    Cabin
# Embarked: 	Port of embarkation

####################### Extracting More Variables From Names ##################################

# Get titles form passenger names gsub is a command to match regular expressions
# Each comma groups things together?
full$Title <- gsub('(.*, )|(\\..*)', '', full$Name)
table(full$Sex, full$Title)

# Put all titles that are rare (have low ocurrences) into a single variable
rare_title <- c('Dona', 'Lady', 'the Countess','Capt', 'Col', 'Don', 
                'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer')

# assume that some titles were entered in wrong and reassign them to what we think they should be
full$Title[full$Title == 'Mlle']        <- 'Miss' 
full$Title[full$Title == 'Ms']          <- 'Miss'
full$Title[full$Title == 'Mme']         <- 'Mrs' 
full$Title[full$Title %in% rare_title]  <- 'Rare Title'

# This made our titles list a lot smaller
table(full$Sex, full$Title)

full$Surname <- sapply(full$Name,  
                       function(x) strsplit(x, split = '[,.]')[[1]][1])

# Splitting the names gave us more variables that we can play with
str(full)

# She goes on to say that there is a way to infer ethnicity based on the surname, which I would love
# to know how she would go about that

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
full$Deck<-factor(sapply(full$Cabin, function(x) strsplit(x, NULL)[[1]][1]))

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

# I'm not sure  why we can assume that person 830 also embarked from Charbourg. 
# I found that they had the same ticket number, which might explain this
full[62, ]
full[830, ]


# The detective work she did yielded the following

# We will infer their values for embarkment based on present data that we can imagine may be relevant: 
# passenger class and fare. We see that they paid $ 80 and $ NA respectively and their classes are 1 
# and NA . So from where did they embark?


# Passenger 1044 had a missing Fare value
full[1044, ]

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
solution <- data.frame(PassengerID = test$PassengerId, Survived = prediction)

# Write the solution to file
write.csv(solution, file = 'rf_mod_Solution.csv', row.names = F)
