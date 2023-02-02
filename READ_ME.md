# Modelling Airbnb's property listing dataset

The aim of the project is to build a framework to systematically train tune, and evaluate models on several tasks that are tackled by the airbnb team.

# Milestone 3
The milestone focusses on preparing and cleaning both the tabular and the image data from the given dataset.

Firstly on brief inspection one of the rows has an incorrect description, causing the later values to all be offset. I wrote a script to quickly solve this by saving all of the correct data in that row to a pandas series and then replacing the incorrect row with it.

Next for the tabular data I used the pandas python package to transfer the data from the csv file into a dataframe which can then easily be manipulated. Using the .dropna pandas method I could easily remove the rows which had missing valus for the ratings columns or the decription, similarly using the .fillna method I added default values for the columns which required them. Cleaning the description data was more complicated, since the they were saved as a string in the form of a python list.  Using the ast.literal_eval function I accessed the information in the list and joined them together in one string, also stripping out any empty strings in the list. These were then saved in a pandas series and used to replace the previous description data.

Finally to work with the image data I used the image library in the PIL package. This allowed me to save the image data with image.open() and resize them with .resize. I resized all of the images to the same height and removed any that were not in the RGB format, then saved them into a new folder.


# Milestone 4

Now that the data has been cleaned, this milestone is focused on implementing a regession model to predict the price per night of a listing.

I started with a simple sgd regression model and implemented it with the default parameters using the sklearn class SGDRegressor(). I also used sklearn to split the data into training and testing sets, with a 70/30 split. The metrics I am looking at is the RMSE on the train and test sets. Running the model multiple times made it very clear that the performance of the model on the test set would vary significantly based on how the data was split, which made evaluating just one model not very useful. Instead I fit and evaluated the model on multiple different splits and then averaged the metrics for them all, which in this case gave:
Train RMSE = 0.761
Test RMSE = 0.791

Next I tuned the hyperparameters to try and optimise the sgd regression model. Because the model accuracy varied a lot, I needed to test the different option multple times (normally 100 or 1000), in order to compare them confidently, but this wouldn't work with many different hyperparamter options and combinations, because that would take too long. So I decided to use the sklearn class GridSearchCV to get an idea for the best and most important options and then manually tuned them by averaging over 100s of possible splits. Doing this for the SGD regressor gave hyperparameters of:

"alpha" : 0.1 \
"max_iter" : 10000 \
"learning_rate" : "adaptive" \
"eta0" : 0.01 \
"n_iter_no_change" : 100  

and average metrics of:

Train RMSE = 0.764 \
Test RMSE = 0.787

Then I tried other regression models and tuned them using the same method, altering how many splits I averaged them over based on how long the model took to fit. I used the DecisionTreeRegressor, the RandomForestRegressor and the GradientBoostingRegressor from sklearn and the results are below.

## Decision Tree:

"max_depth" : None \
"min_samples_split" : 75 \
"min_samples_leaf" : 25 

Train RMSE = 0.761 \
Test RMSE = 0.804


## Random Forest:

"n_estimators" : 1000 \
"max_depth" : 10 \
"min_samples_leaf" : 15 \
"min_samples_split" : 30 

Train RMSE = 0.729 \
Test RMSE = 0.783

## Gradient Boosting:

"n_estimators" : 100 \
"subsample" : 0.5 \
"min_samples_leaf" : 15 \
"min_samples_split" : 60 \
"max_depth" : 2

Train RMSE = 0.691 \
Test RMSE = 0.777


These results show that it is difficult to do much better than the origional model, likely because that is the maximum the data can be used to predict the price. They do show that some models work better than other for this case, with the gradient boosted regressor working the best and the regression tree working the worst. However the gradient boosted and random forest models worked significantly slower than the others, so if I wanted to replicate this with a mch larger data set, then the linear regression model might be a more efficient choice.

# Milestone 5

The next milestone is using similar techniques to predict the category of the properties

I used classification versions of the same algorithms I used in the previous milestones and tuned them using the same methods. The metrics I am evaluating by are the accuracy score on the training and test data sets, again the results varied on the random split, of the data so it was necessary to average over 100 different tests.


## Basic logistic regression:

Training accuracy = 0.436 \
Test accuracy = 0.387


## Tuned logistic regression

"C" : 0.25

Training accuracy = 0.436 \
Test accuracy = 0.389


## Decision tree classifier:

"max_depth" : None \
"min_samples_split" : 150 \
"min_samples_leaf" : 25 

Training accuracy = 0.410 \
Test accuracy = 0.352


## Random forest:

"n_estimators" : 100 \
"max_depth" : None \
"min_samples_leaf" : 3 \
"min_samples_split" : 20

Training accuracy = 0.672 \
Test accuracy = 0.381


## Gradient boosted:

"subsample" : 0.75 \
"learning_rate" : 0.01 \
"n_estimators" : 1000 \
"min_samples_leaf" : 5 \
"min_samples_split" : 4 \
"max_depth" : 2 \
"n_iter_no_change" : None

Training accuracy = 0.700 \
Test accuracy = 0.394



Aside from the decision tree algorithm, the other models were all fairly similar, with the gradient boosted being best by a small amount. Interestingly the gradient boosting model has quite a high training accuracy, meaning it is still very biased, however I was unable to make the model ay less biased without making it perform worse on the test set. A likely explanation is that ~0.4 is the best the data can be used to predict the cateories, but the gradient boosting model has enough complexity to do that while also overfitting to the training set. This could naturally be solved by having more data, and it seems likely that with more data the gradient boosting algorithm would be able to perform better than the logistic regression algorithm. However the gradient boosting algorithm also runs significantly slower than the logistic regression so a it's possible that would make it a more reasonable choice.
