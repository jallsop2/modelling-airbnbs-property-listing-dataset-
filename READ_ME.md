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



Aside from the decision tree algorithm, the other models were all fairly similar, with the gradient boosted being best by a small amount. Interestingly the gradient boosting model has quite a high training accuracy, meaning it is still very biased, however I was unable to make the model any less biased without making it perform worse on the test set. A likely explanation is that ~0.4 is the best the data can be used to predict the cateories, but the gradient boosting model has enough capacity to do that while also overfitting to the training set. This could naturally be solved by having more data, and it seems likely that with more data the gradient boosting algorithm would be able to perform better than the logistic regression algorithm. However the gradient boosting algorithm also runs significantly slower than the logistic regression so it's possible that would make it a more reasonable choice.


# Milestone 6

The aim of this milestone was to use pytorch to create a configurable neural network, and to use this to predict the nightly price of the Airbnb listings.

First I had to create a custom pytorch dataset class, by inheriting `torch.utils.data.Dataset`, which took the airbnb data and turned it into a pytorch tensor, and contained a `__getitem__` method to get the features and label at an index. I also allowed the class to take a list of indices, and only use the data at those indices. This allowed me to split a list of indices using the sklearn train_test_split function and get two custom datasets for the training and test data.

Next I created a class for the neural network by inheriting `torch.nn.module`. This class took in the width of the hidden layers and the depth of the model as parameters, so that they could be externally configured, and then defined the layers using `torch.nn.Linear` and `torch.nn.ReLU`. It also had a forward method to define the forward pass of the model, in this case just running the features through the layers.

I created a dataloader using `torch.utils.data.dataloader.Dataloader` and this allowed me to create a training loop to run minibatch gradient decent. The optimiser and hyperparameters were left as parameters to the training loop, so that they could be  easily changed later.

Now it came to testing and tuning the model, and I got the same issue as in the previous milestones: the effectiveness of the model depended significantly on the split of the training and test data. So, like in the previous milestones, I averaged out the performance of the model over many different splits of the data, and manually changed the hyperparameters to find the best ones. The hyperparameters I was changing were the complexity of the neural network, as well as the type of optimiser and learning rate and weight decay of the optimiser.

I tried two sizes of neural network, a simple one with just one hidden layer of size 8, and a more complex one with three hidden layers of size 32. I tested two optimisers, `torch.optim.SGD` and `torch.optim.Adam`, and tuned the learning rates and weight decay for both of them, eventually giving these results:


## Simple Network

### SGD optimiser

"optimiser" = "torch.optim.SGD" \
"lr" = 0.1 \
"weight_decay" = 0.05 \
"epochs" = 50 \
"hidden_layer_width" = 8 \
"depth" = 2 


Training RMSE = 0.721 \
Test RMSE = 0.787 


### Adam optimiser

"optimiser" = "torch.optim.Adam" \
"lr" = 0.01 \
"weight_decay" = 0.05 \
"epochs" = 50 \
"hidden_layer_width" = 8 \
"depth" = 2 

Training RMSE = 0.708 \
Test RMSE = 0.782


## Complex Network

### SGD optimiser

"optimiser" = "torch.optim.SGD" \
"lr" = 0.05 \
"weight_decay" = 0.05 \
"epochs" = 50 \
"hidden_layer_width" = 32 \
"depth" = 4 

Training RMSE = 0.727 \
Test RMSE = 0.791

### Adam optimiser

"optimiser" = "torch.optim.Adam" \
"lr" = 0.005 \
"weight_decay" = 0.05 \
"epochs" = 50 \
"hidden_layer_width" = 32 \
"depth" = 4 

Training data = 0.706 \
Test data = 0.788



The results show that the more complicated model didn't actually perform better than the simple one, in fact they were almost exactly the same. Therefore it appears that the simple model has enough complexity to fit this problem, and adding more hidden rows simply made it take longer to train. I considered trying an even more complex model, but it seems likely it would perform about the same at best, so wasn't worth it. The results also show that the Adam optimiser works slightly better than the SGD optimiser, which makes sense since it is a more sophisticated algorithm.

The neural networks perform comparably to the regression models from milestone 4, on par with the best linear and gradient boosting models. It makes sense that all of the models perform about the same, since the data is limited in how effectively it can predict the nightly price. Therefore all of the models are able to reach that limit, with only slight differences in performance.

# Milestone 7

The aim of this milestone is to take the models and methods from the previous milestones and apply them to a new use case, in this case predicting the number of bedrooms instead of the nightly price. I used the hyperparameters which were deemed best after tuning in previous milestones, assuming that they would still be the best for this use case, and tested an instance of linear regression, decision tree, random forest, gradient boosted and neural network models. I created a new python file to take these models and apply them to predicting the number of bedrooms, again taking the average perfromance over many different tests. 

## Linear Regression

Training RMSE: 0.586 \
Test RMSE: 0.608


## Rgeression Tree

Training RMSE: 0.630 \
Test RMSE:0.660


## Random Forest

Training RMSE: 0.602 \
Test RMSE: 0.644


## Gradient Boosting

Training RMSE: 0.565 \
Test RMSE: 0.645


## Neural Network (Simple/Adam optimiser)

Training RMSE: 0.538 \
Test RMSE: 0.587


## Neural Network (Complex/Adam optimiser)

Training RMSE: 0.548 \
Test RMSE: 0.589


These results have more differences between the models than the the previous results, showing that the neural networks is the best at predicting the number of bedrooms, followed by the linear regression model and then all of the other models. The results are close enough however to make me believe that the models are correctly tuned for this dataset, and that these models may be used for many more different possibly use cases. Again the simple and complex neural networks perform about the same, from which I can make the same assumtions as before that the added complexity is not needed to predict the data as well as possible given the data.