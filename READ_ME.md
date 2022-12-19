# Modelling Airbnb's property listing dataset

The aim of the project is to build a framework to systematically train tune, and evaluate models on several tasks that are tackled by the airbnb team.

# Milestone 3
The milestone focusses on preparing and cleaning both the tabular and the image data from the given dataset.

Firstly on brief inspection one of the rows has an incorrect description, causing the later values to all be offset. I wrote a script to quickly solve this by saving all of the correct data in that row to a pandas series and then replacing the incorrect row with it.

Next for the tabular data I used the pandas python package to transfer the data from the csv file into a dataframe which can then easily be manipulated. Using the .dropna pandas method I could easily remove the rows which had missing valus for the ratings columns or the decription, similarly using the .fillna method I added default values for the columns which required them. Cleaning the description data was more complicated, since the they were saved as a string in the form of a python list.  Using the ast.literal_eval function I accessed the information in the list and joined them together in one string, also stripping out any empty strings in the list. These were then saved in a pandas series and used to replace the previous description data.

Finally to work with the image data I used the image library in the PIL package. This allowed me to save the image data with image.open() and resize them with .resize. I resized all of the images to the same height and removed any that were not in the RGB format, then saved them into a new folder.
