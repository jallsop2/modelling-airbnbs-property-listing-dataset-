import pandas as pd
import numpy as np

df = pd.read_csv("tabular_data/listing.csv")

def remove_rows_with_missing_ratings(df):

    rating_headers = ["Cleanliness_rating","Accuracy_rating","Communication_rating","Location_rating","Check-in_rating","Value_rating"]
    
    df = df.dropna(axis=0,how='any',subset=rating_headers)
    
    return df
    
    
    """ size = df.shape[0]

    for i in range(size):
        rating_headers = ["Cleanliness_rating","Accuracy_rating","Communication_rating","Location_rating","Check-in_rating","Value_rating"]
        rating_missing = False
        for j in range(6):
            if np.isnan(df[rating_headers[j]][i]) == True:
                rating_missing = True
        
        if rating_missing == True:
            df.drop() """
        
df = remove_rows_with_missing_ratings(df)

print(df)