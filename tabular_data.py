import pandas as pd
import numpy as np
from ast import literal_eval


def remove_rows_with_missing_ratings(df):

    rating_headers = ["Cleanliness_rating","Accuracy_rating","Communication_rating","Location_rating","Check-in_rating","Value_rating"]
    
    df = df.dropna(axis=0,how='any',subset=rating_headers)

    df = df.reset_index(drop=True)
    
    return df
    
        
def combine_description_strings(df):

    df = df.dropna(axis=0,how='any',subset=["Description"])

    df = df.reset_index(drop=True)

    new_descriptions = []
    
    for i in range(df.shape[0]):

        try:
            description_list = literal_eval(df['Description'][i])
        except SyntaxError:
            problem_id = df['ID'][i]
            print(f'\n\nThere is an issue with the description of property {problem_id}\n\n')
            df = df.drop(i)
            continue

        no_whitespace = [string for string in description_list if string.strip()]
        new_description = " ".join(no_whitespace[1:])


        new_descriptions.append(new_description)

    df = df.reset_index(drop=True)

    new_series = pd.Series(new_descriptions)
    df['Description'] = new_series

    return df


def set_default_feature_values(df):

    features = ['guests','beds','bathrooms','bedrooms']

    for feature in features:
        df[feature] = df[feature].fillna(1)

    return df
    
def clean_tabular_data(df):

    df = remove_rows_with_missing_ratings(df)    

    df = combine_description_strings(df)

    df = set_default_feature_values(df)

    return df


if __name__ == '__main__':
    df = pd.read_csv("tabular_data/fixed_listing.csv")

    df = clean_tabular_data(df)

    df.to_csv("tabular_data/clean_tabular_data.csv")

