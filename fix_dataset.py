import pandas as pd
import numpy as np


df = pd.read_csv("tabular_data/listing.csv")

new_data = []

for i in range(3):
    new_data.append(df.iloc[586,i])

for i in range(4,20):
    new_data.append(df.iloc[586,i])

new_data.append(np.nan)

new_row = pd.Series(new_data)

df.iloc[586] = new_row

df = df.drop('Unnamed: 19', axis=1)

df = df.set_index('ID')

df.to_csv("fixed_listing.csv")


""" from PIL import Image
import os

dirs =  os.listdir('images')

for dir in dirs:

    if os.path.isdir(f'images/{dir}/{dir}'):
        os.rmdir(f'images/{dir}/{dir}') """
    


