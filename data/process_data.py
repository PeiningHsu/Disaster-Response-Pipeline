import sys
from sqlalchemy import create_engine
import pandas as pd
import numpy as np


def load_data(messages_filepath, categories_filepath):
    #import csv files
    messages = pd.read_csv('./{}'.format(messages_filepath))
    categories = pd.read_csv('./{}'.format(categories_filepath))
    #concate two dataframes into one df
    df = pd.concat([messages,categories], axis = 1)
    return messages, categories, df

def related_clean(x):
    # if related is 2, amend it to 1
    if x == 2:
        return 1
    else:
        return x

def clean_data(df, categories):
    # create a dataframe of the 36 individual category columns
    categories = categories['categories'].str.split(';',n=36, expand=True)
    # select the first row of the categories dataframe
    row = categories.iloc[1,:]
    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = row.apply(lambda x: x[:-2]).tolist()
    # rename the columns of `categories`
    categories.columns = category_colnames
    for column in categories:
    # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x[-1])
    # convert column from string to numeric
        categories[column] = categories[column].astype('int')
    # Amend 2 in related to 1
    categories['related'] = categories['related'].apply(lambda x: related_clean(x))
    # drop the original categories column from `df`
    df.drop('categories', axis = 1, inplace = True)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis = 1)
    # drop duplicates
    df.drop_duplicates(inplace = True)
    
    return df


def save_data(df, database_filepath):
    # Save the clean dataset into an sqlite database.
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df.to_sql('disaster_data' , engine, index=False, if_exists='replace') 


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        messages, categories, df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df, categories)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()