import sys
import re
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

import matplotlib.pyplot as plt


def load_data(messages_filepath, categories_filepath):
    '''
    Load messages & categories data from csv files into a pandas DataFrame.
    INPUT
        messages_filepath - csv file with messages describing disaster
        categories_filepath - csv file with categories defining each message
    OUTPUT
        df  - dataframe with merged messages & categories
    '''

    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, how='inner', on=['id'])
    return df


def clean_data(df):
    '''
    Clean messages & categories data in pandas DataFrame.
    INPUT
        df - dataframe with merged messages & categories
    OUTPUT
        df  - cleaned dataframe with categories converted to bool
    '''

    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand=True)
    # select the first row of the categories dataframe
    row = categories.iloc[0]
    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = row.apply(lambda x: x[:-2]).values
    categories.columns = category_colnames
    # Convert category values to just numbers 0 or 1.
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1] #apply(lambda x: x[-1]).values
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column], errors='coerce')
        # convert column from numeric to binary
        categories.loc[categories[column] > 1, column] = 1
    # categories.plot.hist(subplots=True, layout=(5,8), figsize = (20, 32), bbox_inches='tight') 
    # plt.savefig('plotted_categorical_data.pdf') 

    # drop the original categories column from `df`
    df.drop('categories', axis=1, inplace=True)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1) #, join='outer', ignore_index=True)
    # drop duplicates
    df.drop_duplicates(inplace=True)
    return df


def save_data(df, database_filename):
    '''
    Save pandas DataFrame to an SQLite database.
    INPUT
        df - cleaned dataframe with merged messages & bool categories
    OUTPUT
        database_filename  - SQLite database (ex. data/DisasterResponse.db)
    '''
    # Save the clean dataset into an sqlite database.
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('DisasterResponseTable', engine, index=False, if_exists='replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
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