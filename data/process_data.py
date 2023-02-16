import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """Load the datasets and merge them

    Args:
        messages_filepath (str): filepath for csv file containing messages 
        categories_filepath (str): filepath for csv file containing categories
    Returns:
        df: merged dataframe containing the content of messages and categories datasets
    """
    #load data from two csv files
    messages = pd.read_csv(r'data/disaster_messages.csv')
    categories = pd.read_csv(r'data/disaster_categories.csv')
    # merge datasets
    df = pd.merge(messages, categories, on='id')
    
    return df

def clean_data(df):
    """Cleaning the dataframe by:
       Spliting categories into separate category columns
       Converting category values to just numbers 0 or 1
       Replacing categories column in df with new category columns

    Args:
        df (dataframe): Containing the merged dataset of messages and categories
        
    Returns:
        df (dataframe): The cleaned version of the input df 
    """
    categories = df.categories.str.split(";",expand=True)
    row = categories.iloc[0]
    category_colnames = row.str[:-2]
    categories.columns = category_colnames
    categories.columns = category_colnames
    categories['related'] = categories['related'].str[-1:]
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1:]

        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    
    # drop the original categories column from `df`
    df.drop('categories', axis='columns', inplace=True)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories], axis = 1)
    
    # drop duplicates
    df.drop_duplicates(inplace=True)
    
    #here are three classes 0,1,2 for related category
    #value 2 could be error,drop the error
    df.drop(df[df['related'] == 2].index, inplace = True)

    return df


def save_data(df, database_filename):
    """Save the cleaned dataframe in database

    Args:
        df (dataframe): cleaned dataframe of messages and categories
        database_filename (str): name of the database file
    """
    # engine = create_engine('sqlite:///' + database_filename)
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('disaster_response', engine, index=False, if_exists='replace')    


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