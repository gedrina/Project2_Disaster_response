# import libraries
import sys
import re
import pandas as pd
import numpy as np
import sys
import pickle
from sqlalchemy import create_engine

import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report


def load_data(database_filepath):
    """Load data from database

    Args:
        database_filepath (str): the path of the database
    Returns:
        X (dataframe): Feature data 
        y (dataframe): Labels data
    """
    #load data
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql(database_filepath[5:-3], engine)
    
    # Data modelling
    X = df['message'] #contains messages
    y = df.drop(['message','genre','id','original'], axis=1) #categories of disaster
    
    return X,y


def tokenize(text):
    """Normalize text, Punctuation removal, tokenize and lemmatization 

    Args:
        text (str): Message
    Returns:
        clean_tokens (list): list of tokens
    """
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
        
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
        
    return clean_tokens


def build_model():
    
    """Builds the ML model 
    Returns:
        cv (GridSearchCV): Machine learning model
    """
    # build pipeline
    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('lr_multi', MultiOutputClassifier(RandomForestClassifier()))
])
    parameters = {
    'tfidf__use_idf':(True, False),
    'lr_multi__estimator__n_estimators':[5, 50]
}

    cv = GridSearchCV(pipeline,param_grid=parameters)
    
    return cv


def evaluate_model(model, X_test, y_test):
    """Evalute the model and print the accuracy of the model and classification_report for every category

    Args:
        model (): _description_
        X_test (string): test messages
        Y_test (string): test labels 
    """
    y_predict = model.predict(X_test)
    
    # Print classification report on test data
    i = 0
    for col in y_test:
        print('Feature {}: {}'.format(i + 1, col))
        print(classification_report(y_test[col], y_predict[:, i]))
        i = i + 1
    accuracy = (y_predict == y_test.values).mean()
    print('Accuaracy of the model: {}'.format(accuracy))
    

def save_model(model, model_filepath):
    """Save the model to picke file for future use

    Args:
        model (GridSearchCV): Trained Machine Lerning model
        model_filepath (str): Path where the model is saved
    """
    pickle.dump(model, open(model_filepath, 'wb'))
    


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        # database_filepath, model_filepath = '../data/disaster_response.db', 'classifier.pkl'
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, y_test)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: '\
              'python models/train_classifier.py data/disaster_response.db models/classifier.pkl')


if __name__ == '__main__':
    main()