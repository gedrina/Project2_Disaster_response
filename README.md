# Disaster Response Pipeline Project

# Introduction

Figure 8 is a company that aids in the transformation of data by offering both human annotators and machine learning capabilities to annotate data at various scales. One area that significantly benefits from data and machine learning is disaster response. The focus of this project is to create a model capable of classifying messages received in real time during a disaster event into 36 pre-defined categories.The project is divided into three sections:

1. Data Processing - Data cleaning ETL pipeline 
    *Loads the messages and categories datasets
    *Merges the two datasets
    *Cleans the data
    *Stores it in a SQLite database
2.Machine Learning Pipeline 
    *Loads data from the SQLite database
    *Splits the dataset into training and test sets
    *Create a machine learning pipeline to train a model which is able to classify text messages in 36 categories
    *Exports the final model as a pickle file
3.Web Application using Flask - display the results of the model in real-time 

#Files Descriptions

- app
| - template
| |- master.html  # main page of web app
| |- go.html  # classification result page of web app
|- run.py  # Flask file that runs app

- data
|- disaster_categories.csv  # categories dataset
|- disaster_messages.csv  # messages dataset
|- process_data.py # ETL process
|- disaster_response.db   # database 

- models
|- train_classifier.py #classification code
|- classifier.pkl  # saved model 

- README.md

# Instructions:
1. Run the requirements file to install the needed libraries
    `pip install -r requirements.txt`

    Ensure that sqlalchemy is of a version < 2.0

2. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/disaster_response.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/disaster_response.db models/classifier.pkl`

    Because of cross-validation in hyperparameter optimization, the model takes a while to run (different number of
    trees, using or not TF-IDF).

    The trained classifier reaches an accuracy of around 94.4%.

3. Go to `app` directory: `cd app`

4. Run your web app: `python run.py`

5. Click the `PREVIEW` button to open the homepage

# Acknowledgements

The development of this application was done in fulfillment of the requirements for the Udacity Data Scientist Nanodegree