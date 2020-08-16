# Disaster Response Pipeline
Analyzing disaster data from Figure Eight (acquired by [Appen](https://appen.com/) in 2019) to build a machine learning pipeline that categorizes these events with the goal of distributing messages to the appropriate disaster relief agencies.

## Project Overview
The code is designed to initiate a web app for emergency operators to exploit during a disaster response. The web app classifies messages (from news outlets, websites, social media) and sorts them into categories for redistribution to appropriate disaster relief agencies.  

## Dataset
The disaster message dataset was retrieved from Figure Eight (prior to acquisition) and provided by Udacity. The 'messages.csv' file contains id, message text, and source (i.e. direct, social, news). The 'categories.csv' file includes id and categories associated with messages (i.e. medical help, earthquake, floods). A similar dataset can be found on the Appen webpage linked [here](https://appen.com/datasets/combined-disaster-response-data/).

## Installation
This repository was written in HTML and python and requires the following Python libraries: pandas, numpy, re, pickle, nltk, flask, json, plotly, sklearn, sqlalchemy, sys.

## File Description
* **data**: This folder contains message and category csv datasets
* **app**: This folder includes the run.py to initiate the web app along with the html templates for the web app design
* **process_data.py**: This code inputs csv files, cleans datasets, and creates a SQL database
* **train_classifier.py**: This code trains the ML model with the SQL database
* **ETL Pipeline Preparation.ipynb**: workspace used to build process_data.py
* **ML Pipeline Preparation.ipynb**: workspace used to build train_classifier.py

## Instructions
1. Run the following commands in the project's root directory to set up the database and model:

  * To run ETL pipeline that cleans data and stores in database

        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
  * To run ML pipeline that trains classifier and saves

        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run the web app:

    `python run.py`

3. Go to http://0.0.0.0:3001/ or go to http://localhost:3001/

## Web App Screenshots
Below are visuals of what to expect when the web app is successfully run. 

<img width="1215" alt="Screen Shot 2020-08-15 at 9 15 29 PM" src="https://user-images.githubusercontent.com/66285135/90326070-7ebacf00-df49-11ea-8fd3-f630121ca741.png">
<img width="1162" alt="Screen Shot 2020-08-15 at 9 15 47 PM" src="https://user-images.githubusercontent.com/66285135/90326081-985c1680-df49-11ea-87a7-3565b798d06f.png">
<img width="1151" alt="Screen Shot 2020-08-15 at 9 13 40 PM" src="https://user-images.githubusercontent.com/66285135/90326096-d22d1d00-df49-11ea-97b3-4a6ea4084612.png">
<img width="1077" alt="Screen Shot 2020-08-15 at 10 51 00 PM" src="https://user-images.githubusercontent.com/66285135/90326097-d5c0a400-df49-11ea-9b16-7ad6bf235d94.png">

## Licensing, Authors, Acknowledgements
This work is licensed under a [Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License](https://creativecommons.org/licenses/by-nc-nd/4.0/). Please refer to Udacity [Terms of Service](https://www.udacity.com/legal) for additional information.
