# Set up workspace
import numpy as np
import pandas as pd
import re
import sys
import pickle
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', 'stopwords'])
import subprocess
def install(package):
    subprocess.call([sys.executable, "-m", "pip", "install", package])
install('termcolor')
from termcolor import colored, cprint
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from nltk import pos_tag
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sqlalchemy import create_engine
from sklearn import preprocessing
from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import TruncatedSVD


def load_data(database_filepath):
    """
    Function: load data from database and return X and y.
    Args:
      database_filepath(str): database file name included path
    Return:
      X(pd.DataFrame): messages for X
      y(pd.DataFrame): labels part in messages for y
      category_names(str):category names
    """
    # Load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('Messages', engine)
    X = df.message.values
    y = df.iloc[:,4:]
    
    # List the columns
    category_names = list(np.array(y.columns))
    return X, y, category_names

  
def tokenize(text):
    """
    Function: tokenize the text
    Args: 
      source string
    Return:
      clean_tokens(str list): clean string list
    """ 
    # Normalize text
    text = re.sub(r'[^a-zA-Z0-9]',' ',text.lower())

    # Token messages
    words = word_tokenize(text)
    tokens = [w for w in words if w not in stopwords.words("english")]

    # Lemmatizer and clean
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """
    Function: build model that consist of pipeline
    Args:
      N/A
    Return
      cv(model): Grid Search model
    """
    # Build pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))])
    # Set parameters
    parameters = {'tfidf__use_idf': (True, False),
              'vect__ngram_range': ((1, 1), (1, 2)),
              'clf__estimator__n_estimators': [10,20],}
    # Create grid search object
    cv = GridSearchCV(estimator=pipeline, param_grid=parameters, n_jobs=-1)
    
    return cv


def evaluate_model(model, X_test, y_test, category_names):
    """
    inputs
        model
        X_test
        y_test
        category_names
    output:
        scores
    """
    y_pred = model.predict(X_test)

    for i, col in enumerate(category_names):
        precision, recall, fscore, support = precision_recall_fscore_support(y_test[col],
                                                                    y_pred[:, i],
                                                                    average='weighted')

        print('\nReport for the column ({}):\n'.format(colored(col, 'red', attrs=['bold', 'underline'])))

        if precision >= 0.75:
            print('Precision: {}'.format(colored(round(precision, 2), 'green')))
        else:
            print('Precision: {}'.format(colored(round(precision, 2), 'yellow')))

        if recall >= 0.75:
            print('Recall: {}'.format(colored(round(recall, 2), 'green')))
        else:
            print('Recall: {}'.format(colored(round(recall, 2), 'yellow')))

        if fscore >= 0.75:
            print('F-score: {}'.format(colored(round(fscore, 2), 'green')))
        else:
            print('F-score: {}'.format(colored(round(fscore, 2), 'yellow')))

    
def save_model(model, model_filepath):
    """
    Function: save model as pickle file.
    Args:
      model:target model
    Return:
      N/A
    """
    pickle.dump(model, open(model_filepath, 'wb'))
    

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y, category_names = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')

if __name__ == '__main__':
    main()