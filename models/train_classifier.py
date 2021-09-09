import sys
import nltk
import numpy as np
nltk.download(['punkt', 'wordnet'])
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from sqlalchemy import create_engine
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

import re
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier

import pickle

def load_data(database_filepath):
    """Load the filepath and return data split in X (train variables), y (answers) and category_names (all possible output classes)"""
    name = 'sqlite:///' + database_filepath
    engine = create_engine(name)
    df = pd.read_sql_table('Disasters', con=engine)
    X = df['message']
    y = df[df.columns[4:]]
    category_names = y.columns
    return X, y, category_names


def tokenize(text):
    """tokenize and transform input text. Return clean text (urls removed)"""
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text)
    
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens


def build_model():
    """Return Grid Search model with pipeline"""

    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
        ])

    parameters = {'clf__estimator__max_depth': [10, 50, None],
              'clf__estimator__min_samples_leaf':[2, 5, 10]}

    cv = GridSearchCV(pipeline, parameters)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """Print model results
    INPUT
    model: model obtained during training
    X_test: dataset group with the variables that will be evaluated
    y_test: dataset group with the answers to the variables evaluated
    category_names: list of possible output categories 
    OUTPUT
    None
    """

    y_pred = model.predict(X_test)
    print('results obtained: ')
    print(y_pred)    



def save_model(model, model_filepath):
    """
    INPUT
    model: with the model generated during training
    model_filepath: file path to save the model 
    """
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

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