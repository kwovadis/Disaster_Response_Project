import sys
import pickle
import re 
import pandas as pd 
import numpy as np 
from sqlalchemy import create_engine

import nltk
nltk.download(['punkt', 'wordnet','omw-1.4'])
from nltk.tokenize import word_tokenize 
from nltk.stem import WordNetLemmatizer

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer 
from sklearn.ensemble import RandomForestClassifier 

from sklearn.metrics import classification_report
from sklearn.metrics import precision_score, recall_score, f1_score


def load_data(database_filepath):
    '''
    Load data from an SQLite database and output training & target data (X, Y), 
    as well as labels (Y.columns).
    INPUT
        database_filepath - SQLite database (ex. data/DisasterResponse.db)
    OUTPUT
        X  - Training data.
        Y - Target data.
        Y.columns - Target data labels.
    '''

    # load data from database
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql("SELECT * FROM DisasterResponseTable", engine)
    
    X = df['message']
    Y = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    return X, Y, Y.columns


def tokenize(text):
    '''
    Clean and tokenize/lemmatize the text data tbefore classifying any posts.
    INPUT
        text - text data that will be tokenized in the ML pipeline
    OUTPUT
        clean_tokens  - clean tokenized text data 
    '''

    # Write a tokenization function to process your text data
    text = re.sub(r'[^\w\s]','',text)
    tokens = word_tokenize(text) 
    lemmatizer = WordNetLemmatizer() 
    clean_tokens = [] 
    for tok in tokens: 
        clean_tok = lemmatizer.lemmatize(tok).lower().strip() 
        clean_tokens.append(clean_tok) 
    return clean_tokens


def build_model():
    '''
    ML pipeline used to predict classifications for 36 categories.
    OUTPUT
        model  - ML pipeline 
    '''
    # # Build a machine learning pipeline
    # model = Pipeline([
    #     ('vect', CountVectorizer(tokenizer=tokenize)),
    #     ('tfidf', TfidfTransformer()),
    #     ('clf', RandomForestClassifier())
    # ])
    # return model

    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ]))

        ])),

        ('clf', RandomForestClassifier())
    ])

    parameters = {
        # 'features__text_pipeline__vect__ngram_range': ((1, 1), (1, 2)),
        'clf__min_samples_split': [2, 4, 6]
    }

    model = GridSearchCV(pipeline, param_grid=parameters)

    return model


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Predict & evaluate quality of classifications using f1 score, precision and 
    recall for each output category.
    INPUT
        model - text data that will be tokenized in the ML pipeline
        X_test - training data
        Y_test - target data
    OUTPUT
        Printed statements for f1 score, precision and recall for each output category 
    '''
    # predict on test data
    Y_pred = model.predict(X_test)
    df_Y_pred = pd.DataFrame(Y_pred, columns=Y_test.columns)
    np_Y_test = Y_test.to_numpy()#.as_matrix()
    
    # Report the f1 score, precision and recall for each output category of the dataset by iterating through the columns and calling sklearn's classification_report on each.
    print('Classification report for all categories:')
    for i in range(len(Y_test.columns)):
        col = Y_test.columns[i]
        print(f'"{col}" category: \n {classification_report(np_Y_test[:,0], Y_pred[:,0])}')


def save_model(model, model_filepath):
    '''
    Save model to a pickle file.
    INPUT
        model - text data that will be tokenized in the ML pipeline
        model_filepath - path to pickle file (ex. models/classifier.pkl)
    OUTPUT
        pickleed model file (ex. models/classifier.pkl) 
    '''
    pickle.dump(model, open(model_filepath, "wb"))


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