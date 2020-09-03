import sys
import re
import pandas as pd
import numpy as np
import pickle
from sqlalchemy import create_engine
from sklearn.metrics import confusion_matrix
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
  
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import FunctionTransformer
from sklearn.linear_model import LogisticRegression
import pickle


def load_data(database_filepath):
    engine = create_engine('sqlite:///./{}'.format(database_filepath))
    df = pd.read_sql_table('disaster_data', con = engine)
    X = df.iloc[:, 1]
    Y = df.iloc[:, 4:]
    category_name = Y.columns
    return X, Y, category_name 

def tokenize(text, language = 'english', lem = True):
   ## normalized
    text_norm = re.sub(r'[^a-zA-Z0-9]'," ", text.lower())
    ## tokenized
    word_text = word_tokenize(text_norm)
    ## skip stop word
    stop = stopwords.words(language)
    words = [w for w in word_text if w not in stop]
    if lem:
        return [WordNetLemmatizer().lemmatize(w) for w in words]
    else:
        return [PorterStemmer().stem(w) for w in words]


def build_model():
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    parameters = {'clf__estimator__n_estimators': [10],'clf__estimator__min_samples_split':[2]}

    cv = GridSearchCV(estimator = pipeline, param_grid=parameters)

    return cv

def display_results(Y_test, y_pred, i, Y):
    labels = np.unique(y_pred)
    confusion_mat = confusion_matrix(Y_test, y_pred)
    accuracy = sum(y_pred == Y_test)/len(y_pred)
    
    Labels =  Y.columns[i]
    Confusion_Matrix =  confusion_mat
    accuracy = accuracy
    return labels, Confusion_Matrix, accuracy

def evaluate_model(model, X_test, Y_test, category_names, Y):
    y_pred = model.predict(X_test)
    y_pred_df = pd.DataFrame(data = y_pred, columns = category_names, index = Y_test.index)
    accuracy_lst = []
    for i in range(y_pred_df.shape[1]):
        labels, Confusion_Matrix, accuracy = display_results(Y_test.iloc[:,i], y_pred_df.iloc[:,i], i, Y)
        accuracy_lst.append(accuracy)
    print('Mean accuracy of Model : {}'.format(np.mean(accuracy_lst)))
   


def save_model(model, model_filepath):
    filename = model_filepath
    pickle.dump(model, open(filename, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print(database_filepath)
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names, Y)

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