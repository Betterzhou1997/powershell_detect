import joblib
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

train_data_ = pd.read_csv("raw_data_fre>1000.csv")
length = len(train_data_)
train_data_ = train_data_[:int(0.8*length)]

vectorizer = TfidfVectorizer(min_df=1, max_df=1, max_features=1000)

train_tfidf_features = vectorizer.fit_transform(train_data_.new_word.values.astype('U'))

with open("tfidf_model", "wb") as fp:
    joblib.dump(vectorizer, fp)

with open("train_tfidf_features", "wb") as fp:
    pickle.dump(train_tfidf_features, fp)

with open("train_labels", "wb") as fp:
    pickle.dump(train_data_.label, fp)