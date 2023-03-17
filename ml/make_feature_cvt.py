import joblib
import pickle
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

train_data_ = pd.read_csv("raw_data_fre>200.csv")
length = len(train_data_)
train_data_ = train_data_[:int(0.8*length)]

vectorizer = CountVectorizer(max_features=1000)

train_cvt_features = vectorizer.fit_transform(train_data_.new_word.values.astype('U'))
print(train_cvt_features.shape)

with open("cvt_model", "wb") as fp:
    joblib.dump(vectorizer, fp)

with open("train_cvt_features", "wb") as fp:
    pickle.dump(train_cvt_features, fp)

with open("train_labels", "wb") as fp:
    pickle.dump(train_data_.label, fp)