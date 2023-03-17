import pandas as pd
import joblib
from sklearn import metrics


with open("tfidf_model", "rb") as fp:
    vectorizer = joblib.load(fp)
with open("train_model", "rb") as fp:
    model = joblib.load(fp)

test_data_ = pd.read_csv("raw_data3.csv")
test_data_ = test_data_[int(0.8*len(test_data_)):]

test_tfidf_features = vectorizer.transform(test_data_.words.values.astype('U'))
y_prob = model.predict_proba(test_tfidf_features)
print(y_prob)
y_pred = model.predict(test_tfidf_features)

y_real = test_data_.label.values

f1_score_ = metrics.f1_score(y_real, y_pred)
precision = metrics.precision_score(y_real, y_pred)
recall = metrics.recall_score(y_real, y_pred)
acc = metrics.accuracy_score(y_real, y_pred)
print(acc, precision, recall, f1_score_)