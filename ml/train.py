import os
import pickle
import sys
import time
from sklearn.model_selection import GridSearchCV
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import xgboost as xgb


# 定义黑样本和白样本文件夹路径
data = {
    "train_black_folders": [
        r"D:\Users\User\Desktop\EDR\powershell\mpsd-main\malicious_pure",

    ],
    "train_white_folders": [
        r"D:\Users\User\Desktop\EDR\powershell\mpsd-main\powershell_benign_dataset",
    ],
    "val_black_folders": [

    ],
    "val_white_folders": [

    ]
}

def load_data(paths, label=0):
    scripts_ = []
    for path in paths:
        print(path)
        for filename in os.listdir(path):
            with open(os.path.join(path, filename), "rb") as f:
                string = f.read().decode("utf-8", errors="ignore").lower()
                scripts_.append(string)
    labels_ = [label] * len(scripts_)
    return scripts_, labels_


def get_x_y(black_folders_, white_folders_):
    black_scripts, black_labels = load_data(black_folders_, label=1)
    white_scripts, white_labels = load_data(white_folders_, label=0)
    scripts_ = black_scripts + white_scripts
    labels_ = black_labels + white_labels
    return scripts_, labels_


def train_xgb(train_scripts_, train_labels_):
    # 使用TF-IDF向量化器对脚本进行特征提取
    max_features = 2000
    max_df = 0.95
    min_df = 0.05
    vectorizer_ = TfidfVectorizer(max_features=max_features, ngram_range=(1, 1), max_df=max_df, min_df=min_df,
                                  token_pattern=r"(?u)\b(?!\d+)\w\w+\b")

    train_features_ = vectorizer_.fit_transform(train_scripts_)
    tf_idf_name = f"max_df_{max_df}_min_df_{min_df}"
    with open(f"tfidf_model_file_command_{tf_idf_name}", "wb") as f:
        joblib.dump(vectorizer_, f)
    data = vectorizer_.get_feature_names()
    df = pd.DataFrame()
    df["words"] = data
    df.to_csv("vocab.csv")
    print("特征提取完毕，训练数据的维度：", train_features_.shape)
    print("开始训练模型...")

    # max_df浮点数或整数，默认值=1.0
    # 在构建词汇表时，忽略文档频率严格高于给定阈值的术语（特定于语料库的停用词）。
    # 如果在[0.0, 1.0]范围内浮动，该参数表示文档的比例，整数绝对计数。如果词汇表不是无，则忽略此参数。
    #
    # min_df浮点数或整数，默认值=1
    # 在构建词汇表时，忽略文档频率严格低于给定阈值的术语。该值在文献中也称为截止值。
    # 如果在[0.0, 1.0]范围内浮动，参数表示文档的比例，整数绝对计数。如果词汇表不是无，则忽略此参数。

    x_train, x_test, y_train, y_test = train_test_split(train_features_, train_labels_, test_size=0.1, random_state=42,
                                                        stratify=train_labels_)

    # # 使用XGB进行分类
    # xgb_model = xgb.XGBClassifier()
    # # 使用GridSearchCV函数进行网格搜索
    # # 定义参数网格
    # param_grid = {
    #     'max_depth': [5, 6, 7],
    #     'learning_rate': [0.03, 0.05, 0.1],
    #     'n_estimators': range(200, 500, 50)
    # }
    # grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=5, n_jobs=5, verbose=2)
    #
    # # 训练模型并搜索最佳参数组合
    # grid_search.fit(x_train, y_train)

    # 输出最佳参数组合和最佳得分
    # print("Best parameters: ", grid_search.best_params_)
    # clf_ = grid_search.best_estimator_

    max_depth = 7
    learning_rate = 0.05
    n_estimators = 400

    xgb_name = f"learning_rate_{learning_rate}_max_depth_{max_depth}_n_estimators_{n_estimators}"
    clf_ = xgb.XGBClassifier(max_depth=max_depth, learning_rate=learning_rate, n_estimators=n_estimators,
                             objective='binary:logistic', n_jobs=5)
    clf_.fit(x_train, y_train)
    y_pred_prob_ = clf_.predict_proba(x_test)
    y_pred_ = []
    for prob in y_pred_prob_:
        if prob[1] > 0.5:
            y_pred_.append(1)
        else:
            y_pred_.append(0)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_).ravel()
    tpr = tp / (tp + fn)
    print("训练检出率:", tpr)
    fpr = fp / (fp + tn)
    print("训练误报率：", fpr)
    with open(f'xgb_model_file_command_{xgb_name}.pkl', 'wb') as f:
        pickle.dump(clf_, f)

    return vectorizer_, clf_


if __name__ == '__main__':
    train = 1
    threshold = 0.90
    if train:
        start_time = time.time()
        print("加载训练数据...")
        train_scripts, train_labels = get_x_y(data["train_black_folders"], data["train_white_folders"])
        end_time = time.time()
        print("数据加载耗时：{:.2f}秒".format(end_time - start_time))
        print("数据加载完成...\n开始进行特征提取...")
        # 将数据集划分为训练集和测试集
        vectorizer, clf = train_xgb(train_scripts, train_labels)
    else:
        print("加载模型...")
        with open("tfidf_model_file_command", "rb") as f:
            vectorizer = joblib.load(f)
            # data = vectorizer.get_feature_names()
            # df = pd.DataFrame()
            # df["words"] = data
            # df.to_csv("vocab.csv")
            # exit(0)
        with open('xgb_model_file_command.pkl', 'rb') as f:
            clf = pickle.load(f)

        print("加载验证数据...")
        start = time.time()
        val_scripts, val_labels = get_x_y(data["val_black_folders"], data["val_white_folders"])
        end = time.time()
        print("验证数据数量：", len(val_labels), "  加载平均耗时:{:.2f}毫秒".format(1000 * (end - start) / len(val_labels)))
        val_features = vectorizer.transform(val_scripts)
        y_pred_prob = clf.predict_proba(val_features)

        y_pred = []
        for proba in y_pred_prob:
            if proba[1] > threshold:
                y_pred.append(1)
            else:
                y_pred.append(0)
        tn2, fp2, fn2, tp2 = confusion_matrix(val_labels, y_pred).ravel()
        tpr2 = tp2 / (tp2 + fn2)
        print("验证检出率:", tpr2)
        fpr2 = fp2 / (fp2 + tn2)
        print("验证误报率：", fpr2)

        # 测试1000次平均推理时间：
        times = []
        for i in range(1000):
            start = time.time()
            test_time_features = vectorizer.transform(val_scripts[i:i + 1])
            y_pred_prob = clf.predict_proba(test_time_features)
            end = time.time()
            times.append(end - start)

        avg_time = sum(times) / 1000
        print("特征提取加模型推理平均时长：{:.2f}毫秒".format(avg_time * 1000))
