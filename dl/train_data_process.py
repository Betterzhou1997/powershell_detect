import os
import pandas as pd
from sklearn.utils import shuffle
from pretrain_data_process import data_preprocess


train_data_path = './data/train_data.csv'
if __name__ == '__main__':
    df = pd.DataFrame()
    white_data_dir = '/home/jovyan/datasets/powershell/clean-powershell-white'
    black_data_dir = '/home/jovyan/datasets/powershell/clean-powershell-black'

    list_ = []
    data_preprocess(os.listdir(white_data_dir), list_, [], white_data_dir)
    data_preprocess(os.listdir(black_data_dir), list_, [], black_data_dir)
    label = [1] * len(os.listdir(white_data_dir))
    label.extend([0] * len(os.listdir(black_data_dir)))

    df["data"] = list_
    df['label'] = label

    df = shuffle(df)
    print("有标注的数据量：", len(df))
    df.to_csv(train_data_path)
