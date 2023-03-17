import os.path
import pickle
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from dl.pretrain_word2vec import word_vocab_path
from dl.train_data_process import train_data_path


class DiabetesDataset(Dataset):  # 继承Dataset类
    def __init__(self, is_train: bool, train_data_path=train_data_path, padding_size=2000):
        self.train_data_path = train_data_path
        self.npy_train_data_path = './data/train_data.npy'
        self.padding_size = padding_size
        f = open(word_vocab_path, 'rb')
        self.word_vocab = pickle.load(f)
        f.close()
        # self.raw_data = pd.read_csv("../raw_data_fre>100.csv")
        if os.path.isfile(self.npy_train_data_path):
            self.raw_data = np.load(self.npy_train_data_path, allow_pickle=True)
        else:
            self.raw_data = self.get_npy_data(self.npy_train_data_path)

        if is_train:
            self.raw_data = self.raw_data[:int(0.8 * len(self.raw_data))]
        else:
            self.raw_data = self.raw_data[int(0.8 * len(self.raw_data)):]

    def get_npy_data(self, npy_train_data_path):
        raw_data = pd.read_csv(self.train_data_path)
        words = raw_data[['data']].values
        labels = raw_data[['label']].values
        data = []
        for i in range(len(words)):
            string = words[i][0]
            try:
                print(labels[i][0], type(labels[i][0]))
                data.append([string.split(), labels[i][0]])
            except:
                continue
        data = np.array(data)
        np.save(npy_train_data_path, data)
        return data

    def __getitem__(self, index):

        data = self.raw_data[index][0]
        res = []
        length = len(data)
        if length > self.padding_size:
            data = data[:self.padding_size]
            self.words2idx(data, res)
        else:
            self.words2idx(data, res)
            res.extend([self.word_vocab['PAD']] * (self.padding_size - length))

        label = self.raw_data[index][1]
        return np.array(res), np.array(label)

    def words2idx(self, words, res):
        for i in words:
            if i in self.word_vocab:
                concat = self.word_vocab[i]
                res.append(concat)
            else:
                res.append(self.word_vocab['UNK'])

    def __len__(self):
        return len(self.raw_data)


if __name__ == '__main__':

    dataset = DiabetesDataset(is_train=True)
    train_loader = DataLoader(dataset=dataset, batch_size=128, shuffle=True, num_workers=4)
    for x, y in train_loader:
        pass
        # print(x.shape, y.shape)
        # print(type(x), type(y))
