import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from dl.pretrain_word2vec import word_embedding_path


class Config(object):
    """配置参数"""

    def __init__(self):

        self.word2vec_path = word_embedding_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备
        self.log_path = 'log'
        self.num_works = 16
        self.dropout = 0.2  # 随机失活
        self.save_path = 'BiLSTM.ckpt'
        self.require_improvement = 100000  # 若超过 xxxx batch效果还没提升，则提前结束训练
        self.num_classes = 2
        self.num_epochs = 1000  # epoch数
        self.batch_size = 128  # mini-batch大小
        self.pad_size = 30  # 每句话处理成的长度(短填长切)
        self.learning_rate = 1e-3  # 学习率
        self.embed = 300  # 字向量维度, 若使用了预训练词向量，则维度统一
        self.hidden_size = 128  # lstm隐藏层
        self.num_layers = 3  # lstm层数
        self.mid_fc_size = 64  # 最后的过渡全连接层神经元数量


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        weight_numpy = np.load(file=config.word2vec_path)

        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(weight_numpy))

        self.lstm = nn.LSTM(config.embed, config.hidden_size, config.num_layers,
                            bidirectional=True, batch_first=True, dropout=config.dropout)
        self.fc = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.mid_fc_size),
            nn.Dropout(p=config.dropout),
            nn.ReLU(),
            nn.Linear(config.mid_fc_size, config.num_classes),
        )

    def forward(self, x):
        # [batch_size, seq_len, embeding]=[128, 32, 300]
        out = self.embedding(x)
        out, _ = self.lstm(out)
        out = self.fc(out[:, -1, :])  # 句子最后时刻的 hidden state
        return out
