import os.path

import gensim
import numpy as np
import pandas as pd
from gensim.models.word2vec import Word2Vec
from pretrain_data_process import processed_pretrain_data_filter_freq_path

vector_size = 300
model_name = f'./models/30w_pretrain_{vector_size}dem.model'
txt_model_name = f'./models/30w_pretrain_{vector_size}dem.txt'
word_embedding_path = f'./models/word_embedding_{vector_size}dem.npy'
word_vocab_path = f'./models/word_vocab_{vector_size}dem.pkl'


def get_pretrain_data(csv_root):
    train_data_ = pd.read_csv(csv_root)
    print('预训练样本数量：', len(train_data_))
    value = train_data_.pretrain_data_filter_freq.values
    sentences_list_ = []
    print("开始处理成可以用于Word2Vec训练的数据...")
    for i, line in enumerate(value):
        print(i)
        try:
            # nltk.word_tokenize(line)
            new_line = line.split()
            sentences_list_.append(new_line)
        except:
            continue
    return sentences_list_


def pretrain(csv_root, model_name_, txt_model_name_, vector_size_, word_embedding_path_, word_vocab_path_):
    """

    :param word_vocab_path_: 
    :param word_embedding_path_: 
    :param csv_root:
    :param model_name_:
    :param txt_model_name_:
    :param vector_size_: Word vector dimensionality
    :return:
    """
    if not os.path.isfile(model_name_):
        sentences_list = get_pretrain_data(csv_root)
        # 设定词向量训练的参数
        num_workers = 8  # Number of threads to run in parallel
        context = 10  # Context window size
        model = Word2Vec(sentences_list, workers=num_workers, window=context, vector_size=vector_size_)
        # If you don't plan to train the model any further, calling
        # init_sims will make the model much more memory-efficient.
        model.init_sims(replace=True)
        # It can be helpful to create a meaningful model name and
        # save the model for later use. You can load it later using Word2Vec.load()
        # 保存为.model格式的模型
        model.save(model_name_)
    # 加载.model格式的模型，保存为TXT格式的，可以被pytorch加载
    model = Word2Vec.load(model_name_)
    if not os.path.isfile(txt_model_name_):
        model.wv.save_word2vec_format(txt_model_name_)
    print(model.wv.most_similar("low"))
    print(model.wv.most_similar("$true"))
    print(model.wv.most_similar("iex"))
    if not os.path.isfile(word_embedding_path_) or not os.path.isfile(word_vocab_path_):
        save_vocab(txt_model_name_, word_embedding_path_, word_vocab_path_)


def save_vocab(txt_model_name_, word_embedding_path, word_vocab_path):
    """

    :param txt_model_name_:
    :param word_embedding_path:
    :param word_vocab_path:
    :return:
    """
    wv_from_text = gensim.models.KeyedVectors.load_word2vec_format(txt_model_name_)
    vocab = wv_from_text.key_to_index
    print("Vocabulary Size: %s" % len(vocab.keys()))
    vector_size_ = wv_from_text.vector_size
    print("Vector Size: %s" % vector_size)
    word_embed = wv_from_text.vectors
    print("Embedding shape: {}".format(word_embed.shape))
    word_vocab = dict()
    word_vocab['PAD'] = 0
    word_vocab['UNK'] = 1
    for key in vocab.keys():
        word_vocab[key] = len(word_vocab.keys())
    unk_embed = np.random.randn(1, vector_size_)
    pad_embed = np.zeros(shape=(1, vector_size_), dtype=float)
    extral_embed = np.concatenate((pad_embed, unk_embed), axis=0)
    word_embed = np.concatenate((extral_embed, word_embed), axis=0)
    print("Embedding shape: {}".format(word_embed.shape))
    np.save(word_embedding_path, word_embed)
    pd.to_pickle(word_vocab, word_vocab_path)


if __name__ == '__main__':
    pretrain(processed_pretrain_data_filter_freq_path, model_name, txt_model_name,
             vector_size, word_embedding_path, word_vocab_path)
