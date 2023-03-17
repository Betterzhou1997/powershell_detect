import os
import re
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from get_pretrain_file_list import files_list_to_npy_path, root_path

dig_replace = re.compile(r'\d')
processed_pretrain_data_path = './data/processed_pretrain_data.csv'
processed_pretrain_data_filter_freq_path = './data/processed_pretrain_data_filter_freq.csv'

Freq = 100
Max_len = 2000


def data_preprocess(path_list, data_list_, name_, root_path=None):
    for idx, powershell_path in enumerate(path_list):
        if root_path:
            powershell_path = os.path.join(root_path, powershell_path)
        print(idx)
        try:
            with open(powershell_path, "rb") as fp:
                string = fp.read().decode("utf-8", errors="ignore").lower()
            string = re.sub(dig_replace, '*', string)
            raw_words = re.findall("[a-z0-9A-Z*$-]+", string)
            words_space = " ".join(w for w in raw_words if (2 < len(w) < 20))
            name_.append(powershell_path)
            data_list_.append(words_space)
        except:
            continue


def filter_word_frequency(freq, string_list, max_len=2000):
    """

    :param freq: 对于每一个词，只有它出现在大于freq个脚本中，才保留，不然就替换为 "UNK"
    :param string_list: 数据格式: [[string1],[string2],...]
    :param max_len: 只保留前多少个词，后续的截断
    :return:
    """

    word_dict = {}
    for i, string in enumerate(string_list):

        try:
            list_ = string.split()
            list_1 = set(list_)
            print(len(list_), len(list_1), i)
            for j in list_1:
                if j in word_dict:
                    word_dict[j] += 1
                else:
                    word_dict[j] = 1
        except:
            continue

    new_string_list = []
    for i, string in enumerate(string_list):
        print(i)
        try:
            list_ = string.split()
            new_string = " ".join([w if (word_dict[w] > freq) else 'UNK' for w in list_])
            new_string_list.append(new_string if len(new_string) <= max_len else new_string[:max_len])
        except:
            continue

    print('词频过滤之前样本数量：', len(string_list))
    print('词频过滤之后样本数量：', len(new_string_list))
    print('word_dict长度', len(word_dict))
    print(f'word_dict中词频大于{freq}的个数', len({w for w in word_dict if (word_dict[w] > freq)}))
    print('word_dict中词频最大的词出现次数', max([word_dict[i] for i in word_dict]))

    return new_string_list


if __name__ == '__main__':
    path_list = np.load(files_list_to_npy_path).tolist()
    print('检测到的路径列表数量：', len(path_list))

    pretrain_data_list = []
    name_ = []
    data_preprocess(path_list, pretrain_data_list, name_, root_path)
    print('能加载的文件数量：', len(pretrain_data_list))

    pretrain_data_list_filter_freq = filter_word_frequency(Freq, pretrain_data_list, max_len=Max_len)
    print(f'经过词频{Freq}过滤后预训练样本数量：', len(pretrain_data_list_filter_freq))

    # 分别保存
    df1 = pd.DataFrame()
    df1['ps_name'] = name_
    df1["pretrain_data"] = pretrain_data_list
    df1 = shuffle(df1)
    df1.to_csv(processed_pretrain_data_path)

    # 过滤词频以后保存到csv中
    df2 = pd.DataFrame()
    df2["pretrain_data_filter_freq"] = pretrain_data_list_filter_freq
    df2 = shuffle(df2)
    df2.to_csv(processed_pretrain_data_filter_freq_path)
