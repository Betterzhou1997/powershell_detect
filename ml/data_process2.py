import os
import re
import pandas as pd
import enchant as EC
import jieba
from sklearn.utils import shuffle

dict1 = EC.Dict("en_US")
dig_replace = re.compile(r'\d')


def detect_word_is_legel(word):
    import pkg_resources
    from symspellpy.symspellpy import SymSpell
    sym_spell = SymSpell(max_dictionary_edit_distance=0, prefix_length=7)
    dictionary_path = pkg_resources.resource_filename("symspellpy", "frequency_dictionary_en_82_765.txt")
    sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)

    result = sym_spell.word_segmentation(word)
    print("{}, {}, {}".format(result.corrected_string, result.distance_sum, result.log_prob_sum))
    return result.corrected_string, result.distance_sum, result.log_prob_sum


def filters(word):
    if dict1.check(word):
        return True
    suggest = dict1.suggest(word)
    return any([' ' in j or '-' in j for j in suggest])


def data_preprocess(data_dir, date_list_, name_):
    for idx, powershell_name in enumerate(os.listdir(data_dir)):
        powershell_path = os.path.join(data_dir, powershell_name)
        print(powershell_name, idx)
        with open(powershell_path, "rb") as fp:
            string = fp.read().decode("utf-8", errors="ignore").lower()
        # string = re.sub(dig_replace, '*', string)
        raw_words = re.findall("[a-z0-9A-Z*$-]+", string)

        words_space = " ".join(w for w in raw_words if (2 < len(w) < 20))
        name_.append(powershell_path)
        date_list_.append(words_space)


df = pd.DataFrame()
white_data_dir = '/home/jovyan/datasets/powershell/clean-powershell-white'
black_data_dir = '/home/jovyan/datasets/powershell/clean-powershell-black'

list_ = []
name_ = []
data_preprocess(white_data_dir, list_, name_)
data_preprocess(black_data_dir, list_, name_)
label = [1] * len(os.listdir(white_data_dir))
label.extend([0] * len(os.listdir(black_data_dir)))
df['name'] = name_
df["words"] = list_
df['label'] = label

df = shuffle(df)
df.to_csv('raw_data_microsoft_1.csv')
