import os
import numpy as np

root_path = r'/home/jovyan/datasets/powershell/PowerShellCorpus/'
files_list_to_npy_path = './data/pretrain_data_path_list.npy'


def get_all_sub_files_path(path_):
    total_path_list_ = []
    for root, dirs, files in os.walk(path_):
        for file in files:
            file_path = os.path.join(root.replace(path_, ''), file)
            total_path_list_.append(file_path)
    print(len(total_path_list_))
    return total_path_list_


if __name__ == '__main__':
    # 将预训练的文件路径加载到npy文件中
    total_path_list = get_all_sub_files_path(root_path)
    # 保存
    np_list = np.array(total_path_list)
    np.save(files_list_to_npy_path, total_path_list)
    # 加载出来看看
    total_path_list = np.load(files_list_to_npy_path).tolist()
    print(len(total_path_list))
