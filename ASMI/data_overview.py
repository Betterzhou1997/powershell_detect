import os
black_dir = r'/home/jovyan/datasets/powershell/asmi_data_0303/black/content/'

for content in os.listdir(black_dir):
    content_path = os.path.join(black_dir, content, 'content')
    print('\n\n=======================================================')
    with open(content_path, 'r', encoding='UTF-8') as f:
        for line in f:
            if 'content' in line:
                print(line.strip())

